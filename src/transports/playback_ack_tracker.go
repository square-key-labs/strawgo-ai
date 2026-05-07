package transports

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

// PlaybackAckTracker resolves "playback is complete on the client" for the
// output transport. The shape of completion depends on the
// playbackStrategy in force at request time:
//
//   - stratLocal: complete is synchronous (transport is an in-process
//     speaker; send-complete == play-complete).
//   - stratUserAck: wait for the application's TriggerPlaybackComplete
//     call (the user-ack sentinel).
//   - stratSerializerAck: ask the serializer to emit a playback-ack
//     request, then wait for the matching PlaybackCompleteFrame from
//     the client (Twilio mark echo, Asterisk QUEUE_DRAINED).
//   - stratDrainPad: schedule a delayed completion using the
//     transport-configured drain pad.
//
// In all async cases a fallback timer fires after fallbackDuration to
// keep the session from wedging when an ack is lost. The result —
// either a real completion or the fallback — is signalled by closing
// (or sending on) the channel returned by Done().
//
// Phase 3 #44 lifted this state machine out of WebSocketOutputProcessor.
type PlaybackAckTracker struct {
	sendFn           func(data interface{}) error
	serializer       serializers.FrameSerializer
	resolveStrategy  func() playbackStrategy
	drainPadFunc     func() time.Duration
	fallbackDuration time.Duration
	log              *logger.Logger

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Channels driving the run loop
	requestCh   chan struct{}
	notifyCh    chan string   // PlaybackCompleteFrame correlation IDs
	userTrigger chan struct{} // TriggerPlaybackComplete sentinel
	cancelCh    chan struct{} // BotSpeechTracker abort (interrupt)

	// Outbound: closed-channel-style emission of the next completion
	// signal. The signal is a string correlation ID (or an empty string
	// for the fallback path). A buffered slot of 1 lets the producer
	// run without blocking on the consumer.
	doneCh chan struct{}

	// stopped guards Stop against double-close.
	stopped atomic.Bool
}

// playbackTransportFns is the minimal contract NewPlaybackAckTracker
// needs from its host transport.
type playbackTransportFns struct {
	send             func(data interface{}) error
	serializer       serializers.FrameSerializer
	resolveStrategy  func() playbackStrategy
	drainPad         func() time.Duration
	fallbackDuration time.Duration
	log              *logger.Logger
}

// NewPlaybackAckTracker constructs and starts a tracker. Stop must be
// called for clean shutdown.
func NewPlaybackAckTracker(d playbackTransportFns) *PlaybackAckTracker {
	if d.log == nil {
		d.log = logger.WithPrefix("PlaybackAckTracker")
	}
	t := &PlaybackAckTracker{
		sendFn:           d.send,
		serializer:       d.serializer,
		resolveStrategy:  d.resolveStrategy,
		drainPadFunc:     d.drainPad,
		fallbackDuration: d.fallbackDuration,
		log:              d.log,
		requestCh:        make(chan struct{}, 1),
		notifyCh:         make(chan string, 8),
		userTrigger:      make(chan struct{}, 1),
		cancelCh:         make(chan struct{}, 1),
		doneCh:           make(chan struct{}, 1),
	}
	t.ctx, t.cancel = context.WithCancel(context.Background())
	t.wg.Add(1)
	go t.run()
	return t
}

// RequestComplete asks the tracker to resolve the next playback
// completion. Call when the audio path goes idle and the LLM has
// finished. Subsequent RequestComplete calls before the previous
// resolves are coalesced.
func (t *PlaybackAckTracker) RequestComplete() {
	select {
	case t.requestCh <- struct{}{}:
	default:
	}
}

// Done returns a channel that receives a value once playback completion
// resolves (real ack, drain pad, fallback, or stratLocal immediate).
// One value per RequestComplete; Cancel drops any in-flight request
// without emitting on Done.
func (t *PlaybackAckTracker) Done() <-chan struct{} { return t.doneCh }

// Notify forwards an inbound PlaybackCompleteFrame's correlation ID.
// Drops on overflow; the run loop coalesces matching IDs.
func (t *PlaybackAckTracker) Notify(correlationID string) {
	select {
	case t.notifyCh <- correlationID:
	default:
	}
}

// UserTrigger forwards an application-side TriggerPlaybackComplete call.
// Only meaningful when stratUserAck is in effect at request time.
func (t *PlaybackAckTracker) UserTrigger() {
	select {
	case t.userTrigger <- struct{}{}:
	default:
	}
}

// Cancel aborts any in-flight request without emitting on Done. Use on
// InterruptionFrame to clear pending state cleanly.
func (t *PlaybackAckTracker) Cancel() {
	select {
	case t.cancelCh <- struct{}{}:
	default:
	}
}

// Stop terminates the run loop. Idempotent.
func (t *PlaybackAckTracker) Stop() {
	if !t.stopped.CompareAndSwap(false, true) {
		return
	}
	if t.cancel != nil {
		t.cancel()
	}
	t.wg.Wait()
}

func (t *PlaybackAckTracker) run() {
	defer t.wg.Done()

	var pendingCorrelationID string
	var fallbackTimer *time.Timer
	var fallbackTimerC <-chan time.Time

	defer func() {
		if fallbackTimer != nil {
			fallbackTimer.Stop()
		}
	}()

	armFallback := func() {
		if fallbackTimer != nil {
			fallbackTimer.Stop()
		}
		fallbackTimer = time.NewTimer(t.fallbackDuration)
		fallbackTimerC = fallbackTimer.C
	}

	clearFallback := func() {
		if fallbackTimer != nil {
			fallbackTimer.Stop()
			fallbackTimer = nil
			fallbackTimerC = nil
		}
	}

	emitDone := func() {
		select {
		case t.doneCh <- struct{}{}:
		default:
			t.log.Debug("doneCh full; consumer not keeping up")
		}
	}

	hasPending := func() bool { return pendingCorrelationID != "" }

	for {
		select {
		case <-t.ctx.Done():
			return

		case <-t.requestCh:
			if hasPending() {
				t.log.Debug("RequestComplete coalesced: previous request still pending (%s)", pendingCorrelationID)
				continue
			}

			switch t.resolveStrategy() {
			case stratLocal:
				t.log.Info("Playback complete (local transport, immediate)")
				emitDone()

			case stratUserAck:
				t.log.Info("Awaiting user playback-complete trigger (fallback in %v)", t.fallbackDuration)
				pendingCorrelationID = correlationUserAck
				armFallback()

			case stratSerializerAck:
				ackSer, ok := t.serializer.(serializers.PlaybackAckSerializer)
				if !ok {
					t.log.Warn("Strategy says serializer-ack but serializer does not implement PlaybackAckSerializer; emitting complete")
					emitDone()
					continue
				}
				correlationID := fmt.Sprintf("playback-%d", time.Now().UnixNano())
				data, err := ackSer.SerializePlaybackDoneAck(correlationID)
				if err != nil || data == nil {
					t.log.Warn("Playback-done ack unavailable (err=%v); emitting complete", err)
					emitDone()
					continue
				}
				if sendErr := t.sendFn(data); sendErr != nil {
					t.log.Warn("Failed to send playback-done ack (%v); emitting complete", sendErr)
					emitDone()
					continue
				}
				t.log.Info("Sent playback-done ack request (fallback in %v)", t.fallbackDuration)
				pendingCorrelationID = correlationID
				armFallback()

			case stratDrainPad:
				pad := t.drainPadFunc()
				t.log.Info("Awaiting drain pad %v then complete (fallback in %v)", pad, t.fallbackDuration)
				pendingCorrelationID = correlationDrainPad
				notify := t.notifyCh
				time.AfterFunc(pad, func() {
					select {
					case notify <- correlationDrainPad:
					default:
					}
				})
				armFallback()
			}

		case correlationID := <-t.notifyCh:
			if !hasPending() {
				t.log.Debug("Ignoring playback completion (no pending request, got %q)", correlationID)
				continue
			}
			if pendingCorrelationID != "" && correlationID != "" && correlationID != pendingCorrelationID {
				t.log.Debug("Ignoring stale playback completion (got %s, waiting for %s)",
					correlationID, pendingCorrelationID)
				continue
			}
			t.log.Info("Playback complete (client ack received)")
			clearFallback()
			pendingCorrelationID = ""
			emitDone()

		case <-t.userTrigger:
			if pendingCorrelationID != correlationUserAck {
				t.log.Debug("UserTrigger ignored (pending=%q)", pendingCorrelationID)
				continue
			}
			t.log.Info("Playback complete (user trigger)")
			clearFallback()
			pendingCorrelationID = ""
			emitDone()

		case <-fallbackTimerC:
			if !hasPending() {
				continue
			}
			t.log.Warn("Playback-done fallback fired (no client ack in %v)", t.fallbackDuration)
			fallbackTimer = nil
			fallbackTimerC = nil
			pendingCorrelationID = ""
			emitDone()

		case <-t.cancelCh:
			if hasPending() {
				t.log.Debug("Pending playback request canceled (was %s)", pendingCorrelationID)
			}
			clearFallback()
			pendingCorrelationID = ""
			// Drain any stale notify so a late ack from the canceled
			// request doesn't get matched against the next one.
			for {
				select {
				case <-t.notifyCh:
				default:
					goto cancelDone
				}
			}
		cancelDone:
		}
	}
}
