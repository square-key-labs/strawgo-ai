package transports

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// BotSpeechTracker emits BotStartedSpeakingFrame and
// BotStoppedSpeakingFrame upstream as the audio path's bot-speech
// state changes. It subscribes to the Pacer's lifecycle events and to
// a PlaybackAckTracker for the "is playback complete" signal.
//
// Lifecycle:
//   - PacerStarted → emit BotStartedSpeakingFrame, botSpeaking=true.
//   - PacerIdle, after the LLM has signalled end → ask the
//     PlaybackAckTracker to resolve completion. When ack tracker's
//     Done() fires (immediate for local, after ack for telephony,
//     after fallback for stuck flows), emit BotStoppedSpeakingFrame.
//   - InterruptionFrame → emit BotStoppedSpeakingFrame if speaking,
//     cancel any pending ack request.
//
// The LLM-end gate (LLMResponseEnded) is a deliberate seam: the bot
// should not stop speaking before the LLM has finished generating,
// even if the audio path momentarily idles between TTS chunks.
//
// Phase 3 #44 lifted this state machine out of WebSocketOutputProcessor.
type BotSpeechTracker struct {
	pushFrame  func(frame frames.Frame, direction frames.FrameDirection) error
	ackTracker *PlaybackAckTracker
	log        *logger.Logger

	pacerEvents <-chan PacerEvent

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// llmResponseEnded gates the PacerIdle → completion path. Touched
	// from HandleFrame (writer) and the run goroutine (reader); a
	// regular mutex is enough since reads are infrequent.
	llmMu            sync.Mutex
	llmResponseEnded bool

	// interruptCh signals the run loop to emit a stop frame and reset
	// botSpeaking, regardless of pacer/ack state.
	interruptCh chan struct{}

	stopped atomic.Bool
}

// NewBotSpeechTracker constructs and starts a tracker. Stop must be
// called for clean shutdown.
func NewBotSpeechTracker(
	pacerEvents <-chan PacerEvent,
	ackTracker *PlaybackAckTracker,
	pushFrame func(frame frames.Frame, direction frames.FrameDirection) error,
	log *logger.Logger,
) *BotSpeechTracker {
	if log == nil {
		log = logger.WithPrefix("BotSpeechTracker")
	}
	t := &BotSpeechTracker{
		pushFrame:   pushFrame,
		ackTracker:  ackTracker,
		log:         log,
		pacerEvents: pacerEvents,
		interruptCh: make(chan struct{}, 1),
	}
	t.ctx, t.cancel = context.WithCancel(context.Background())
	t.wg.Add(1)
	go t.run()
	return t
}

// LLMResponseEnded marks the LLM as finished generating. Called from
// HandleFrame on LLMFullResponseEndFrame.
func (t *BotSpeechTracker) LLMResponseEnded() {
	t.llmMu.Lock()
	t.llmResponseEnded = true
	t.llmMu.Unlock()
}

// ResetLLMState clears the LLM-end gate. Called from HandleFrame on
// TTSStartedFrame so the next utterance starts with a fresh gate.
func (t *BotSpeechTracker) ResetLLMState() {
	t.llmMu.Lock()
	t.llmResponseEnded = false
	t.llmMu.Unlock()
}

// Interrupt signals the tracker to emit BotStoppedSpeakingFrame (if
// currently speaking) and clear in-flight ack state. Called from
// HandleFrame on InterruptionFrame.
func (t *BotSpeechTracker) Interrupt() {
	select {
	case t.interruptCh <- struct{}{}:
	default:
	}
}

// Stop terminates the run loop. Idempotent.
func (t *BotSpeechTracker) Stop() {
	if !t.stopped.CompareAndSwap(false, true) {
		return
	}
	if t.cancel != nil {
		t.cancel()
	}
	t.wg.Wait()
}

func (t *BotSpeechTracker) run() {
	defer t.wg.Done()

	botSpeaking := false

	for {
		select {
		case <-t.ctx.Done():
			t.log.Debug("BotSpeechTracker stopped")
			return

		case ev := <-t.pacerEvents:
			switch ev {
			case PacerStarted:
				if !botSpeaking {
					t.log.Info("Bot started speaking")
					t.pushFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
					botSpeaking = true
				}

			case PacerIdle:
				if !botSpeaking {
					continue
				}
				t.llmMu.Lock()
				llmEnded := t.llmResponseEnded
				t.llmMu.Unlock()
				if !llmEnded {
					t.log.Debug("Pacer idle but LLM still generating, waiting...")
					continue
				}
				t.ackTracker.RequestComplete()
			}

		case <-t.ackTracker.Done():
			if botSpeaking {
				t.log.Info("Playback complete; emitting BotStoppedSpeakingFrame")
				t.pushFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
				botSpeaking = false
			}

		case <-t.interruptCh:
			t.ackTracker.Cancel()
			if botSpeaking {
				t.log.Debug("Interrupt: emitting BotStoppedSpeakingFrame")
				t.pushFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
				botSpeaking = false
			}
		}
	}
}
