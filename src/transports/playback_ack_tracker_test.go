package transports

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type stubAckSerializer struct {
	failNext atomic.Bool
}

func (s *stubAckSerializer) Type() serializers.SerializerType { return serializers.SerializerTypeText }
func (s *stubAckSerializer) Setup(frames.Frame) error         { return nil }
func (s *stubAckSerializer) Serialize(frames.Frame) (interface{}, error) {
	return nil, nil
}
func (s *stubAckSerializer) Deserialize(interface{}) (frames.Frame, error) { return nil, nil }
func (s *stubAckSerializer) Cleanup() error                                { return nil }
func (s *stubAckSerializer) SerializePlaybackDoneAck(correlationID string) (interface{}, error) {
	if s.failNext.CompareAndSwap(true, false) {
		return nil, nil
	}
	return "ack:" + correlationID, nil
}

type capturedSend struct {
	mu      sync.Mutex
	count   atomic.Int32
	last    interface{}
	matched chan string
}

func newCapturedSend() *capturedSend {
	return &capturedSend{matched: make(chan string, 8)}
}

func (c *capturedSend) send(data interface{}) error {
	c.mu.Lock()
	c.last = data
	c.mu.Unlock()
	c.count.Add(1)
	if s, ok := data.(string); ok {
		select {
		case c.matched <- s:
		default:
		}
	}
	return nil
}

type strategyResolver struct{ s atomic.Int32 }

func (r *strategyResolver) get() playbackStrategy { return playbackStrategy(r.s.Load()) }
func (r *strategyResolver) set(s playbackStrategy) { r.s.Store(int32(s)) }

func newAckTracker(t *testing.T, s serializers.FrameSerializer, resolver *strategyResolver, fallback time.Duration, drainPad time.Duration) (*PlaybackAckTracker, *capturedSend) {
	t.Helper()
	cs := newCapturedSend()
	tr := NewPlaybackAckTracker(playbackTransportFns{
		send:             cs.send,
		serializer:       s,
		resolveStrategy:  resolver.get,
		drainPad:         func() time.Duration { return drainPad },
		fallbackDuration: fallback,
	})
	t.Cleanup(func() { tr.Stop() })
	return tr, cs
}

func waitDone(t *testing.T, tr *PlaybackAckTracker, timeout time.Duration) {
	t.Helper()
	select {
	case <-tr.Done():
	case <-time.After(timeout):
		t.Fatal("timed out waiting for ack tracker Done()")
	}
}

func assertNoDone(t *testing.T, tr *PlaybackAckTracker, dur time.Duration) {
	t.Helper()
	select {
	case <-tr.Done():
		t.Fatal("unexpected Done() signal")
	case <-time.After(dur):
	}
}

func TestPlaybackAckTracker_LocalCompletesImmediately(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratLocal)
	tr, _ := newAckTracker(t, &mockSerializer{}, r, time.Second, 0)

	tr.RequestComplete()
	waitDone(t, tr, 100*time.Millisecond)
}

func TestPlaybackAckTracker_SerializerAckRequestsAndCompletesOnNotify(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratSerializerAck)
	tr, cs := newAckTracker(t, &stubAckSerializer{}, r, time.Second, 0)

	tr.RequestComplete()

	// Wait for the ack request to be sent.
	var corr string
	select {
	case s := <-cs.matched:
		corr = s[len("ack:"):]
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for ack request send")
	}

	// Done() should not fire yet — we haven't notified.
	assertNoDone(t, tr, 50*time.Millisecond)

	tr.Notify(corr)
	waitDone(t, tr, 100*time.Millisecond)
}

func TestPlaybackAckTracker_SerializerAckIgnoresMismatchedNotify(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratSerializerAck)
	tr, cs := newAckTracker(t, &stubAckSerializer{}, r, time.Second, 0)

	tr.RequestComplete()
	<-cs.matched

	tr.Notify("wrong-correlation-id")
	assertNoDone(t, tr, 50*time.Millisecond)
}

func TestPlaybackAckTracker_FallbackFiresWhenAckLost(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratSerializerAck)
	tr, cs := newAckTracker(t, &stubAckSerializer{}, r, 80*time.Millisecond, 0)

	tr.RequestComplete()
	<-cs.matched

	// Don't notify — fallback should fire.
	waitDone(t, tr, 200*time.Millisecond)
}

func TestPlaybackAckTracker_UserTriggerCompletes(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratUserAck)
	tr, _ := newAckTracker(t, &mockSerializer{}, r, time.Second, 0)

	tr.RequestComplete()
	assertNoDone(t, tr, 30*time.Millisecond)

	tr.UserTrigger()
	waitDone(t, tr, 100*time.Millisecond)
}

func TestPlaybackAckTracker_UserTriggerIgnoredOutsideUserAckMode(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratSerializerAck)
	tr, cs := newAckTracker(t, &stubAckSerializer{}, r, 200*time.Millisecond, 0)

	tr.RequestComplete()
	<-cs.matched

	tr.UserTrigger()
	assertNoDone(t, tr, 50*time.Millisecond)
}

func TestPlaybackAckTracker_DrainPadCompletesAfterDelay(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratDrainPad)
	tr, _ := newAckTracker(t, &mockSerializer{}, r, time.Second, 30*time.Millisecond)

	tr.RequestComplete()
	assertNoDone(t, tr, 10*time.Millisecond)
	waitDone(t, tr, 100*time.Millisecond)
}

func TestPlaybackAckTracker_CancelDropsPending(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratSerializerAck)
	tr, cs := newAckTracker(t, &stubAckSerializer{}, r, 200*time.Millisecond, 0)

	tr.RequestComplete()
	corr := <-cs.matched
	corrID := corr[len("ack:"):]

	tr.Cancel()

	// Late notify with the canceled request's correlation ID must not
	// fire Done().
	tr.Notify(corrID)
	assertNoDone(t, tr, 80*time.Millisecond)

	// Cancel must not block subsequent requests.
	tr.RequestComplete()
	select {
	case <-cs.matched:
	case <-time.After(100 * time.Millisecond):
		t.Fatal("RequestComplete after Cancel did not re-engage")
	}
}

func TestPlaybackAckTracker_SerializerAckFallsBackOnSendError(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratSerializerAck)
	failingSerializer := &stubAckSerializer{}
	failingSerializer.failNext.Store(true)
	tr, _ := newAckTracker(t, failingSerializer, r, time.Second, 0)

	tr.RequestComplete()
	// Should emit Done synchronously because the ack send returned nil
	// data (treated as unavailable).
	waitDone(t, tr, 100*time.Millisecond)
}

func TestPlaybackAckTracker_StopIsIdempotent(t *testing.T) {
	r := &strategyResolver{}
	r.set(stratLocal)
	tr, _ := newAckTracker(t, &mockSerializer{}, r, time.Second, 0)

	tr.Stop()
	tr.Stop() // must not panic
}
