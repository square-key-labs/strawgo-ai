package switcher

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// waitFor polls condition up to timeout. Used to bridge the async frame
// pipeline -- collector.HandleFrame runs on a goroutine, so a synchronous
// snapshot right after emitError can race the frame's arrival.
func waitFor(t *testing.T, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	if !cond() {
		t.Fatalf("condition not met within %v", timeout)
	}
}

// fakeService is a minimal AIService for tests. It implements
// services.AIService via embedded BaseProcessor and exposes hooks the
// failover tests can manipulate (e.g. push an upstream ErrorFrame on
// demand).
type fakeService struct {
	*processors.BaseProcessor
	initCalls    atomic.Int32
	cleanupCalls atomic.Int32
	frameCount   atomic.Int32

	// initErr / cleanupErr force the lifecycle methods to return errors
	// for test paths that exercise failover's error handling.
	initErr    error
	cleanupErr error
}

func newFakeService(name string) *fakeService {
	f := &fakeService{}
	f.BaseProcessor = processors.NewBaseProcessor(name, f)
	return f
}

func (f *fakeService) Initialize(ctx context.Context) error {
	f.initCalls.Add(1)
	return f.initErr
}

func (f *fakeService) Cleanup() error {
	f.cleanupCalls.Add(1)
	return f.cleanupErr
}

func (f *fakeService) HandleFrame(_ context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	f.frameCount.Add(1)
	// Forward downstream so the failover sees the frame come "out" of us.
	return f.PushFrame(frame, direction)
}

// emitError pushes an ErrorFrame upstream as if the inner service had
// detected a failure. Optionally marks it fatal via metadata.
func (f *fakeService) emitError(fatal bool) error {
	ef := frames.NewErrorFrame(errors.New("boom"))
	if fatal {
		ef.SetMetadata(frames.MetadataKeyFatal, true)
	}
	return f.PushFrame(ef, frames.Upstream)
}

// upstreamCollector captures frames the Failover pushes upstream so the
// tests can inspect what propagated past the switcher.
type upstreamCollector struct {
	*processors.BaseProcessor
	mu     sync.Mutex
	frames []frames.Frame
}

func newUpstreamCollector() *upstreamCollector {
	c := &upstreamCollector{}
	c.BaseProcessor = processors.NewBaseProcessor("UpstreamCollector", c)
	return c
}

func (c *upstreamCollector) HandleFrame(_ context.Context, frame frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, frame)
	c.mu.Unlock()
	return nil
}

func (c *upstreamCollector) snapshot() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]frames.Frame, len(c.frames))
	copy(out, c.frames)
	return out
}

func setupFailover(t *testing.T, svcs []services.AIService) (*Failover, *upstreamCollector) {
	t.Helper()
	f := NewFailover(svcs)
	collector := newUpstreamCollector()
	// Wire collector as the Failover's upstream neighbor so PushFrame(Upstream)
	// lands in the collector.
	f.SetPrev(collector)

	ctx := context.Background()
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("collector Start: %v", err)
	}
	for _, svc := range svcs {
		if err := svc.Start(ctx); err != nil {
			t.Fatalf("service Start: %v", err)
		}
	}
	if err := f.Start(ctx); err != nil {
		t.Fatalf("failover Start: %v", err)
	}
	if err := f.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	t.Cleanup(func() {
		_ = f.Stop()
		for _, svc := range svcs {
			_ = svc.Stop()
		}
		_ = collector.Stop()
	})
	return f, collector
}

func TestFailoverInitOnlyActiveService(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")

	f, _ := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	if got := a.initCalls.Load(); got != 1 {
		t.Fatalf("expected A initialized once, got %d", got)
	}
	if got := b.initCalls.Load(); got != 0 {
		t.Fatalf("expected B not initialized initially, got %d", got)
	}
	if f.ActiveIndex() != 0 {
		t.Fatalf("expected active index 0, got %d", f.ActiveIndex())
	}
}

func TestFailoverSwitchesOnNonFatalError(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")

	var switched SwitchEvent
	gotSwitch := make(chan struct{}, 1)
	f := NewFailover([]services.AIService{a, b}, WithOnSwitch(func(ev SwitchEvent) {
		switched = ev
		select {
		case gotSwitch <- struct{}{}:
		default:
		}
	}))
	collector := newUpstreamCollector()
	f.SetPrev(collector)

	ctx := context.Background()
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("collector Start: %v", err)
	}
	if err := a.Start(ctx); err != nil {
		t.Fatalf("a Start: %v", err)
	}
	if err := b.Start(ctx); err != nil {
		t.Fatalf("b Start: %v", err)
	}
	if err := f.Start(ctx); err != nil {
		t.Fatalf("failover Start: %v", err)
	}
	if err := f.Initialize(ctx); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	t.Cleanup(func() {
		_ = f.Stop()
		_ = a.Stop()
		_ = b.Stop()
		_ = collector.Stop()
	})

	if err := a.emitError(false); err != nil {
		t.Fatalf("emit error: %v", err)
	}

	// Wait briefly for the switch event to propagate through the bridge.
	<-gotSwitch

	if f.ActiveIndex() != 1 {
		t.Fatalf("expected active index 1 after non-fatal failure, got %d", f.ActiveIndex())
	}
	if got := a.cleanupCalls.Load(); got != 1 {
		t.Fatalf("expected A cleanup once, got %d", got)
	}
	if got := b.initCalls.Load(); got != 1 {
		t.Fatalf("expected B initialized once after failover, got %d", got)
	}
	if switched.FromIndex != 0 || switched.ToIndex != 1 {
		t.Fatalf("switch event indices wrong: %+v", switched)
	}

	// The ErrorFrame should also have propagated upstream as informational.
	waitFor(t, 200*time.Millisecond, func() bool {
		for _, fr := range collector.snapshot() {
			if _, ok := fr.(*frames.ErrorFrame); ok {
				return true
			}
		}
		return false
	})
}

func TestFailoverDoesNotSwitchOnFatal(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")

	f, collector := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	if err := a.emitError(true); err != nil {
		t.Fatalf("emit error: %v", err)
	}

	if f.ActiveIndex() != 0 {
		t.Fatalf("expected active index unchanged on fatal, got %d", f.ActiveIndex())
	}
	if got := a.cleanupCalls.Load(); got != 0 {
		t.Fatalf("expected A NOT cleaned up on fatal, got %d", got)
	}
	if got := b.initCalls.Load(); got != 0 {
		t.Fatalf("expected B NOT initialized on fatal, got %d", got)
	}

	// Fatal ErrorFrame must still propagate upstream so the runner can act.
	waitFor(t, 200*time.Millisecond, func() bool {
		for _, fr := range collector.snapshot() {
			if ef, ok := fr.(*frames.ErrorFrame); ok && ef.IsFatal() {
				return true
			}
		}
		return false
	})
}

func TestFailoverExhaustedFallbacksPropagates(t *testing.T) {
	a := newFakeService("A")

	f, collector := setupFailover(t, []services.AIService{a})
	defer f.Cleanup()

	if err := a.emitError(false); err != nil {
		t.Fatalf("emit error: %v", err)
	}

	if f.ActiveIndex() != 0 {
		t.Fatalf("expected active index unchanged when no fallback, got %d", f.ActiveIndex())
	}

	// When no fallback remains, the ErrorFrame is escalated to fatal so
	// the pipeline runner stops pretending the failure is recoverable.
	waitFor(t, 200*time.Millisecond, func() bool {
		for _, fr := range collector.snapshot() {
			if ef, ok := fr.(*frames.ErrorFrame); ok && ef.IsFatal() {
				return true
			}
		}
		return false
	})
}

func TestFailoverIgnoresStaleErrorFromInactive(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")

	f, _ := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	// First failure: A -> B. Now B is active.
	if err := a.emitError(false); err != nil {
		t.Fatalf("emit error from A: %v", err)
	}
	waitFor(t, 200*time.Millisecond, func() bool { return f.ActiveIndex() == 1 })

	// A emits another non-fatal error (stale). Active is B; we must NOT
	// switch beyond B just because the dead service is still grumbling.
	beforeB := b.cleanupCalls.Load()
	if err := a.emitError(false); err != nil {
		t.Fatalf("emit stale error from A: %v", err)
	}
	// Give the bridge a moment to process; nothing should switch.
	time.Sleep(50 * time.Millisecond)
	if f.ActiveIndex() != 1 {
		t.Fatalf("expected active index 1 after stale error, got %d", f.ActiveIndex())
	}
	if got := b.cleanupCalls.Load(); got != beforeB {
		t.Fatalf("B should not have been Cleanup-ed by stale-A error; calls before=%d after=%d", beforeB, got)
	}
}

func TestFailoverFallbackInitFailureKeepsActive(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")
	b.initErr = errors.New("dial failed")

	f, _ := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	if err := a.emitError(false); err != nil {
		t.Fatalf("emit error: %v", err)
	}
	// Give the switch attempt a beat to run and fail.
	time.Sleep(50 * time.Millisecond)

	if f.ActiveIndex() != 0 {
		t.Fatalf("expected active index unchanged when fallback init fails, got %d", f.ActiveIndex())
	}
	if got := a.cleanupCalls.Load(); got != 1 {
		t.Fatalf("expected A cleanup attempted (got %d) -- failover always tries to clean the failed service before init", got)
	}
}

func TestFailoverCleanupErrorStillSwitches(t *testing.T) {
	a := newFakeService("A")
	a.cleanupErr = errors.New("cleanup hiccup")
	b := newFakeService("B")

	f, _ := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	if err := a.emitError(false); err != nil {
		t.Fatalf("emit error: %v", err)
	}
	waitFor(t, 200*time.Millisecond, func() bool { return f.ActiveIndex() == 1 })

	if got := b.initCalls.Load(); got != 1 {
		t.Fatalf("expected B initialized once, got %d", got)
	}
}

func TestFailoverConcurrentRouteVsSwitch(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")

	f, _ := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	// Spam HandleFrame on one goroutine while another fires the failover
	// trigger. With switchMu, the in-flight HandleFrame on A finishes
	// before A's Cleanup runs, and the cleanup is observable atomically
	// from outside. The race detector enforces correctness.
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = f.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream)
			}
		}
	}()

	// Wait for some traffic to land, then trigger failover.
	time.Sleep(20 * time.Millisecond)
	if err := a.emitError(false); err != nil {
		t.Fatalf("emit error: %v", err)
	}
	waitFor(t, 200*time.Millisecond, func() bool { return f.ActiveIndex() == 1 })
	close(stop)
	wg.Wait()

	if got := a.cleanupCalls.Load(); got != 1 {
		t.Fatalf("expected A cleanup once, got %d", got)
	}
}

func TestFailoverRoutesIncomingToActive(t *testing.T) {
	a := newFakeService("A")
	b := newFakeService("B")

	f, _ := setupFailover(t, []services.AIService{a, b})
	defer f.Cleanup()

	// Send a frame in -- only the active service (A) should see it.
	if err := f.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}

	if got := a.frameCount.Load(); got != 1 {
		t.Fatalf("expected A to receive 1 frame, got %d", got)
	}
	if got := b.frameCount.Load(); got != 0 {
		t.Fatalf("expected B (dormant) to receive 0 frames, got %d", got)
	}
}
