// Package switcher provides composite processors that route frames across
// multiple interchangeable AIService instances.
package switcher

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// SwitchEvent describes a failover transition. Passed to OnSwitch handlers.
type SwitchEvent struct {
	FromIndex int
	ToIndex   int
	From      services.AIService
	To        services.AIService
	Error     *frames.ErrorFrame
}

// Failover wraps a list of AIService instances and routes the active one's
// frames into the pipeline. When the active service emits a non-fatal
// ErrorFrame upstream, the Failover advances to the next service, calls
// Cleanup on the failed one, Initialize on the new one, and propagates the
// ErrorFrame upstream (informational). The pipeline keeps moving on the new
// service starting with the *next* incoming frame.
//
// The Failover does NOT replay frames the failed service had already started
// processing. Downstream consumers may have observed partial output (e.g.
// half a TTS utterance), so a replay would double up. Pipecat
// "service_switcher_strategy_failover" follows the same contract.
//
// Fatal ErrorFrames (per ErrorFrame.IsFatal) are propagated unchanged with
// no switch — they signal pipeline-level failure that the runner is
// expected to handle (typically by terminating).
//
// Concurrency: HandleFrame, the inner-frame bridge, and lifecycle methods
// can run on different goroutines. activeIdx is atomic, and switching is
// serialized via switchMu so two concurrent error frames cannot both
// initialize the next service.
type Failover struct {
	*processors.BaseProcessor

	services []services.AIService

	// activeIdx is the index of the AIService currently receiving frames.
	// Read atomically on the hot path; advanced under switchMu during failover.
	activeIdx atomic.Int32

	// switchMu serializes the failover routine itself so two concurrent
	// non-fatal ErrorFrames cannot each Cleanup+Initialize past each other.
	// It deliberately does NOT cover HandleFrame's call into the inner
	// service: ProcessFrame is synchronous and the inner service may push
	// an upstream ErrorFrame back through frameBridge into handleErrorFrame
	// on the same goroutine, so holding switchMu around ProcessFrame would
	// deadlock when handleErrorFrame tries to take it. Concurrent
	// HandleFrame vs. switch-driven Cleanup is safe because each inner
	// service's own concurrency model (e.g. STT lifecycleMu, LLM streamMu)
	// already serializes ProcessFrame against its own Cleanup.
	switchMu sync.Mutex

	// ctxMu/ctx track the most-recent context the Failover has seen so a
	// switch triggered from an error frame can re-initialize the next
	// service with a still-live context. Mirrors reconnect/wrap.go.
	ctxMu sync.RWMutex
	ctx   context.Context

	onSwitch func(SwitchEvent)

	log *logger.Logger
}

// Option configures a Failover.
type Option func(*Failover)

// WithOnSwitch registers a callback that fires after a successful service
// switch. The handler runs on the goroutine that processed the error frame;
// it should not block.
func WithOnSwitch(fn func(SwitchEvent)) Option {
	return func(f *Failover) {
		f.onSwitch = fn
	}
}

// NewFailover wraps the provided AIServices in a failover composite. The
// first service in the slice is the initial active. The composite routes
// incoming frames to the active service and intercepts its upstream output
// via internal frame bridges.
//
// At least one service must be provided.
func NewFailover(svcs []services.AIService, opts ...Option) *Failover {
	if len(svcs) == 0 {
		panic("switcher: NewFailover requires at least one service")
	}

	f := &Failover{
		services: svcs,
		log:      logger.WithPrefix("ServiceSwitcherFailover"),
	}
	f.BaseProcessor = processors.NewBaseProcessor("ServiceSwitcherFailover", f)

	for i, svc := range svcs {
		svc.SetPrev(&frameBridge{
			owner:      f,
			direction:  frames.Upstream,
			serviceIdx: i,
			name:       "ServiceSwitcherUpstreamBridge",
		})
		svc.Link(&frameBridge{
			owner:      f,
			direction:  frames.Downstream,
			serviceIdx: i,
			name:       "ServiceSwitcherDownstreamBridge",
		})
	}

	for _, opt := range opts {
		opt(f)
	}
	return f
}

// Active returns the AIService currently receiving frames.
func (f *Failover) Active() services.AIService {
	return f.services[f.activeIdx.Load()]
}

// ActiveIndex returns the index of the active service.
func (f *Failover) ActiveIndex() int {
	return int(f.activeIdx.Load())
}

// Initialize initializes the active service only. Other services stay
// dormant until the failover path activates them. Mirrors pipecat's
// failover lifecycle (lazy init for non-active services).
func (f *Failover) Initialize(ctx context.Context) error {
	f.setContext(ctx)
	return f.Active().Initialize(ctx)
}

// Cleanup tears down whichever service is currently active. Services that
// were swapped out earlier already had Cleanup called as part of the switch
// path, so calling Cleanup again on them is the inner service's concern.
// switchMu serializes against an in-flight handleErrorFrame so a switch
// cannot race with shutdown and leave a freshly-Initialized fallback dangling.
func (f *Failover) Cleanup() error {
	f.switchMu.Lock()
	defer f.switchMu.Unlock()
	return f.services[f.activeIdx.Load()].Cleanup()
}

// Start tracks the runner's context for use by error-driven re-init, then
// starts the BaseProcessor's frame loop.
func (f *Failover) Start(ctx context.Context) error {
	f.setContext(ctx)
	return f.BaseProcessor.Start(ctx)
}

// HandleFrame forwards to the active inner service. The inner service's
// upstream/downstream output is captured by frameBridge and routed back
// through handleInnerFrame. activeIdx is read atomically (no lock):
// holding switchMu here would deadlock the same goroutine when the inner
// service synchronously emits an upstream ErrorFrame that re-enters
// handleErrorFrame.
func (f *Failover) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	f.setContext(ctx)
	active := f.services[f.activeIdx.Load()]
	return active.ProcessFrame(ctx, frame, direction)
}

// handleInnerFrame is invoked for every frame an inner service emits.
// Upstream ErrorFrames trigger the failover decision; everything else is
// forwarded onto the Failover's own pipeline neighbors unchanged.
// fromIdx records which inner service produced the frame so a stale
// ErrorFrame from a service we already swapped away from cannot
// re-trigger another switch.
func (f *Failover) handleInnerFrame(frame frames.Frame, direction frames.FrameDirection, fromIdx int) error {
	if errFrame, ok := frame.(*frames.ErrorFrame); ok && direction == frames.Upstream {
		return f.handleErrorFrame(errFrame, fromIdx)
	}
	return f.PushFrame(frame, direction)
}

// handleErrorFrame applies the failover decision: fatal -> propagate
// untouched; non-fatal -> Cleanup current, Initialize next, propagate
// the same ErrorFrame upstream as informational signal. If no fallback
// remains, the ErrorFrame is escalated to fatal so the runner can act
// (it would otherwise loop forever asking the same dead service for
// frames).
func (f *Failover) handleErrorFrame(errFrame *frames.ErrorFrame, fromIdx int) error {
	if errFrame.IsFatal() {
		return f.PushFrame(errFrame, frames.Upstream)
	}

	f.switchMu.Lock()
	defer f.switchMu.Unlock()

	cur := int(f.activeIdx.Load())
	if fromIdx != cur {
		// Stale error from a previously-active service. Don't switch on
		// behalf of someone who isn't current, and DON'T propagate either:
		// a flapping dead service would otherwise flood upstream
		// consumers with phantom errors. Pipecat's failover does the
		// same gating in service_switcher.py.
		f.log.Debug("Dropping stale ErrorFrame from service %d (active is %d)", fromIdx, cur)
		return nil
	}

	next := cur + 1
	if next >= len(f.services) {
		// No fallback remains. Escalate to fatal so the pipeline runner
		// stops pretending this is recoverable.
		f.log.Warn("Active service %d emitted non-fatal error and no fallback remains; escalating to fatal: %v", cur, errFrame.Error)
		errFrame.SetMetadata(frames.MetadataKeyFatal, true)
		return f.PushFrame(errFrame, frames.Upstream)
	}

	from := f.services[cur]
	to := f.services[next]

	f.log.Warn("Active service %d failed, switching to service %d: %v", cur, next, errFrame.Error)

	if cleanupErr := from.Cleanup(); cleanupErr != nil {
		f.log.Warn("Cleanup of failed service %d returned: %v", cur, cleanupErr)
	}

	if initErr := to.Initialize(f.reinitContext()); initErr != nil {
		f.log.Error("Initialize of fallback service %d failed: %v", next, initErr)
		// Don't advance activeIdx — surface the original error and let the
		// next inbound frame retry via lazy init or trigger another switch.
		return f.PushFrame(errFrame, frames.Upstream)
	}

	f.activeIdx.Store(int32(next))

	if f.onSwitch != nil {
		f.onSwitch(SwitchEvent{
			FromIndex: cur,
			ToIndex:   next,
			From:      from,
			To:        to,
			Error:     errFrame,
		})
	}

	return f.PushFrame(errFrame, frames.Upstream)
}

func (f *Failover) reinitContext() context.Context {
	f.ctxMu.RLock()
	ctx := f.ctx
	f.ctxMu.RUnlock()
	if ctx == nil {
		return context.Background()
	}
	return ctx
}

func (f *Failover) setContext(ctx context.Context) {
	if ctx == nil {
		return
	}
	f.ctxMu.Lock()
	f.ctx = ctx
	f.ctxMu.Unlock()
}

// frameBridge is the placeholder processor inserted as each inner service's
// upstream/downstream neighbor. It is intentionally minimal: every frame
// produced by an inner service is routed back to the Failover via
// handleInnerFrame, which then forwards (or intercepts) appropriately.
//
// serviceIdx records which inner service this bridge belongs to. Without
// it, an upstream ErrorFrame arriving from a service that is no longer
// active (because we already switched away) would be misattributed to the
// current active and trigger an extra spurious switch.
type frameBridge struct {
	owner      *Failover
	direction  frames.FrameDirection
	serviceIdx int
	name       string
}

func (b *frameBridge) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}

func (b *frameBridge) QueueFrame(frame frames.Frame, _ frames.FrameDirection) error {
	return b.owner.handleInnerFrame(frame, b.direction, b.serviceIdx)
}

func (b *frameBridge) PushFrame(frames.Frame, frames.FrameDirection) error {
	return nil
}

func (b *frameBridge) Link(processors.FrameProcessor) {}

func (b *frameBridge) SetPrev(processors.FrameProcessor) {}

func (b *frameBridge) Start(context.Context) error { return nil }

func (b *frameBridge) Stop() error { return nil }

func (b *frameBridge) Name() string { return b.name }
