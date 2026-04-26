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
	// Read on the hot path, advanced under switchMu.
	activeIdx atomic.Int32

	// switchMu serializes the failover routine so concurrent error frames
	// from the same active service don't each trigger a switch.
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

	for _, svc := range svcs {
		svc.SetPrev(&frameBridge{
			owner:     f,
			direction: frames.Upstream,
			name:      "ServiceSwitcherUpstreamBridge",
		})
		svc.Link(&frameBridge{
			owner:     f,
			direction: frames.Downstream,
			name:      "ServiceSwitcherDownstreamBridge",
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
func (f *Failover) Cleanup() error {
	return f.Active().Cleanup()
}

// Start tracks the runner's context for use by error-driven re-init, then
// starts the BaseProcessor's frame loop.
func (f *Failover) Start(ctx context.Context) error {
	f.setContext(ctx)
	return f.BaseProcessor.Start(ctx)
}

// HandleFrame forwards to the active inner service. The inner service's
// upstream/downstream output is captured by frameBridge and routed back
// through handleInnerFrame.
func (f *Failover) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	f.setContext(ctx)
	return f.Active().ProcessFrame(ctx, frame, direction)
}

// handleInnerFrame is invoked for every frame an inner service emits.
// Upstream ErrorFrames trigger the failover decision; everything else is
// forwarded onto the Failover's own pipeline neighbors unchanged.
func (f *Failover) handleInnerFrame(frame frames.Frame, direction frames.FrameDirection) error {
	if errFrame, ok := frame.(*frames.ErrorFrame); ok && direction == frames.Upstream {
		return f.handleErrorFrame(errFrame)
	}
	return f.PushFrame(frame, direction)
}

// handleErrorFrame applies the failover decision: fatal -> propagate
// untouched; non-fatal -> Cleanup current, Initialize next, propagate
// the same ErrorFrame upstream as informational signal. If no fallback
// remains, the ErrorFrame is propagated unchanged.
func (f *Failover) handleErrorFrame(errFrame *frames.ErrorFrame) error {
	if errFrame.IsFatal() {
		return f.PushFrame(errFrame, frames.Upstream)
	}

	f.switchMu.Lock()
	defer f.switchMu.Unlock()

	cur := f.activeIdx.Load()
	next := cur + 1
	if int(next) >= len(f.services) {
		f.log.Warn("Active service %d emitted non-fatal error and no fallback remains: %v", cur, errFrame.Error)
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

	f.activeIdx.Store(next)

	if f.onSwitch != nil {
		f.onSwitch(SwitchEvent{
			FromIndex: int(cur),
			ToIndex:   int(next),
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
type frameBridge struct {
	owner     *Failover
	direction frames.FrameDirection
	name      string
}

func (b *frameBridge) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}

func (b *frameBridge) QueueFrame(frame frames.Frame, _ frames.FrameDirection) error {
	return b.owner.handleInnerFrame(frame, b.direction)
}

func (b *frameBridge) PushFrame(frames.Frame, frames.FrameDirection) error {
	return nil
}

func (b *frameBridge) Link(processors.FrameProcessor) {}

func (b *frameBridge) SetPrev(processors.FrameProcessor) {}

func (b *frameBridge) Start(context.Context) error { return nil }

func (b *frameBridge) Stop() error { return nil }

func (b *frameBridge) Name() string { return b.name }
