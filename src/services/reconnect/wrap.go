package reconnect

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// Policy controls reconnect attempts for wrapped STT services.
type Policy struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
}

type wrappedSTT struct {
	*processors.BaseProcessor

	inner services.STTService

	policy Policy

	ctxMu sync.RWMutex
	ctx   context.Context

	reconnecting atomic.Bool
}

type frameBridge struct {
	owner     *wrappedSTT
	direction frames.FrameDirection
	name      string
}

// WrapSTT wraps an STT service with reconnect-on-error behavior.
func WrapSTT(inner services.STTService, policy Policy) services.STTService {
	wrapper := &wrappedSTT{
		inner:  inner,
		policy: policy,
	}
	wrapper.BaseProcessor = processors.NewBaseProcessor("ReconnectSTT", wrapper)

	inner.SetPrev(&frameBridge{
		owner:     wrapper,
		direction: frames.Upstream,
		name:      "ReconnectSTTUpstreamBridge",
	})
	inner.Link(&frameBridge{
		owner:     wrapper,
		direction: frames.Downstream,
		name:      "ReconnectSTTDownstreamBridge",
	})

	return wrapper
}

func (w *wrappedSTT) SetLanguage(lang string) {
	w.inner.SetLanguage(lang)
}

func (w *wrappedSTT) SetModel(model string) {
	w.inner.SetModel(model)
}

func (w *wrappedSTT) Initialize(ctx context.Context) error {
	w.setContext(ctx)
	return w.inner.Initialize(ctx)
}

func (w *wrappedSTT) Cleanup() error {
	return w.inner.Cleanup()
}

func (w *wrappedSTT) Start(ctx context.Context) error {
	w.setContext(ctx)
	return w.BaseProcessor.Start(ctx)
}

func (w *wrappedSTT) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	w.setContext(ctx)
	return w.inner.ProcessFrame(ctx, frame, direction)
}

func (w *wrappedSTT) handleInnerFrame(frame frames.Frame, direction frames.FrameDirection) error {
	if errFrame, ok := frame.(*frames.ErrorFrame); ok && direction == frames.Upstream {
		return w.handleErrorFrame(errFrame)
	}

	return w.PushFrame(frame, direction)
}

func (w *wrappedSTT) handleErrorFrame(frame *frames.ErrorFrame) error {
	if w.policy.MaxRetries == 0 {
		return w.PushFrame(frame, frames.Upstream)
	}

	if !w.reconnecting.CompareAndSwap(false, true) {
		return nil
	}
	defer w.reconnecting.Store(false)

	ctx := w.reconnectContext()

	for attempt := 1; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return w.PushFrame(frame, frames.Upstream)
		}

		_ = w.inner.Cleanup()

		delay := w.retryDelay(attempt)
		if delay > 0 {
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return w.PushFrame(frame, frames.Upstream)
			case <-timer.C:
			}
		}

		if err := w.inner.Initialize(ctx); err == nil {
			return nil
		}

		if !w.shouldRetry(attempt) {
			return w.PushFrame(frame, frames.Upstream)
		}
	}
}

func (w *wrappedSTT) shouldRetry(attempt int) bool {
	switch w.policy.MaxRetries {
	case -1:
		return true
	case 0:
		return false
	default:
		return attempt < w.policy.MaxRetries
	}
}

func (w *wrappedSTT) retryDelay(attempt int) time.Duration {
	const maxDuration = time.Duration(1<<63 - 1)

	delay := w.policy.BaseDelay
	if delay <= 0 {
		return 0
	}

	for i := 1; i < attempt; i++ {
		if delay > maxDuration/2 {
			delay = maxDuration
			break
		}
		delay *= 2
	}

	if w.policy.MaxDelay > 0 && delay > w.policy.MaxDelay {
		return w.policy.MaxDelay
	}

	return delay
}

func (w *wrappedSTT) reconnectContext() context.Context {
	w.ctxMu.RLock()
	ctx := w.ctx
	w.ctxMu.RUnlock()

	if ctx == nil {
		return context.Background()
	}

	return ctx
}

func (w *wrappedSTT) setContext(ctx context.Context) {
	if ctx == nil {
		return
	}

	w.ctxMu.Lock()
	w.ctx = ctx
	w.ctxMu.Unlock()
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

func (b *frameBridge) Start(context.Context) error {
	return nil
}

func (b *frameBridge) Stop() error {
	return nil
}

func (b *frameBridge) Name() string {
	return b.name
}
