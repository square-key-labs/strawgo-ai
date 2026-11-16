package llmservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Option func(*Service)

func WithName(name string) Option {
	return func(s *Service) { s.name = name }
}

func WithSettings(settings llm.Settings) Option {
	return func(s *Service) { s.settings = settings }
}

func WithSequentialFunctionCalls() Option {
	return func(s *Service) { s.sequential = true }
}

func WithFunctionCallTimeout(d time.Duration) Option {
	return func(s *Service) { s.callTimeout = d }
}

type registration struct {
	handler llm.FunctionHandler
	options llm.FunctionOptions
}

type runningCall struct {
	cancel               context.CancelFunc
	name                 string
	cancelOnInterruption bool
	cancellableByLLM     bool
	id                   string
	seq                  uint64
	args                 map[string]any
	cancelledByLLM       bool
	done                 chan struct{}
}

type Service struct {
	*processors.BaseProcessor

	name        string
	provider    llm.Provider
	sequential  bool
	callTimeout time.Duration

	mu           sync.RWMutex
	settings     llm.Settings
	functions    map[string]registration
	catchAll     *registration
	ttfbReported bool

	outMu sync.Mutex

	callsMu sync.Mutex
	calls   map[string]*runningCall
	callsWG sync.WaitGroup
	callSeq uint64
}

const stopGrace = 2 * time.Second

func New(provider llm.Provider, opts ...Option) *Service {
	s := &Service{
		provider:  provider,
		functions: make(map[string]registration),
		calls:     make(map[string]*runningCall),
	}
	for _, opt := range opts {
		opt(s)
	}
	if s.name == "" {
		s.name = provider.Name()
	}
	s.BaseProcessor = processors.NewBaseProcessor(s.name, s, processors.WithConcurrentSystemFrames())
	return s
}

func (s *Service) Provider() llm.Provider {
	return s.provider
}

func (s *Service) Settings() llm.Settings {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.settings
}

func (s *Service) UpdateSettings(delta llm.Settings) {
	s.mu.Lock()
	s.settings = s.settings.Merge(delta)
	s.mu.Unlock()
}

func (s *Service) RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions) {
	reg := registration{handler: handler, options: llm.DefaultFunctionOptions()}
	if len(opts) > 0 {
		reg.options = opts[0]
	}
	s.checkOptions(name, reg.options)
	s.mu.Lock()
	if name == "" {
		s.catchAll = &reg
	} else {
		s.functions[name] = reg
	}
	s.mu.Unlock()
}

func (s *Service) UnregisterFunction(name string) {
	s.mu.Lock()
	if name == "" {
		s.catchAll = nil
	} else {
		delete(s.functions, name)
	}
	s.mu.Unlock()
}

func (s *Service) HasFunction(name string) bool {
	_, ok := s.lookup(name)
	return ok
}

func (s *Service) lookup(name string) (registration, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if reg, ok := s.functions[name]; ok {
		return reg, true
	}
	if s.catchAll != nil {
		return *s.catchAll, true
	}
	return registration{}, false
}

func (s *Service) RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error) {
	var b strings.Builder
	err := s.provider.Stream(ctx, snapshot, s.Settings().Merge(delta), func(ev llm.Event) {
		if t, ok := ev.(llm.TextDelta); ok {
			b.WriteString(t.Text)
		}
	})
	return b.String(), err
}

func (s *Service) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.LLMContextFrame:
		if direction != frames.Downstream || f.Context == nil {
			return s.PushFrame(frame, direction)
		}
		return s.generate(ctx, f.Context)
	case *frames.LLMUpdateSettingsFrame:
		if f.Service != nil && f.Service.Name() != s.Name() {
			return s.PushFrame(frame, direction)
		}
		s.UpdateSettings(f.Delta)
		return nil
	case *frames.InterruptionFrame:
		s.cancelCalls(true)
		s.outMu.Lock()
		defer s.outMu.Unlock()
		return s.PushFrame(frame, direction)
	case *frames.CancelFrame:
		s.cancelCalls(false)
	case *frames.StartFrame:
		if err := s.PushFrame(frame, direction); err != nil {
			return err
		}
		return s.PushFrame(frames.NewLLMServiceMetadataFrame(s.Name(), false), frames.Upstream)
	case *frames.ServiceSwitcherRequestMetadataFrame:
		if f.Service != nil && f.Service.Name() == s.Name() {
			_ = s.PushFrame(frames.NewLLMServiceMetadataFrame(s.Name(), false), frames.Upstream)
		}
	}
	return s.PushFrame(frame, direction)
}

func (s *Service) Stop() error {
	s.cancelCalls(false)
	err := s.BaseProcessor.Stop()
	done := make(chan struct{})
	go func() {
		s.callsWG.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(stopGrace):
		logger.Warn("[%s] function calls still running after stop", s.Name())
	}
	return err
}

func (s *Service) emit(ctx context.Context, frame frames.Frame) {
	s.outMu.Lock()
	defer s.outMu.Unlock()
	if ctx.Err() != nil {
		return
	}
	_ = s.PushFrame(frame, frames.Downstream)
}

func (s *Service) generate(ctx context.Context, c *llm.Context) error {
	snapshot := s.withCancelTools(c.Snapshot())
	settings := s.Settings()
	cfg := s.StartConfig()
	base := frames.MetricsBase{Processor: s.Name(), Model: settings.Model}

	s.emit(ctx, frames.NewLLMFullResponseStartFrame())

	started := time.Now()
	gotFirst := false
	thinking := false
	signature := ""
	var calls []llm.ToolCall
	var usage llm.Usage
	gotUsage := false
	var info llm.ResponseInfo
	var done llm.Done
	var refusal *llm.Refusal
	var ttfb time.Duration
	ttfatPending := false
	thought := false

	first := func() {
		if gotFirst {
			return
		}
		gotFirst = true
		if !cfg.EnableMetrics {
			return
		}
		s.mu.Lock()
		skip := cfg.ReportOnlyInitialTTFB && s.ttfbReported
		s.ttfbReported = true
		s.mu.Unlock()
		if !skip {
			ttfb = time.Since(started)
			ttfatPending = true
			s.emit(ctx, frames.NewMetricsFrame(frames.TTFBMetricsData{MetricsBase: base, Value: ttfb}))
		}
	}
	answered := func() {
		if !ttfatPending {
			return
		}
		ttfatPending = false
		if !thought {
			return
		}
		ttfat := time.Since(started)
		s.emit(ctx, frames.NewMetricsFrame(frames.TTFATMetricsData{MetricsBase: base, TTFAT: ttfat, TTFB: ttfb, ThinkingTime: ttfat - ttfb}))
	}
	endThought := func() {
		if thinking {
			thinking = false
			s.emit(ctx, frames.NewLLMThoughtEndFrame(signature))
			signature = ""
		}
	}

	err := s.provider.Stream(ctx, snapshot, settings, func(ev llm.Event) {
		if ctx.Err() != nil {
			return
		}
		switch e := ev.(type) {
		case llm.TextDelta:
			if e.Text == "" {
				return
			}
			first()
			answered()
			endThought()
			s.emit(ctx, frames.NewLLMTextFrame(e.Text))
		case llm.ThoughtDelta:
			first()
			thought = true
			if !thinking {
				thinking = true
				s.emit(ctx, frames.NewLLMThoughtStartFrame(true, s.Name()))
			}
			if e.Signature != "" {
				signature = e.Signature
			}
			if e.Text != "" {
				s.emit(ctx, frames.NewLLMThoughtTextFrame(e.Text))
			}
		case llm.ToolCall:
			first()
			answered()
			calls = append(calls, e)
		case llm.UsageEvent:
			usage = e.Usage
			gotUsage = true
		case llm.ResponseInfo:
			info = info.Merge(e)
		case llm.Refusal:
			if refusal == nil {
				refusal = &e
			}
		case llm.Done:
			done = e
		}
	})
	if err == nil && refusal != nil {
		err = &llm.RefusalError{Refusal: *refusal}
	}

	endThought()
	s.emit(ctx, frames.NewLLMFullResponseEndFrame())

	var metrics []frames.MetricsData
	if cfg.EnableMetrics {
		metrics = append(metrics, frames.ProcessingMetricsData{MetricsBase: base, Value: time.Since(started)})
	}
	if cfg.EnableUsageMetrics && gotUsage {
		metrics = append(metrics, frames.LLMUsageMetricsData{MetricsBase: base, Value: usage})
	}
	if cfg.EnableMetrics && (info != llm.ResponseInfo{} || done != llm.Done{}) {
		metrics = append(metrics, frames.LLMResponseMetricsData{
			MetricsBase:   base,
			ResponseID:    info.ResponseID,
			RequestID:     info.RequestID,
			ServedModel:   info.Model,
			StopReason:    done.StopReason,
			RawStopReason: done.RawStopReason,
		})
	}
	if len(metrics) > 0 {
		_ = s.PushFrame(frames.NewMetricsFrame(metrics...), frames.Downstream)
	}

	if ctx.Err() != nil {
		return nil
	}
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return nil
		}
		if _, refused := llm.AsRefusal(err); refused {
			return s.pushError(fmt.Sprintf("%s response refused", s.Name()), err, frames.ErrorCategoryContentFiltered)
		}
		return s.pushError(fmt.Sprintf("%s generation failed", s.Name()), err, frames.ClassifyError(err))
	}
	if done.StopReason == llm.StopReasonLength {
		logger.Warn("[%s] response truncated by the output token limit (response %s, request %s)", s.Name(), info.ResponseID, info.RequestID)
	}
	if len(calls) > 0 {
		s.dispatchCalls(c, calls)
	}
	return nil
}

func (s *Service) pushError(msg string, err error, category frames.ErrorCategory) error {
	return s.PushErrorFrame(frames.NewCategorizedErrorFrame(fmt.Errorf("%s: %w", msg, err), category))
}

func (s *Service) broadcast(build func() frames.Frame) {
	if err := s.BroadcastFrame(nil, build); err != nil {
		logger.Error("[%s] broadcast failed: %v", s.Name(), err)
	}
}

func (s *Service) dispatchCalls(c *llm.Context, calls []llm.ToolCall) {
	lifetime := s.Context()
	if lifetime == nil {
		return
	}

	infos := make([]frames.FunctionCallInfo, len(calls))
	for i, call := range calls {
		infos[i] = frames.FunctionCallInfo{ToolCallID: call.ID, FunctionName: call.Name}
	}
	s.broadcast(func() frames.Frame { return frames.NewFunctionCallsStartedFrame(infos) })

	groupID := fmt.Sprintf("%s-%d", s.Name(), time.Now().UnixNano())
	type prepared struct {
		call llm.ToolCall
		reg  registration
		ok   bool
		args map[string]any
		err  error
		ctx  context.Context
	}
	batch := make([]prepared, len(calls))
	for i, call := range calls {
		p := prepared{call: call}
		p.reg, p.ok = s.resolve(call.Name)
		if len(call.Arguments) > 0 {
			p.err = json.Unmarshal(call.Arguments, &p.args)
		}
		if p.args == nil {
			p.args = map[string]any{}
		}

		timeout := p.reg.options.Timeout
		if timeout == 0 {
			timeout = s.callTimeout
		}
		var cancel context.CancelFunc
		if timeout > 0 {
			p.ctx, cancel = context.WithTimeout(lifetime, timeout)
		} else {
			p.ctx, cancel = context.WithCancel(lifetime)
		}
		s.callsMu.Lock()
		s.callSeq++
		s.calls[call.ID] = &runningCall{
			cancel:               cancel,
			name:                 call.Name,
			cancelOnInterruption: p.reg.options.CancelOnInterruption,
			cancellableByLLM:     cancellableByLLM(p.reg.options),
			id:                   call.ID,
			seq:                  s.callSeq,
			args:                 p.args,
			done:                 make(chan struct{}),
		}
		s.callsMu.Unlock()
		batch[i] = p

		args, cancelOnInterruption := p.args, p.reg.options.CancelOnInterruption
		s.broadcast(func() frames.Frame {
			f := frames.NewFunctionCallInProgressFrame(call.ID, call.Name, args, cancelOnInterruption)
			f.GroupID = groupID
			f.Signature = call.Signature
			return f
		})
	}

	run := func(p prepared) {
		defer s.callsWG.Done()
		result, err := s.execute(p.ctx, c, p.call, p.reg, p.ok, p.args, p.err)

		cancelled := errors.Is(p.ctx.Err(), context.Canceled)
		s.callsMu.Lock()
		rc := s.calls[p.call.ID]
		delete(s.calls, p.call.ID)
		byLLM := rc != nil && rc.cancelledByLLM
		s.callsMu.Unlock()
		if rc != nil {
			rc.cancel()
			defer close(rc.done)
		}

		if byLLM {
			s.broadcast(func() frames.Frame {
				f := frames.NewFunctionCallCancelFrame(p.call.ID, p.call.Name)
				f.Result = llm.ToolResultCancelled
				return f
			})
			return
		}
		if err != nil && cancelled {
			s.broadcast(func() frames.Frame {
				return frames.NewFunctionCallCancelFrame(p.call.ID, p.call.Name)
			})
			return
		}
		delivery, runLLM := resultDelivery(result, p.reg.options)
		s.broadcast(func() frames.Frame {
			f := frames.NewFunctionCallResultFrame(p.call.ID, p.call.Name, result.Value, runLLM)
			f.Arguments = p.args
			f.Delivery = delivery
			if err != nil {
				f.Error = err.Error()
				if f.Result == nil {
					f.Result = map[string]any{"error": err.Error()}
				}
			}
			return f
		})
	}

	s.callsWG.Add(len(batch))
	if s.sequential {
		go func() {
			for _, p := range batch {
				run(p)
			}
		}()
		return
	}
	for _, p := range batch {
		go run(p)
	}
}

func (s *Service) execute(ctx context.Context, c *llm.Context, call llm.ToolCall, reg registration, registered bool, args map[string]any, argErr error) (result llm.FunctionResult, err error) {
	if err := ctx.Err(); err != nil {
		return llm.FunctionResult{}, err
	}
	if !registered {
		return llm.FunctionResult{}, fmt.Errorf("function %q is not registered", call.Name)
	}
	if argErr != nil {
		return llm.FunctionResult{}, fmt.Errorf("invalid arguments for %q: %w", call.Name, argErr)
	}
	type outcome struct {
		result llm.FunctionResult
		err    error
	}
	done := make(chan outcome, 1)
	go func() {
		var out outcome
		defer func() {
			if r := recover(); r != nil {
				out.err = fmt.Errorf("function %q panicked: %v", call.Name, r)
			}
			done <- out
		}()
		out.result, out.err = reg.handler(ctx, llm.FunctionCallParams{
			ToolCallID:   call.ID,
			FunctionName: call.Name,
			Arguments:    args,
			RawArguments: json.RawMessage(call.Arguments),
			Context:      c,
			ReportProgress: func(progress llm.FunctionResult) bool {
				return s.reportProgress(call, args, reg.options, progress)
			},
		})
	}()
	select {
	case out := <-done:
		return out.result, out.err
	case <-ctx.Done():
		return llm.FunctionResult{}, ctx.Err()
	}
}

func (s *Service) cancelCalls(onlyInterruptible bool) {
	s.callsMu.Lock()
	var cancels []context.CancelFunc
	for _, rc := range s.calls {
		if onlyInterruptible && !rc.cancelOnInterruption {
			continue
		}
		cancels = append(cancels, rc.cancel)
	}
	s.callsMu.Unlock()
	for _, cancel := range cancels {
		cancel()
	}
}
