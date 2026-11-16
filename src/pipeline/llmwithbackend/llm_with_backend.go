package llmwithbackend

import (
	"context"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const DefaultMaxBackendSteps = 10

type Option func(*LLMWithBackend)

func WithRequestStrategy(strategy RequestStrategy) Option {
	return func(w *LLMWithBackend) { w.strategy = strategy }
}

func WithFrontendPairingInstruction(text string) Option {
	return func(w *LLMWithBackend) { w.frontendPairing = text }
}

func WithBackendPairingInstruction(text string) Option {
	return func(w *LLMWithBackend) { w.backendPairing = text }
}

func WithBackendContext(c *llm.Context) Option {
	return func(w *LLMWithBackend) { w.backendContext = c }
}

func WithDelegateOptions(opts llm.FunctionOptions) Option {
	return func(w *LLMWithBackend) { w.delegateOptions = opts }
}

func WithRespondOnDelegate(respond bool) Option {
	return func(w *LLMWithBackend) { w.respondOnDelegate = respond }
}

func WithMaxBackendSteps(n int) Option {
	return func(w *LLMWithBackend) { w.maxSteps = n }
}

type LLMWithBackend struct {
	*llmservice.Service

	backend           *llmservice.Service
	backendContext    *llm.Context
	strategy          RequestStrategy
	frontendPairing   string
	backendPairing    string
	delegateOptions   llm.FunctionOptions
	respondOnDelegate bool
	maxSteps          int
	tool              llm.FunctionSchema

	busy chan struct{}
}

func New(frontend, backend *llmservice.Service, opts ...Option) *LLMWithBackend {
	w := &LLMWithBackend{
		Service:           frontend,
		backend:           backend,
		frontendPairing:   FrontendPairingInstruction,
		backendPairing:    BackendPairingInstruction,
		delegateOptions:   llm.FunctionOptions{CancellableByLLM: true},
		respondOnDelegate: true,
		maxSteps:          DefaultMaxBackendSteps,
		busy:              make(chan struct{}, 1),
	}
	for _, opt := range opts {
		opt(w)
	}
	if w.strategy == nil {
		w.strategy = NewTranscriptStrategy()
	}
	if w.backendContext == nil {
		w.backendContext = llm.NewContext()
	}
	if w.maxSteps <= 0 {
		w.maxSteps = DefaultMaxBackendSteps
	}
	w.tool = w.strategy.Tool()
	w.tool.Name = DelegateToolName
	frontend.AppendSystemInstruction(w.FrontendInstruction())
	frontend.AddBuiltinTool(w.tool)
	frontend.RegisterFunction(DelegateToolName, w.delegate, w.delegateOptions)
	return w
}

func (w *LLMWithBackend) Frontend() *llmservice.Service { return w.Service }

func (w *LLMWithBackend) Backend() *llmservice.Service { return w.backend }

func (w *LLMWithBackend) BackendContext() *llm.Context { return w.backendContext }

func (w *LLMWithBackend) Strategy() RequestStrategy { return w.strategy }

func (w *LLMWithBackend) Tool() llm.FunctionSchema { return w.tool }

func (w *LLMWithBackend) FrontendInstruction() string {
	stop := ""
	if w.delegateOptions.CancellableByLLM && !w.delegateOptions.CancelOnInterruption {
		stop = StopInstruction
	}
	pairing := strings.ReplaceAll(w.frontendPairing, StopPlaceholder, stop)
	guidance := w.strategy.FrontendInstruction()
	if !strings.Contains(pairing, StrategyPlaceholder) {
		return joinNonEmpty(pairing, guidance)
	}
	pairing = strings.ReplaceAll(pairing, StrategyPlaceholder, guidance)
	pairing = strings.ReplaceAll(pairing, "  ", " ")
	return strings.TrimSpace(strings.ReplaceAll(pairing, " \n", "\n"))
}

func (w *LLMWithBackend) delegate(ctx context.Context, params llm.FunctionCallParams) (llm.FunctionResult, error) {
	request, ok := w.strategy.ComposeRequest(params)
	if !ok {
		return quiet(map[string]any{"status": "already_delegated"}), nil
	}
	if w.respondOnDelegate {
		run := true
		params.ReportProgress(llm.FunctionResult{Value: map[string]any{"status": "delegated"}, RunLLM: &run})
	}
	select {
	case w.busy <- struct{}{}:
	case <-ctx.Done():
		return llm.FunctionResult{}, ctx.Err()
	}
	defer func() { <-w.busy }()

	w.backendContext.Add(llm.User(request))
	answer, err := w.work(ctx, params)
	if ctx.Err() != nil {
		return llm.FunctionResult{}, ctx.Err()
	}
	if err != nil {
		w.reportError(err)
		return llm.FunctionResult{Value: BackendMessagePrefix + "The work could not be completed.\n\n" + BackendMessageRelayInstruction}, err
	}
	if answer == "" {
		return quiet(map[string]any{"status": "done"}), nil
	}
	return llm.FunctionResult{Value: BackendMessagePrefix + answer + "\n\n" + BackendMessageRelayInstruction}, nil
}

func quiet(value any) llm.FunctionResult {
	run := false
	return llm.FunctionResult{Value: value, RunLLM: &run}
}

func joinNonEmpty(parts ...string) string {
	var kept []string
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, "\n\n")
}
