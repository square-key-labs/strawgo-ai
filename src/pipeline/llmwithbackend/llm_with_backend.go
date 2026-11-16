package llmwithbackend

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const (
	DefaultMaxBackendSteps = 10
	DefaultAckWait         = 2 * time.Second
)

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

func WithBackendHistoryLimit(messages int) Option {
	return func(w *LLMWithBackend) { w.historyLimit = messages }
}

func WithProgressNotesKept(keep bool) Option {
	return func(w *LLMWithBackend) { w.keepNotes = keep }
}

func WithAckWait(d time.Duration) Option {
	return func(w *LLMWithBackend) { w.ackWait = d }
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
	historyLimit      int
	keepNotes         bool
	ackWait           time.Duration
	tool              llm.FunctionSchema

	busy chan struct{}

	acksMu sync.Mutex
	acks   map[string]chan struct{}
}

func New(frontend, backend *llmservice.Service, opts ...Option) *LLMWithBackend {
	if frontend == nil || backend == nil {
		panic("llmwithbackend: frontend and backend must both be set")
	}
	if frontend == backend {
		panic("llmwithbackend: the frontend and the backend must be different services")
	}
	w := &LLMWithBackend{
		Service:           frontend,
		backend:           backend,
		frontendPairing:   FrontendPairingInstruction,
		backendPairing:    BackendPairingInstruction,
		delegateOptions:   llm.FunctionOptions{CancellableByLLM: true},
		respondOnDelegate: true,
		maxSteps:          DefaultMaxBackendSteps,
		ackWait:           DefaultAckWait,
		busy:              make(chan struct{}, 1),
		acks:              make(map[string]chan struct{}),
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
	frontend.AddRequestObserver(w.observeRequest)
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
	var acked <-chan struct{}
	if w.respondOnDelegate {
		acked = w.expectAck(params.ToolCallID)
		defer w.forgetAck(params.ToolCallID)
		run := true
		if !params.ReportProgress(llm.FunctionResult{Value: map[string]any{"status": "delegated"}, RunLLM: &run}) {
			acked = nil
		}
	}
	result, err := w.handle(ctx, params)
	if ctx.Err() != nil {
		return llm.FunctionResult{}, ctx.Err()
	}
	w.awaitAck(ctx, acked)
	return result, err
}

func (w *LLMWithBackend) handle(ctx context.Context, params llm.FunctionCallParams) (llm.FunctionResult, error) {
	select {
	case w.busy <- struct{}{}:
	case <-ctx.Done():
		return llm.FunctionResult{}, ctx.Err()
	}
	defer func() { <-w.busy }()

	request, err := w.strategy.ComposeRequest(params)
	if errors.Is(err, ErrNothingNew) {
		return llm.FunctionResult{Value: map[string]any{"status": "already_delegated"}}, nil
	}
	if err != nil {
		return llm.FunctionResult{}, err
	}
	w.trimHistory()
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

func (w *LLMWithBackend) expectAck(id string) <-chan struct{} {
	ch := make(chan struct{})
	w.acksMu.Lock()
	w.acks[id] = ch
	w.acksMu.Unlock()
	return ch
}

func (w *LLMWithBackend) forgetAck(id string) {
	w.acksMu.Lock()
	delete(w.acks, id)
	w.acksMu.Unlock()
}

func (w *LLMWithBackend) observeRequest(snapshot llm.Snapshot) {
	w.acksMu.Lock()
	for id, ch := range w.acks {
		if hasAck(snapshot.Messages, id) {
			close(ch)
			delete(w.acks, id)
		}
	}
	w.acksMu.Unlock()
	if !w.keepNotes && len(settledNotes(snapshot.Messages)) > 0 {
		w.dropNotes()
	}
}

func settledDelegations(messages []llm.Message) []string {
	var settled []string
	for _, m := range messages {
		if r, ok := m.ToolResult(); ok && r.Name == DelegateToolName && r.Content != llm.ToolResultInProgress {
			settled = append(settled, r.ToolCallID)
		}
	}
	return settled
}

func settledNotes(messages []llm.Message) []int {
	settled := settledDelegations(messages)
	if len(settled) == 0 {
		return nil
	}
	var notes []int
	for i, m := range messages {
		if m.Role == llm.RoleDeveloper && slices.ContainsFunc(settled, func(id string) bool { return isNote(m.Text(), id) }) {
			notes = append(notes, i)
		}
	}
	return notes
}

func hasAck(messages []llm.Message, id string) bool {
	for _, m := range messages {
		if m.Role == llm.RoleDeveloper && isNote(m.Text(), id) && strings.Contains(m.Text(), `"status":"delegated"`) {
			return true
		}
	}
	return false
}

func isNote(text, id string) bool {
	return strings.Contains(text, "function "+DelegateToolName+" (call "+id+")")
}

func (w *LLMWithBackend) awaitAck(ctx context.Context, acked <-chan struct{}) {
	if acked == nil {
		return
	}
	timer := time.NewTimer(w.ackWait)
	defer timer.Stop()
	select {
	case <-acked:
	case <-ctx.Done():
	case <-timer.C:
	}
}

func (w *LLMWithBackend) dropNotes() {
	_ = w.PushFrame(frames.NewLLMMessagesTransformFrame(func(messages []llm.Message) []llm.Message {
		settled := settledDelegations(messages)
		return slices.DeleteFunc(messages, func(m llm.Message) bool {
			return m.Role == llm.RoleDeveloper && slices.ContainsFunc(settled, func(s string) bool { return isNote(m.Text(), s) })
		})
	}, nil), frames.Upstream)
}

func (w *LLMWithBackend) trimHistory() {
	if w.historyLimit <= 0 {
		return
	}
	w.backendContext.Transform(func(messages []llm.Message) []llm.Message {
		lead := 0
		for lead < len(messages) && (messages[lead].Role == llm.RoleSystem || messages[lead].Role == llm.RoleDeveloper) {
			lead++
		}
		if len(messages)-lead <= w.historyLimit {
			return messages
		}
		start := len(messages)
		for i := max(lead, len(messages)-w.historyLimit); i < len(messages); i++ {
			if messages[i].Role == llm.RoleUser {
				start = i
				break
			}
		}
		return append(messages[:lead:lead], messages[start:]...)
	})
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
