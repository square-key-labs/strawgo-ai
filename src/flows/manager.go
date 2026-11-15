package flows

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

var (
	ErrNotInitialized     = errors.New("flows: manager is not initialized")
	ErrAlreadyInitialized = errors.New("flows: manager is already initialized")
	ErrClosed             = errors.New("flows: manager is closed")
)

type LLM interface {
	RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions)
	UnregisterFunction(name string)
	RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error)
}

type Task interface {
	QueueFrames(list []frames.Frame) error
	OnFrameReachedDownstream(fn pipeline.FrameHandler, examples ...frames.Frame)
	OnFrameReachedUpstream(fn pipeline.FrameHandler, examples ...frames.Frame)
}

type Config struct {
	LLM             LLM
	Task            Task
	Context         *llm.Context
	Processor       *Processor
	ContextStrategy ContextStrategy
	GlobalFunctions []Function
	Actions         map[string]ActionHandler
}

type functionTable struct {
	node string
	gen  uint64
	fns  map[string]Function
}

type pendingTransition struct {
	next     *Node
	function string
	gen      uint64
}

type job struct {
	fn   func(ctx context.Context) error
	done chan error
}

type workerKey struct{}

type handlerKey struct{}

type handlerSlot struct {
	m    *Manager
	mu   sync.Mutex
	next *Node
	done bool
}

func (s *handlerSlot) capture(node *Node) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return false
	}
	s.next = node
	return true
}

func (s *handlerSlot) close() *Node {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.done = true
	return s.next
}

type Manager struct {
	llm      LLM
	task     Task
	context  *llm.Context
	strategy ContextStrategy
	globals  []Function
	state    *State

	ctx    context.Context
	cancel context.CancelFunc

	actionsMu sync.RWMutex
	actions   map[string]ActionHandler

	jobsMu     sync.Mutex
	jobs       []job
	wake       chan struct{}
	started    bool
	workerDone chan struct{}

	trackMu  sync.Mutex
	inflight int
	idle     chan struct{}
	deferred []Action

	table      atomic.Pointer[functionTable]
	registered map[string]llm.FunctionOptions
	generation atomic.Uint64
	nodeSeq    atomic.Uint64

	currentMu sync.RWMutex
	current   string

	pendingMu sync.Mutex
	pending   map[string]pendingTransition

	initialized atomic.Bool
	hookStarted atomic.Bool
	warnedHook  atomic.Bool
}

func NewManager(cfg Config) (*Manager, error) {
	if cfg.LLM == nil {
		return nil, errors.New("flows: Config.LLM is required")
	}
	if cfg.Task == nil {
		return nil, errors.New("flows: Config.Task is required")
	}
	if cfg.Context == nil {
		return nil, errors.New("flows: Config.Context is required")
	}
	if cfg.Processor == nil {
		return nil, errors.New("flows: Config.Processor is required; place it between the LLM and the assistant context aggregator")
	}
	if err := cfg.ContextStrategy.validate(); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	for _, f := range cfg.GlobalFunctions {
		if err := checkFunction("global functions", f); err != nil {
			return nil, err
		}
		if seen[f.Name] {
			return nil, fmt.Errorf("flows: duplicate global function %q", f.Name)
		}
		seen[f.Name] = true
	}

	ctx, cancel := context.WithCancel(context.Background())
	m := &Manager{
		llm:        cfg.LLM,
		task:       cfg.Task,
		context:    cfg.Context,
		strategy:   cfg.ContextStrategy,
		globals:    append([]Function(nil), cfg.GlobalFunctions...),
		state:      NewState(),
		ctx:        ctx,
		cancel:     cancel,
		actions:    make(map[string]ActionHandler, len(cfg.Actions)),
		wake:       make(chan struct{}, 1),
		workerDone: make(chan struct{}),
		registered: make(map[string]llm.FunctionOptions),
		pending:    make(map[string]pendingTransition),
	}
	for k, v := range cfg.Actions {
		m.actions[k] = v
	}
	m.table.Store(&functionTable{fns: map[string]Function{}})
	if !cfg.Processor.manager.CompareAndSwap(nil, m) {
		cancel()
		return nil, errors.New("flows: Config.Processor is already bound to another manager")
	}

	cfg.Task.OnFrameReachedDownstream(m.onDownstream,
		(*actionDoneFrame)(nil), (*functionActionFrame)(nil), (*frames.BotStoppedSpeakingFrame)(nil),
		(*frames.EndFrame)(nil), (*frames.CancelFrame)(nil), (*frames.StopFrame)(nil))
	cfg.Task.OnFrameReachedUpstream(func(frames.Frame) { m.onBotStoppedSpeaking() }, (*frames.BotStoppedSpeakingFrame)(nil))
	return m, nil
}

func (m *Manager) State() *State { return m.state }

func (m *Manager) LLM() LLM { return m.llm }

func (m *Manager) Task() Task { return m.task }

func (m *Manager) Context() *llm.Context { return m.context }

func (m *Manager) CurrentContext() []llm.Message { return m.context.Messages() }

func (m *Manager) CurrentNode() string {
	m.currentMu.RLock()
	defer m.currentMu.RUnlock()
	return m.current
}

func (m *Manager) RegisterAction(actionType string, handler ActionHandler) {
	m.actionsMu.Lock()
	m.actions[actionType] = handler
	m.actionsMu.Unlock()
}

func (m *Manager) actionHandler(actionType string) (ActionHandler, bool) {
	m.actionsMu.RLock()
	defer m.actionsMu.RUnlock()
	h, ok := m.actions[actionType]
	return h, ok && h != nil
}

func (m *Manager) Initialize(ctx context.Context, initial *Node) error {
	if !m.initialized.CompareAndSwap(false, true) {
		return ErrAlreadyInitialized
	}
	m.jobsMu.Lock()
	if !m.started {
		m.started = true
		go m.work()
	}
	m.jobsMu.Unlock()
	if initial == nil {
		return nil
	}
	return m.SetNode(ctx, initial)
}

func (m *Manager) SetNode(ctx context.Context, node *Node) error {
	if node == nil || node == NoResponse {
		return errors.New("flows: SetNode requires a node")
	}
	if !m.initialized.Load() {
		return ErrNotInitialized
	}
	if slot, ok := ctx.Value(handlerKey{}).(*handlerSlot); ok && slot.m == m && slot.capture(node) {
		return nil
	}
	if m.inWorker(ctx) {
		m.enqueue(func(ctx context.Context) error { return m.setNode(ctx, node) }, false)
		return nil
	}
	done := m.enqueue(func(ctx context.Context) error { return m.setNode(ctx, node) }, true)
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Manager) Close() {
	m.cancel()
	m.jobsMu.Lock()
	started := m.started
	m.jobsMu.Unlock()
	if started {
		<-m.workerDone
	}
}

func (m *Manager) workerContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, workerKey{}, m)
}

func (m *Manager) inWorker(ctx context.Context) bool {
	owner, _ := ctx.Value(workerKey{}).(*Manager)
	return owner == m
}

func (m *Manager) enqueue(fn func(ctx context.Context) error, wait bool) <-chan error {
	j := job{fn: fn}
	if wait {
		j.done = make(chan error, 1)
	}
	m.jobsMu.Lock()
	if m.ctx.Err() != nil {
		m.jobsMu.Unlock()
		if j.done != nil {
			j.done <- ErrClosed
		}
		return j.done
	}
	m.jobs = append(m.jobs, j)
	m.jobsMu.Unlock()
	select {
	case m.wake <- struct{}{}:
	default:
	}
	return j.done
}

func (m *Manager) work() {
	defer close(m.workerDone)
	ctx := m.workerContext(m.ctx)
	for {
		m.jobsMu.Lock()
		if len(m.jobs) == 0 {
			m.jobsMu.Unlock()
			select {
			case <-m.wake:
				continue
			case <-m.ctx.Done():
				m.drain()
				return
			}
		}
		j := m.jobs[0]
		m.jobs[0] = job{}
		m.jobs = m.jobs[1:]
		m.jobsMu.Unlock()

		err := m.runJob(ctx, j.fn)
		if j.done != nil {
			j.done <- err
		} else if err != nil {
			logger.Error("[flows] %v", err)
		}
	}
}

func (m *Manager) runJob(ctx context.Context, fn func(context.Context) error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("flows: transition panicked: %v", r)
		}
	}()
	return fn(ctx)
}

func (m *Manager) drain() {
	m.jobsMu.Lock()
	jobs := m.jobs
	m.jobs = nil
	m.jobsMu.Unlock()
	for _, j := range jobs {
		if j.done != nil {
			j.done <- ErrClosed
		}
	}
}

func (m *Manager) onDownstream(f frames.Frame) {
	switch f := f.(type) {
	case *actionDoneFrame:
		if f.m == m {
			m.release()
		}
	case *functionActionFrame:
		if f.m == m {
			m.runFunctionAction(f)
		}
	case *frames.BotStoppedSpeakingFrame:
		m.onBotStoppedSpeaking()
	case *frames.EndFrame, *frames.CancelFrame, *frames.StopFrame:
		m.cancel()
	}
}

func (m *Manager) setPending(toolCallID string, p pendingTransition) {
	if !m.hookStarted.Load() && m.warnedHook.CompareAndSwap(false, true) {
		logger.Warn("[flows] transition requested but the flow processor has not started; add the flows.Processor between the LLM and the assistant context aggregator")
	}
	m.pendingMu.Lock()
	m.pending[toolCallID] = p
	m.pendingMu.Unlock()
}

func (m *Manager) takePending(toolCallID string) (pendingTransition, bool) {
	m.pendingMu.Lock()
	defer m.pendingMu.Unlock()
	p, ok := m.pending[toolCallID]
	if ok {
		delete(m.pending, toolCallID)
	}
	return p, ok
}

func (m *Manager) scheduleTransition(p pendingTransition) {
	m.enqueue(func(ctx context.Context) error {
		if p.gen != m.generation.Load() {
			logger.Debug("[flows] dropping stale transition from %s", p.function)
			return nil
		}
		logger.Debug("[flows] %s transitions to %s", p.function, p.next.Name)
		return m.setNode(ctx, p.next)
	}, false)
}

func checkFunction(where string, f Function) error {
	if f.Name == "" {
		return fmt.Errorf("flows: %s has a function without a name", where)
	}
	if f.Handler == nil && f.Next == nil {
		return fmt.Errorf("flows: %s function %q needs a Handler or a Next node", where, f.Name)
	}
	return nil
}

func (m *Manager) buildTable(name string, n *Node) (*functionTable, []llm.FunctionSchema, error) {
	where := fmt.Sprintf("node %q", name)
	fns := make(map[string]Function, len(m.globals)+len(n.Functions))
	schemas := make([]llm.FunctionSchema, 0, len(m.globals)+len(n.Functions))
	for _, f := range m.globals {
		fns[f.Name] = f
		schemas = append(schemas, f.Schema())
	}
	for _, f := range n.Functions {
		if err := checkFunction(where, f); err != nil {
			return nil, nil, err
		}
		if _, dup := fns[f.Name]; dup {
			return nil, nil, fmt.Errorf("flows: %s declares function %q more than once (or it is also global)", where, f.Name)
		}
		fns[f.Name] = f
		schemas = append(schemas, f.Schema())
	}
	return &functionTable{node: name, fns: fns}, schemas, nil
}

func (m *Manager) setNode(ctx context.Context, node *Node) error {
	name := node.Name
	if name == "" {
		name = fmt.Sprintf("node-%d", m.nodeSeq.Add(1))
	}
	if err := m.prepareAndSet(ctx, name, node); err != nil {
		return fmt.Errorf("flows: set node %q: %w", name, err)
	}
	return nil
}

func (m *Manager) prepareAndSet(ctx context.Context, name string, node *Node) error {
	m.pendingMu.Lock()
	clear(m.pending)
	m.pendingMu.Unlock()

	if node.ContextStrategy != nil {
		if err := node.ContextStrategy.validate(); err != nil {
			return err
		}
	}
	n, err := m.state.renderNode(name, node)
	if err != nil {
		return err
	}
	if err := m.validateActions(name, n.PreActions); err != nil {
		return err
	}
	if err := m.validateActions(name, n.PostActions); err != nil {
		return err
	}
	table, schemas, err := m.buildTable(name, n)
	if err != nil {
		return err
	}

	m.scheduleDeferred(nil)
	if err := m.runActions(ctx, n.PreActions); err != nil {
		return err
	}

	table.gen = m.generation.Add(1)
	m.swapFunctions(table)

	list, err := m.contextFrames(ctx, n, schemas)
	if err != nil {
		return err
	}
	if err := m.task.QueueFrames(list); err != nil {
		return err
	}

	m.currentMu.Lock()
	m.current = name
	m.currentMu.Unlock()

	if !n.WaitForUser {
		if err := m.task.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()}); err != nil {
			return err
		}
	}
	if len(n.PostActions) > 0 {
		if n.WaitForUser {
			m.scheduleDeferred(n.PostActions)
		} else if err := m.runActions(ctx, n.PostActions); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manager) swapFunctions(table *functionTable) {
	m.table.Store(table)
	for name, f := range table.fns {
		opts := f.options()
		if prev, ok := m.registered[name]; ok && prev == opts {
			continue
		}
		m.llm.RegisterFunction(name, m.trampoline(name), opts)
		m.registered[name] = opts
	}
	for name := range m.registered {
		if _, ok := table.fns[name]; !ok {
			m.llm.UnregisterFunction(name)
			delete(m.registered, name)
		}
	}
}

func (m *Manager) contextFrames(ctx context.Context, n *Node, schemas []llm.FunctionSchema) ([]frames.Frame, error) {
	var list []frames.Frame
	if n.RoleMessage != "" {
		settings := frames.NewLLMUpdateSettingsFrame(llm.Settings{SystemInstruction: n.RoleMessage})
		settings.ReachInactiveServices = true
		list = append(list, settings)
	}

	strategy := m.strategy
	if n.ContextStrategy != nil {
		strategy = *n.ContextStrategy
	}
	messages := append([]llm.Message(nil), n.RoleMessages...)
	kind := strategy.Kind
	if kind == ResetWithSummary {
		summary, err := m.summarize(ctx, strategy)
		switch {
		case err != nil:
			logger.Warn("[flows] summary for node %q failed, appending instead: %v", n.Name, err)
			kind = Append
		case strings.TrimSpace(summary) == "":
			logger.Warn("[flows] summary for node %q was empty, appending instead", n.Name)
			kind = Append
		default:
			messages = append(messages, llm.Developer("Here's a summary of the conversation:\n"+summary))
		}
	}
	messages = append(messages, n.TaskMessages...)

	var update frames.Frame
	if kind == Append {
		f := frames.NewLLMMessagesAppendFrame(messages, false)
		f.SetSurvivesInterruption(true)
		update = f
	} else {
		f := frames.NewLLMMessagesUpdateFrame(messages, false)
		f.SetSurvivesInterruption(true)
		update = f
	}
	tools := frames.NewLLMSetToolsFrame(llm.Tools{Functions: schemas})
	tools.SetSurvivesInterruption(true)
	return append(list, update, tools), nil
}

func (m *Manager) summarize(ctx context.Context, strategy ContextStrategy) (string, error) {
	timeout := strategy.SummaryTimeout
	if timeout <= 0 {
		timeout = DefaultSummaryTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	history, err := json.Marshal(transcript(m.context.Messages()))
	if err != nil {
		return "", err
	}
	snapshot := llm.Snapshot{Messages: []llm.Message{llm.Developer("Conversation history: " + string(history))}}
	return m.llm.RunInference(ctx, snapshot, llm.Settings{SystemInstruction: strategy.SummaryPrompt})
}

type transcriptEntry struct {
	Role       string `json:"role"`
	Content    string `json:"content,omitempty"`
	ToolCalls  []any  `json:"tool_calls,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

func transcript(messages []llm.Message) []transcriptEntry {
	out := make([]transcriptEntry, 0, len(messages))
	for _, msg := range messages {
		e := transcriptEntry{Role: string(msg.Role), Content: msg.Text()}
		for _, c := range msg.ToolCalls() {
			e.ToolCalls = append(e.ToolCalls, map[string]any{"name": c.Name, "arguments": string(c.Arguments)})
		}
		if r, ok := msg.ToolResult(); ok {
			e.ToolCallID = r.ToolCallID
			e.Content = r.Content
		}
		out = append(out, e)
	}
	return out
}

func (m *Manager) trampoline(name string) llm.FunctionHandler {
	return func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		table := m.table.Load()
		f, ok := table.fns[name]
		if !ok {
			return errorResult(fmt.Errorf("function %q is not available in node %q", name, table.node)), nil
		}
		slot := &handlerSlot{m: m}
		result, next, err := m.invoke(context.WithValue(ctx, handlerKey{}, slot), f, p)
		if captured := slot.close(); next == nil {
			next = captured
		}
		if err != nil {
			logger.Warn("[flows] function %s failed: %v", name, err)
			return errorResult(err), nil
		}
		if next == nil {
			next = f.Next
		}
		if result == nil {
			result = map[string]any{"status": "acknowledged"}
		}
		if next == nil {
			return llm.FunctionResult{Value: result}, nil
		}
		run := false
		if next != NoResponse {
			m.setPending(p.ToolCallID, pendingTransition{next: next, function: name, gen: table.gen})
		}
		return llm.FunctionResult{Value: result, RunLLM: &run}, nil
	}
}

func (m *Manager) invoke(ctx context.Context, f Function, p llm.FunctionCallParams) (result any, next *Node, err error) {
	if f.Handler == nil {
		return nil, nil, nil
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("function %q panicked: %v", f.Name, r)
		}
	}()
	args := make(Args, len(p.Arguments))
	for k, v := range p.Arguments {
		args[k] = v
	}
	return f.Handler(ctx, args, m)
}

func errorResult(err error) llm.FunctionResult {
	return llm.FunctionResult{Value: map[string]any{"status": "error", "error": err.Error()}}
}
