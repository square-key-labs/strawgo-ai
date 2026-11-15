package flows

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type FlowConfig struct {
	InitialNode     string                `json:"initial_node"`
	Nodes           map[string]NodeConfig `json:"nodes"`
	GlobalFunctions []FunctionConfig      `json:"global_functions,omitempty"`
}

type NodeConfig struct {
	RoleMessage        string           `json:"role_message,omitempty"`
	TaskMessages       []MessageConfig  `json:"task_messages"`
	Functions          []FunctionConfig `json:"functions,omitempty"`
	PreActions         []ActionConfig   `json:"pre_actions,omitempty"`
	PostActions        []ActionConfig   `json:"post_actions,omitempty"`
	ContextStrategy    string           `json:"context_strategy,omitempty"`
	SummaryPrompt      string           `json:"summary_prompt,omitempty"`
	RespondImmediately *bool            `json:"respond_immediately,omitempty"`
}

type MessageConfig struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type FunctionConfig struct {
	Name                 string         `json:"name"`
	Description          string         `json:"description,omitempty"`
	Properties           map[string]any `json:"properties,omitempty"`
	Required             []string       `json:"required,omitempty"`
	Handler              string         `json:"handler,omitempty"`
	TransitionOnly       bool           `json:"transition_only,omitempty"`
	TransitionTo         *Transition    `json:"transition_to,omitempty"`
	CancelOnInterruption bool           `json:"cancel_on_interruption,omitempty"`
	TimeoutSecs          float64        `json:"timeout_secs,omitempty"`
}

type Transition struct {
	Node    string
	Field   string
	Cases   map[string]string
	Default string
}

type branchJSON struct {
	Field   string            `json:"field"`
	Cases   map[string]string `json:"cases"`
	Default string            `json:"default,omitempty"`
}

func (t *Transition) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err == nil {
		*t = Transition{Node: name}
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var b branchJSON
	if err := dec.Decode(&b); err != nil {
		return fmt.Errorf("transition_to must be a node name or {field, cases, default}: %w", err)
	}
	cases := make(map[string]string, len(b.Cases))
	for k, v := range b.Cases {
		cases[caseKey(k)] = v
	}
	*t = Transition{Field: b.Field, Cases: cases, Default: b.Default}
	return nil
}

func (t Transition) MarshalJSON() ([]byte, error) {
	if t.Field == "" {
		return json.Marshal(t.Node)
	}
	return json.Marshal(branchJSON{Field: t.Field, Cases: t.Cases, Default: t.Default})
}

func (t Transition) targets() []string {
	if t.Field == "" {
		if t.Node == "" {
			return nil
		}
		return []string{t.Node}
	}
	keys := make([]string, 0, len(t.Cases))
	for k := range t.Cases {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]string, 0, len(keys)+1)
	for _, k := range keys {
		out = append(out, t.Cases[k])
	}
	if t.Default != "" {
		out = append(out, t.Default)
	}
	return out
}

type ActionConfig struct {
	Type                string
	Text                string
	AppendTextToContext *bool
	Handler             string
	Params              map[string]any
}

func (a *ActionConfig) UnmarshalJSON(data []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	out := ActionConfig{Params: map[string]any{}}
	for k, v := range raw {
		switch k {
		case "type":
			s, ok := v.(string)
			if !ok {
				return errors.New("action type must be a string")
			}
			out.Type = s
		case "handler":
			s, ok := v.(string)
			if !ok {
				return errors.New("action handler must be a string")
			}
			out.Handler = s
		case "text":
			s, ok := v.(string)
			if !ok {
				return errors.New("action text must be a string")
			}
			out.Text = s
		case "append_text_to_context":
			b, ok := v.(bool)
			if !ok {
				return errors.New("append_text_to_context must be a boolean")
			}
			out.AppendTextToContext = &b
		default:
			out.Params[k] = v
		}
	}
	*a = out
	return nil
}

func (a ActionConfig) MarshalJSON() ([]byte, error) {
	out := make(map[string]any, len(a.Params)+4)
	for k, v := range a.Params {
		out[k] = v
	}
	out["type"] = a.Type
	if a.Text != "" {
		out["text"] = a.Text
	}
	if a.Handler != "" {
		out["handler"] = a.Handler
	}
	if a.AppendTextToContext != nil {
		out["append_text_to_context"] = *a.AppendTextToContext
	}
	return json.Marshal(out)
}

type Registry struct {
	Handlers map[string]Handler
	Actions  map[string]ActionHandler
}

type Problem struct {
	Node     string
	Function string
	Message  string
}

func (p Problem) String() string {
	var where []string
	if p.Node != "" {
		where = append(where, fmt.Sprintf("node %q", p.Node))
	}
	if p.Function != "" {
		where = append(where, fmt.Sprintf("function %q", p.Function))
	}
	if len(where) == 0 {
		return p.Message
	}
	return strings.Join(where, " ") + ": " + p.Message
}

type ValidationError struct {
	Problems []Problem
}

func (e *ValidationError) Error() string {
	lines := make([]string, len(e.Problems))
	for i, p := range e.Problems {
		lines[i] = "  - " + p.String()
	}
	return fmt.Sprintf("flows: invalid flow config (%d problems):\n%s", len(e.Problems), strings.Join(lines, "\n"))
}

func ParseConfig(data []byte) (*FlowConfig, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var cfg FlowConfig
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("flows: parse flow config: %w", err)
	}
	return &cfg, nil
}

func LoadJSON(data []byte, reg Registry) (*Flow, error) {
	cfg, err := ParseConfig(data)
	if err != nil {
		return nil, err
	}
	return cfg.Build(reg)
}

type Flow struct {
	config  *FlowConfig
	initial *Node
	nodes   map[string]*Node
	globals []Function
}

func (f *Flow) Config() *FlowConfig { return f.config }

func (f *Flow) Initial() *Node { return f.initial }

func (f *Flow) Node(name string) (*Node, bool) {
	n, ok := f.nodes[name]
	return n, ok
}

func (f *Flow) GlobalFunctions() []Function { return append([]Function(nil), f.globals...) }

func (f *Flow) Mermaid() string { return Mermaid(f.initial, f.globals...) }

type builder struct {
	cfg      *FlowConfig
	reg      Registry
	nodes    map[string]*Node
	problems []Problem
}

func (b *builder) problem(node, function, format string, args ...any) {
	b.problems = append(b.problems, Problem{Node: node, Function: function, Message: fmt.Sprintf(format, args...)})
}

func (c *FlowConfig) Build(reg Registry) (*Flow, error) {
	b := &builder{cfg: c, reg: reg, nodes: make(map[string]*Node, len(c.Nodes))}
	if len(c.Nodes) == 0 {
		b.problem("", "", "flow has no nodes")
	}
	if c.InitialNode == "" {
		b.problem("", "", "initial_node is required")
	} else if _, ok := c.Nodes[c.InitialNode]; !ok {
		b.problem("", "", "initial_node %q is not a defined node", c.InitialNode)
	}

	names := make([]string, 0, len(c.Nodes))
	for name := range c.Nodes {
		names = append(names, name)
		b.nodes[name] = &Node{Name: name}
	}
	sort.Strings(names)

	globalNames := map[string]bool{}
	globals := make([]Function, 0, len(c.GlobalFunctions))
	for _, fc := range c.GlobalFunctions {
		if globalNames[fc.Name] {
			b.problem("", fc.Name, "duplicate global function")
		}
		globalNames[fc.Name] = true
		if f, ok := b.function("", fc); ok {
			globals = append(globals, f)
		}
	}

	for _, name := range names {
		b.node(name, c.Nodes[name], globalNames)
	}
	b.checkReachable(names)

	if len(b.problems) > 0 {
		return nil, &ValidationError{Problems: b.problems}
	}
	return &Flow{config: c, initial: b.nodes[c.InitialNode], nodes: b.nodes, globals: globals}, nil
}

func (b *builder) node(name string, nc NodeConfig, globalNames map[string]bool) {
	n := b.nodes[name]
	n.RoleMessage = nc.RoleMessage
	n.WaitForUser = nc.RespondImmediately != nil && !*nc.RespondImmediately
	for i, mc := range nc.TaskMessages {
		switch llm.Role(mc.Role) {
		case llm.RoleSystem, llm.RoleDeveloper, llm.RoleUser, llm.RoleAssistant:
			n.TaskMessages = append(n.TaskMessages, llm.Text(llm.Role(mc.Role), mc.Content))
		default:
			b.problem(name, "", "task_messages[%d] has unsupported role %q", i, mc.Role)
		}
	}

	switch nc.ContextStrategy {
	case "":
		if nc.SummaryPrompt != "" {
			b.problem(name, "", "summary_prompt is only used with context_strategy reset_with_summary")
		}
	case "append":
		n.ContextStrategy = &ContextStrategy{Kind: Append}
	case "reset":
		n.ContextStrategy = &ContextStrategy{Kind: Reset}
	case "reset_with_summary":
		if nc.SummaryPrompt == "" {
			b.problem(name, "", "context_strategy reset_with_summary requires summary_prompt")
		}
		n.ContextStrategy = &ContextStrategy{Kind: ResetWithSummary, SummaryPrompt: nc.SummaryPrompt}
	default:
		b.problem(name, "", "unknown context_strategy %q (want append, reset or reset_with_summary)", nc.ContextStrategy)
	}

	seen := map[string]bool{}
	for _, fc := range nc.Functions {
		if seen[fc.Name] {
			b.problem(name, fc.Name, "duplicate function name")
		}
		seen[fc.Name] = true
		if globalNames[fc.Name] {
			b.problem(name, fc.Name, "function is also declared in global_functions")
		}
		if f, ok := b.function(name, fc); ok {
			n.Functions = append(n.Functions, f)
		}
	}
	n.PreActions = b.actions(name, "pre_actions", nc.PreActions)
	n.PostActions = b.actions(name, "post_actions", nc.PostActions)
}

func (b *builder) function(node string, fc FunctionConfig) (Function, bool) {
	ok := true
	if fc.Name == "" {
		b.problem(node, "", "function without a name")
		return Function{}, false
	}
	f := Function{
		Name:                 fc.Name,
		Description:          fc.Description,
		Properties:           fc.Properties,
		Required:             fc.Required,
		CancelOnInterruption: fc.CancelOnInterruption,
		Timeout:              time.Duration(fc.TimeoutSecs * float64(time.Second)),
	}
	if fc.TimeoutSecs < 0 {
		b.problem(node, fc.Name, "timeout_secs must not be negative")
		ok = false
	}

	var transition Transition
	if fc.TransitionTo != nil {
		transition = *fc.TransitionTo
		if transition.Field == "" && transition.Node == "" {
			b.problem(node, fc.Name, "transition_to is empty")
			ok = false
		}
		if transition.Field != "" && len(transition.Cases) == 0 {
			b.problem(node, fc.Name, "transition_to branch on %q has no cases", transition.Field)
			ok = false
		}
		for _, target := range transition.targets() {
			if t, exists := b.nodes[target]; exists {
				f.edges = append(f.edges, t)
			} else {
				b.problem(node, fc.Name, "transitions to unknown node %q", target)
				ok = false
			}
		}
	}

	if fc.TransitionOnly {
		if fc.Description == "" {
			b.problem(node, fc.Name, "transition_only function needs a description")
			ok = false
		}
		if fc.Handler != "" {
			b.problem(node, fc.Name, "transition_only function must not name a handler")
			ok = false
		}
		if transition.Node == "" || transition.Field != "" {
			b.problem(node, fc.Name, "transition_only function must name the node it transitions to")
			ok = false
		}
		if ok {
			f.Next = b.nodes[transition.Node]
			f.edges = nil
		}
		return f, ok
	}

	handlerName := fc.Handler
	if handlerName == "" {
		handlerName = fc.Name
	}
	h, found := b.reg.Handlers[handlerName]
	if !found || h == nil {
		b.problem(node, fc.Name, "handler %q is not in the registry", handlerName)
		return f, false
	}
	if !ok {
		return f, false
	}
	f.Handler = b.configHandler(fc.Name, transition, h)
	return f, true
}

func (b *builder) configHandler(name string, transition Transition, h Handler) Handler {
	nodes := b.nodes
	return func(ctx context.Context, args Args, m *Manager) (any, *Node, error) {
		result, next, err := h(ctx, args, m)
		if err != nil || next != nil {
			return result, next, err
		}
		if transition.Field == "" {
			if transition.Node == "" {
				return result, nil, nil
			}
			return result, nodes[transition.Node], nil
		}
		fields, ok := asMap(result)
		if !ok {
			return result, nil, fmt.Errorf("function %q branches on result field %q, but its result is not an object", name, transition.Field)
		}
		value, ok := fields[transition.Field]
		if !ok {
			return result, nil, fmt.Errorf("function %q branches on result field %q, but its result has no such field", name, transition.Field)
		}
		target, ok := transition.Cases[caseKey(value)]
		if !ok {
			target = transition.Default
		}
		if target == "" {
			return result, nil, nil
		}
		return result, nodes[target], nil
	}
}

func caseKey(v any) string {
	var text string
	switch t := v.(type) {
	case bool:
		return strconv.FormatBool(t)
	case float64:
		text = strconv.FormatFloat(t, 'f', -1, 64)
	case string:
		text = t
	default:
		text = fmt.Sprint(v)
	}
	if lower := strings.ToLower(text); lower == "true" || lower == "false" {
		return lower
	}
	return text
}

func (b *builder) actions(node, where string, list []ActionConfig) []Action {
	out := make([]Action, 0, len(list))
	for i, ac := range list {
		label := fmt.Sprintf("%s[%d]", where, i)
		skip := ac.AppendTextToContext != nil && !*ac.AppendTextToContext
		switch ac.Type {
		case "":
			b.problem(node, "", "%s has no type", label)
		case kindTTSSay:
			if ac.Text == "" {
				b.problem(node, "", "%s tts_say needs text", label)
			}
			if ac.Handler != "" {
				b.problem(node, "", "%s tts_say does not take a handler", label)
			}
			out = append(out, TTSSay{Text: ac.Text, SkipContext: skip})
		case kindEnd:
			if ac.Handler != "" {
				b.problem(node, "", "%s end_conversation does not take a handler", label)
			}
			out = append(out, EndConversation{Text: ac.Text, SkipContext: skip})
		case kindFunction:
			h, ok := b.reg.Actions[ac.Handler]
			if ac.Handler == "" || !ok || h == nil {
				b.problem(node, "", "%s function action handler %q is not in the registry", label, ac.Handler)
				continue
			}
			action := Custom{Type: kindFunction, Params: actionParams(ac)}
			out = append(out, FunctionAction{Fn: func(ctx context.Context, m *Manager) error { return h(ctx, action, m) }})
		default:
			name := ac.Handler
			if name == "" {
				name = ac.Type
			}
			h, ok := b.reg.Actions[name]
			if !ok || h == nil {
				b.problem(node, "", "%s action %q has no handler %q in the registry", label, ac.Type, name)
				continue
			}
			out = append(out, registryAction{Custom: Custom{Type: ac.Type, Params: actionParams(ac)}, handler: h})
		}
	}
	return out
}

func actionParams(ac ActionConfig) map[string]any {
	out := make(map[string]any, len(ac.Params)+1)
	for k, v := range ac.Params {
		out[k] = v
	}
	if ac.Text != "" {
		out["text"] = ac.Text
	}
	return out
}

type registryAction struct {
	Custom
	handler ActionHandler
}

func (a registryAction) Execute(ctx context.Context, m *Manager) error {
	return a.handler(ctx, a.Custom, m)
}

func (b *builder) checkReachable(names []string) {
	if _, ok := b.nodes[b.cfg.InitialNode]; !ok {
		return
	}
	reached := map[string]bool{b.cfg.InitialNode: true}
	queue := []string{b.cfg.InitialNode}
	var globalTargets []string
	for _, fc := range b.cfg.GlobalFunctions {
		if fc.TransitionTo != nil {
			globalTargets = append(globalTargets, fc.TransitionTo.targets()...)
		}
	}
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		targets := append([]string(nil), globalTargets...)
		for _, fc := range b.cfg.Nodes[name].Functions {
			if fc.TransitionTo != nil {
				targets = append(targets, fc.TransitionTo.targets()...)
			}
		}
		for _, t := range targets {
			if _, exists := b.nodes[t]; exists && !reached[t] {
				reached[t] = true
				queue = append(queue, t)
			}
		}
	}
	for _, name := range names {
		if !reached[name] {
			b.problem(name, "", "node is unreachable from initial_node %q", b.cfg.InitialNode)
		}
	}
}
