package flows

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type Args map[string]any

func Decode[T any](args Args) (T, error) {
	var out T
	raw, err := json.Marshal(args)
	if err != nil {
		return out, fmt.Errorf("flows: encode arguments: %w", err)
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return out, fmt.Errorf("flows: decode arguments into %T: %w", out, err)
	}
	return out, nil
}

type Handler func(ctx context.Context, args Args, m *Manager) (result any, next *Node, err error)

var NoResponse = &Node{Name: "flows.NoResponse"}

type Function struct {
	Name                 string
	Description          string
	Properties           map[string]any
	Required             []string
	Handler              Handler
	Next                 *Node
	CancelOnInterruption bool
	Timeout              time.Duration

	edges []*Node
}

func (f Function) Schema() llm.FunctionSchema {
	return llm.FunctionSchema{
		Name:        f.Name,
		Description: f.Description,
		Properties:  f.Properties,
		Required:    f.Required,
	}
}

func (f Function) options() llm.FunctionOptions {
	return llm.FunctionOptions{CancelOnInterruption: f.CancelOnInterruption, Timeout: f.Timeout}
}

type StrategyKind int

const (
	Append StrategyKind = iota
	Reset
	ResetWithSummary
)

func (k StrategyKind) String() string {
	switch k {
	case Append:
		return "append"
	case Reset:
		return "reset"
	case ResetWithSummary:
		return "reset_with_summary"
	}
	return fmt.Sprintf("StrategyKind(%d)", int(k))
}

const DefaultSummaryTimeout = 5 * time.Second

type ContextStrategy struct {
	Kind           StrategyKind
	SummaryPrompt  string
	SummaryTimeout time.Duration
}

func (s ContextStrategy) validate() error {
	switch s.Kind {
	case Append, Reset:
		return nil
	case ResetWithSummary:
		if s.SummaryPrompt == "" {
			return fmt.Errorf("flows: %s requires a summary prompt", s.Kind)
		}
		return nil
	}
	return fmt.Errorf("flows: unknown context strategy %s", s.Kind)
}

type Node struct {
	Name            string
	RoleMessage     string
	RoleMessages    []llm.Message
	TaskMessages    []llm.Message
	Functions       []Function
	PreActions      []Action
	PostActions     []Action
	ContextStrategy *ContextStrategy
	WaitForUser     bool
}
