package strawtest

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type FakeCall struct {
	Name      string
	Arguments map[string]any
}

type FakeTurn struct {
	Reply string
	Calls []FakeCall
}

type FakeLLM struct {
	turns []FakeTurn

	mu       sync.Mutex
	requests []llm.Snapshot
}

func NewFakeLLM(turns ...FakeTurn) *FakeLLM {
	return &FakeLLM{turns: turns}
}

func (f *FakeLLM) Name() string { return "FakeLLM" }

func (f *FakeLLM) Requests() []llm.Snapshot {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llm.Snapshot(nil), f.requests...)
}

func (f *FakeLLM) Stream(ctx context.Context, snapshot llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
	f.mu.Lock()
	f.requests = append(f.requests, snapshot)
	f.mu.Unlock()

	users, afterTools := 0, false
	for _, m := range snapshot.Messages {
		switch m.Role {
		case llm.RoleUser:
			users++
			afterTools = false
		case llm.RoleTool:
			afterTools = true
		}
	}
	if users == 0 {
		return fmt.Errorf("strawtest: FakeLLM got a request with no user message")
	}
	if users > len(f.turns) {
		return fmt.Errorf("strawtest: FakeLLM has no scripted turn %d (it has %d)", users, len(f.turns))
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	turn := f.turns[users-1]
	if len(turn.Calls) > 0 && !afterTools {
		for i, c := range turn.Calls {
			args := []byte("{}")
			if c.Arguments != nil {
				encoded, err := json.Marshal(c.Arguments)
				if err != nil {
					return fmt.Errorf("strawtest: FakeLLM arguments for %s: %w", c.Name, err)
				}
				args = encoded
			}
			emit(llm.ToolCall{ID: fmt.Sprintf("call_%d_%d", users, i+1), Name: c.Name, Arguments: args})
		}
		emit(llm.Done{StopReason: llm.StopReasonToolCalls})
		return nil
	}
	if turn.Reply != "" {
		emit(llm.TextDelta{Text: turn.Reply})
	}
	emit(llm.Done{StopReason: llm.StopReasonStop})
	return nil
}
