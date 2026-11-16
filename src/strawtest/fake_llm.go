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

	mu        sync.Mutex
	requests  []llm.Snapshot
	next      int
	current   int
	callIDs   map[string]bool
	answered  bool
	callCount int
}

func NewFakeLLM(turns ...FakeTurn) *FakeLLM {
	return &FakeLLM{turns: turns, current: -1}
}

func (f *FakeLLM) Name() string { return "FakeLLM" }

func (f *FakeLLM) Requests() []llm.Snapshot {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llm.Snapshot(nil), f.requests...)
}

func (f *FakeLLM) continuesCurrent(snapshot llm.Snapshot) bool {
	if f.current < 0 || f.answered || len(f.callIDs) == 0 {
		return false
	}
	for _, m := range snapshot.Messages {
		for _, p := range m.Parts {
			if r, ok := p.(llm.ToolResultPart); ok && f.callIDs[r.ToolCallID] {
				return true
			}
		}
	}
	return false
}

func (f *FakeLLM) Stream(ctx context.Context, snapshot llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
	f.mu.Lock()
	f.requests = append(f.requests, snapshot)
	if !f.continuesCurrent(snapshot) {
		if f.next >= len(f.turns) {
			n := f.next + 1
			f.mu.Unlock()
			return fmt.Errorf("strawtest: FakeLLM has no scripted turn %d (it has %d)", n, len(f.turns))
		}
		f.current, f.next = f.next, f.next+1
		f.callIDs, f.answered = nil, false
	}
	turn := f.turns[f.current]
	var calls []llm.ToolCall
	if len(turn.Calls) > 0 && f.callIDs == nil {
		f.callIDs = map[string]bool{}
		for _, c := range turn.Calls {
			args := []byte("{}")
			if c.Arguments != nil {
				encoded, err := json.Marshal(c.Arguments)
				if err != nil {
					f.mu.Unlock()
					return fmt.Errorf("strawtest: FakeLLM arguments for %s: %w", c.Name, err)
				}
				args = encoded
			}
			f.callCount++
			id := fmt.Sprintf("fake_call_%d", f.callCount)
			f.callIDs[id] = true
			calls = append(calls, llm.ToolCall{ID: id, Name: c.Name, Arguments: args})
		}
	} else {
		f.answered = true
	}
	f.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return err
	}
	if len(calls) > 0 {
		for _, c := range calls {
			emit(c)
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
