package realtime

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type recordingOwner struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (o *recordingOwner) Name() string             { return "owner" }
func (o *recordingOwner) Context() context.Context { return context.Background() }
func (o *recordingOwner) BroadcastFrame(_ context.Context, ctor func() frames.Frame) error {
	o.mu.Lock()
	o.frames = append(o.frames, ctor())
	o.mu.Unlock()
	return nil
}

func (o *recordingOwner) count(name string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	n := 0
	for _, f := range o.frames {
		if f.Name() == name {
			n++
		}
	}
	return n
}

func runInterrupted(t *testing.T, opts ...llm.FunctionOptions) (*recordingOwner, Outcome, bool) {
	t.Helper()
	owner := &recordingOwner{}
	fns := NewFunctions(owner)
	entered := make(chan struct{})
	release := make(chan struct{})
	var handlerCancelled bool
	fns.Register("book", func(ctx context.Context, _ llm.FunctionCallParams) (llm.FunctionResult, error) {
		close(entered)
		select {
		case <-release:
			return llm.FunctionResult{Value: "booked"}, nil
		case <-ctx.Done():
			handlerCancelled = true
			return llm.FunctionResult{}, ctx.Err()
		}
	}, opts...)

	outcomes := make(chan Outcome, 2)
	fns.Dispatch(llm.NewContext(), []llm.ToolCall{{ID: "c1", Name: "book"}}, func(o Outcome) { outcomes <- o })
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("handler never started")
	}
	fns.Cancel(true)
	close(release)
	fns.Wait()
	if len(outcomes) != 1 {
		t.Fatalf("expected exactly one outcome, got %d", len(outcomes))
	}
	return owner, <-outcomes, handlerCancelled
}

func TestFunctionsDefaultSurvivesInterruption(t *testing.T) {
	owner, outcome, handlerCancelled := runInterrupted(t)
	if handlerCancelled || outcome.Cancelled || outcome.Value != "booked" {
		t.Fatalf("default function was cancelled by interruption: %+v", outcome)
	}
	if owner.count("FunctionCallResultFrame") != 1 || owner.count("FunctionCallCancelFrame") != 0 {
		t.Fatalf("expected one result and no cancel, got %d results %d cancels", owner.count("FunctionCallResultFrame"), owner.count("FunctionCallCancelFrame"))
	}
}

func TestFunctionsOptInCancelSettlesOnce(t *testing.T) {
	owner, outcome, _ := runInterrupted(t, llm.FunctionOptions{CancelOnInterruption: true})
	settled := owner.count("FunctionCallResultFrame") + owner.count("FunctionCallCancelFrame")
	if settled != 1 {
		t.Fatalf("expected the call to settle once, got %d frames", settled)
	}
	if outcome.Cancelled != (owner.count("FunctionCallCancelFrame") == 1) {
		t.Fatalf("outcome and frames disagree: %+v", outcome)
	}
}
