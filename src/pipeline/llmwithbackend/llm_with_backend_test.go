package llmwithbackend

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const waitTimeout = 5 * time.Second

type replyFunc func(ctx context.Context, snap llm.Snapshot) ([]llm.Event, error)

type fakeLLM struct {
	name     string
	fn       replyFunc
	mu       sync.Mutex
	snaps    []llm.Snapshot
	settings []llm.Settings
}

func (f *fakeLLM) Name() string { return f.name }

func (f *fakeLLM) Stream(ctx context.Context, snap llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	f.mu.Lock()
	f.snaps = append(f.snaps, snap)
	f.settings = append(f.settings, settings)
	f.mu.Unlock()
	events, err := f.fn(ctx, snap)
	for _, e := range events {
		emit(e)
	}
	return err
}

func (f *fakeLLM) requests() ([]llm.Snapshot, []llm.Settings) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llm.Snapshot(nil), f.snaps...), append([]llm.Settings(nil), f.settings...)
}

func (f *fakeLLM) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.snaps)
}

func say(text string) []llm.Event {
	return []llm.Event{llm.TextDelta{Text: text}, llm.Done{StopReason: llm.StopReasonStop}}
}

func call(id, name, args string) []llm.Event {
	return []llm.Event{llm.ToolCall{ID: id, Name: name, Arguments: []byte(args)}, llm.Done{StopReason: llm.StopReasonToolCalls}}
}

func toolResult(messages []llm.Message, id string) (llm.ToolResultPart, bool) {
	for _, m := range messages {
		if r, ok := m.ToolResult(); ok && r.ToolCallID == id {
			return r, true
		}
	}
	return llm.ToolResultPart{}, false
}

func hasTool(snap llm.Snapshot, name string) bool {
	for _, f := range snap.Tools.Functions {
		if f.Name == name {
			return true
		}
	}
	return false
}

func delegatingFrontend(args, ack, relay string) replyFunc {
	return func(_ context.Context, snap llm.Snapshot) ([]llm.Event, error) {
		r, ok := toolResult(snap.Messages, "d1")
		switch {
		case !ok:
			return call("d1", DelegateToolName, args), nil
		case r.Content == llm.ToolResultInProgress:
			return say(ack), nil
		default:
			return say(relay), nil
		}
	}
}

type bot struct {
	t            *testing.T
	conversation *llm.Context
	task         *pipeline.PipelineTask
	mu           sync.Mutex
	down         []frames.Frame
	up           []frames.Frame
}

func runBot(t *testing.T, proc processors.FrameProcessor) *bot {
	t.Helper()
	b := &bot{t: t, conversation: llm.NewContext(llm.System("You are a test bot."))}
	pair := aggregators.NewContextAggregatorPair(b.conversation, aggregators.ContextAggregatorPairParams{})
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.CancelOnIdleTimeout = false
	cfg.EnableMetrics = true
	cfg.EnableUsageMetrics = true
	b.task = pipeline.NewPipelineTaskWithConfig(pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), proc, pair.Assistant()}), cfg)
	b.task.OnFrameReachedDownstream(func(f frames.Frame) {
		b.mu.Lock()
		b.down = append(b.down, f)
		b.mu.Unlock()
	})
	b.task.OnFrameReachedUpstream(func(f frames.Frame) {
		b.mu.Lock()
		b.up = append(b.up, f)
		b.mu.Unlock()
	})
	done := make(chan struct{})
	go func() {
		_ = b.task.Run(context.Background())
		close(done)
	}()
	t.Cleanup(func() {
		_ = b.task.QueueFrame(frames.NewEndFrame())
		select {
		case <-done:
		case <-time.After(waitTimeout):
			b.task.Cancel()
			<-done
		}
	})
	b.waitFor("the pipeline to start", func() bool {
		for _, f := range b.downstream() {
			if _, ok := f.(*frames.StartFrame); ok {
				return true
			}
		}
		return false
	})
	return b
}

func (b *bot) downstream() []frames.Frame {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]frames.Frame(nil), b.down...)
}

func (b *bot) upstream() []frames.Frame {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]frames.Frame(nil), b.up...)
}

func (b *bot) userSays(text string) {
	b.t.Helper()
	if err := b.task.QueueFrame(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.User(text)}, true)); err != nil {
		b.t.Fatal(err)
	}
}

func (b *bot) queue(f frames.Frame) {
	b.t.Helper()
	if err := b.task.QueueFrame(f); err != nil {
		b.t.Fatal(err)
	}
}

func (b *bot) waitFor(desc string, cond func() bool) {
	b.t.Helper()
	deadline := time.Now().Add(waitTimeout)
	for !cond() {
		if time.Now().After(deadline) {
			b.t.Fatalf("timed out waiting for %s; bot said %q", desc, b.said())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func (b *bot) said() string {
	var out strings.Builder
	for _, f := range b.downstream() {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			out.WriteString(tf.Text)
			out.WriteString("|")
		}
	}
	return out.String()
}

func (b *bot) settled(id string) (llm.ToolResultPart, bool) {
	r, ok := toolResult(b.conversation.Messages(), id)
	return r, ok && r.Content != llm.ToolResultInProgress
}

func (b *bot) checkOneCallOneResult(id string) {
	b.t.Helper()
	calls, results := 0, 0
	for _, m := range b.conversation.Messages() {
		for _, p := range m.Parts {
			switch v := p.(type) {
			case llm.ToolCallPart:
				if v.ID == id {
					calls++
				}
			case llm.ToolResultPart:
				if v.ToolCallID == id {
					results++
				}
			}
		}
	}
	if calls != 1 || results != 1 {
		b.t.Fatalf("conversation has %d calls and %d results for %s, want 1 and 1: %+v", calls, results, id, b.conversation.Messages())
	}
}

func newPair(frontend, backend *fakeLLM, opts ...Option) *LLMWithBackend {
	return New(
		llmservice.New(frontend),
		llmservice.New(backend, llmservice.WithName("thinker"), llmservice.WithSettings(llm.Settings{Model: "big-model", SystemInstruction: "You look things up."})),
		opts...,
	)
}

func TestSimpleQuestionIsAnsweredByTheFrontendAlone(t *testing.T) {
	frontend := &fakeLLM{name: "fast", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return say("Paris is the capital of France."), nil
	}}
	backend := &fakeLLM{name: "slow", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return say("unexpected"), nil
	}}
	w := newPair(frontend, backend)
	b := runBot(t, w)

	b.userSays("What is the capital of France?")
	b.waitFor("the frontend's answer", func() bool { return strings.Contains(b.said(), "Paris is the capital") })

	if backend.count() != 0 {
		t.Fatalf("backend ran %d times", backend.count())
	}
	snaps, settings := frontend.requests()
	if !hasTool(snaps[0], DelegateToolName) {
		t.Fatalf("frontend request lacks the delegate tool: %+v", snaps[0].Tools)
	}
	if !strings.Contains(settings[0].SystemInstruction, "THE BACKEND:") || !strings.Contains(settings[0].SystemInstruction, TranscriptFrontendInstruction) || strings.Contains(settings[0].SystemInstruction, StrategyPlaceholder) {
		t.Fatalf("frontend instruction = %q", settings[0].SystemInstruction)
	}
	if !strings.Contains(settings[0].SystemInstruction, "cancel_delegate") {
		t.Fatalf("frontend instruction does not say how to stop work: %q", settings[0].SystemInstruction)
	}
	if !b.conversation.Tools().Empty() {
		t.Fatalf("the delegate tool leaked into the context: %+v", b.conversation.Tools())
	}
}

func TestDelegatedQuestionIsAcknowledgedThenAnsweredOnce(t *testing.T) {
	acked := make(chan struct{})
	var ackOnce sync.Once
	front := delegatingFrontend("{}", "Let me check.", "It is 21 degrees in Paris.")
	frontend := &fakeLLM{name: "fast", fn: func(ctx context.Context, snap llm.Snapshot) ([]llm.Event, error) {
		events, err := front(ctx, snap)
		if r, ok := toolResult(snap.Messages, "d1"); ok && r.Content == llm.ToolResultInProgress {
			ackOnce.Do(func() { close(acked) })
		}
		return events, err
	}}
	var weatherCalls atomic.Int32
	backend := &fakeLLM{name: "slow", fn: func(ctx context.Context, snap llm.Snapshot) ([]llm.Event, error) {
		if _, ok := toolResult(snap.Messages, "b1"); !ok {
			select {
			case <-acked:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			return append([]llm.Event{
				llm.TextDelta{Text: "Checking the weather."},
				llm.UsageEvent{Usage: llm.Usage{PromptTokens: 7, CompletionTokens: 3}},
			}, call("b1", "get_weather", `{"city":"Paris"}`)...), nil
		}
		return say("In Paris it is 21 degrees."), nil
	}}
	w := newPair(frontend, backend)
	w.Backend().RegisterFunction("get_weather", func(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		weatherCalls.Add(1)
		if p.Arguments["city"] != "Paris" {
			t.Errorf("get_weather args = %v", p.Arguments)
		}
		return llm.FunctionResult{Value: map[string]any{"temp_c": 21}}, nil
	})
	b := runBot(t, w)

	b.userSays("What's the weather in Paris?")
	b.waitFor("the relayed answer", func() bool { return strings.Contains(b.said(), "21 degrees in Paris") })
	b.waitFor("the relay to be recorded", func() bool {
		for _, m := range b.conversation.Messages() {
			if m.Role == llm.RoleAssistant && strings.Contains(m.Text(), "21 degrees in Paris") {
				return true
			}
		}
		return false
	})

	said := b.said()
	if !strings.Contains(said, "Let me check.") || strings.Count(said, "21 degrees in Paris") != 1 || strings.Index(said, "Let me check.") > strings.Index(said, "21 degrees") {
		t.Fatalf("bot said %q", said)
	}
	if weatherCalls.Load() != 1 {
		t.Fatalf("get_weather ran %d times", weatherCalls.Load())
	}
	if backend.count() != 2 {
		t.Fatalf("backend ran %d times, want 2", backend.count())
	}
	b.checkOneCallOneResult("d1")
	r, _ := b.settled("d1")
	if !strings.Contains(r.Content, "Backend: In Paris it is 21 degrees.") || r.IsError {
		t.Fatalf("delegate result = %+v", r)
	}

	var notes []string
	for _, m := range b.conversation.Messages() {
		if m.Role == llm.RoleDeveloper {
			notes = append(notes, m.Text())
		}
	}
	joined := strings.Join(notes, "\n")
	for _, want := range []string{`"status":"delegated"`, `Backend (note): Checking the weather.`, `Backend (working): get_weather(city=\"Paris\")`} {
		if !strings.Contains(joined, want) {
			t.Fatalf("progress notes lack %s:\n%s", want, joined)
		}
	}

	history := w.BackendContext().Messages()
	roles := make([]string, len(history))
	for i, m := range history {
		roles[i] = string(m.Role)
	}
	if strings.Join(roles, ",") != "user,assistant,tool,assistant" {
		t.Fatalf("backend history roles = %v", roles)
	}
	if res, _ := toolResult(history, "b1"); res.Content != `{"temp_c":21}` {
		t.Fatalf("backend tool result = %+v", res)
	}
	_, settings := backend.requests()
	if settings[0].Model != "big-model" || !strings.HasPrefix(settings[0].SystemInstruction, "You look things up.\n\n") || !strings.Contains(settings[0].SystemInstruction, BackendPairingInstruction) {
		t.Fatalf("backend settings = %+v", settings[0])
	}
	if w.Backend().Settings().SystemInstruction != "You look things up." {
		t.Fatalf("backend settings were changed: %q", w.Backend().Settings().SystemInstruction)
	}

	var backendMetrics []string
	for _, f := range b.downstream() {
		if mf, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range mf.Data {
				if d.Base().Processor == "thinker" {
					if d.Base().Model != "big-model" {
						t.Fatalf("backend metric model = %q", d.Base().Model)
					}
					switch d.(type) {
					case frames.TTFBMetricsData:
						backendMetrics = append(backendMetrics, "ttfb")
					case frames.ProcessingMetricsData:
						backendMetrics = append(backendMetrics, "processing")
					case frames.LLMUsageMetricsData:
						backendMetrics = append(backendMetrics, "usage")
					}
				}
			}
		}
	}
	if got := strings.Join(backendMetrics, ","); got != "ttfb,processing,usage,ttfb,processing" {
		t.Fatalf("backend metrics = %s", got)
	}
}

func TestFrontendModelCancelsTheDelegation(t *testing.T) {
	started := make(chan struct{})
	var cancelled atomic.Bool
	var lookups atomic.Int32
	frontend := &fakeLLM{name: "fast", fn: func(_ context.Context, snap llm.Snapshot) ([]llm.Event, error) {
		last := snap.Messages[len(snap.Messages)-1]
		d1, _ := toolResult(snap.Messages, "d1")
		c1, cancelCalled := toolResult(snap.Messages, "c1")
		switch {
		case last.Role == llm.RoleUser && last.Text() == "Look up my order":
			return call("d1", DelegateToolName, "{}"), nil
		case last.Role == llm.RoleUser && last.Text() == "Never mind":
			if !hasTool(snap, llm.CancelToolName(DelegateToolName)) {
				return say("no cancel tool offered"), nil
			}
			return call("c1", llm.CancelToolName(DelegateToolName), "{}"), nil
		case cancelCalled && c1.Content != llm.ToolResultInProgress:
			return say("Okay, I dropped it."), nil
		case d1.Content == llm.ToolResultInProgress:
			return say("On it."), nil
		}
		return say("unexpected"), nil
	}}
	backend := &fakeLLM{name: "slow", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return call("b1", "lookup_order", `{}`), nil
	}}
	w := newPair(frontend, backend)
	w.Backend().RegisterFunction("lookup_order", func(ctx context.Context, _ llm.FunctionCallParams) (llm.FunctionResult, error) {
		lookups.Add(1)
		close(started)
		<-ctx.Done()
		cancelled.Store(true)
		return llm.FunctionResult{}, ctx.Err()
	})
	b := runBot(t, w)

	b.userSays("Look up my order")
	<-started
	b.waitFor("the acknowledgement", func() bool { return strings.Contains(b.said(), "On it.") })
	b.userSays("Never mind")
	b.waitFor("the confirmation", func() bool { return strings.Contains(b.said(), "Okay, I dropped it.") })

	if !cancelled.Load() || lookups.Load() != 1 {
		t.Fatalf("backend tool cancelled=%v ran=%d", cancelled.Load(), lookups.Load())
	}
	r, _ := b.settled("d1")
	if r.Content != llm.ToolResultCancelled {
		t.Fatalf("delegate result = %+v", r)
	}
	b.checkOneCallOneResult("d1")
	if res, _ := toolResult(w.BackendContext().Messages(), "b1"); res.Content != llm.ToolResultInterrupted {
		t.Fatalf("backend tool result = %+v", res)
	}
	if backend.count() != 1 {
		t.Fatalf("backend ran %d times after cancellation", backend.count())
	}
	if strings.Contains(b.said(), "no cancel tool") {
		t.Fatalf("bot said %q", b.said())
	}
}

func TestInterruptionDuringDelegation(t *testing.T) {
	for _, tc := range []struct {
		name        string
		opts        llm.FunctionOptions
		wantsResult bool
	}{
		{name: "default options keep the backend working", opts: llm.FunctionOptions{CancellableByLLM: true}, wantsResult: true},
		{name: "cancel on interruption stops it", opts: llm.FunctionOptions{CancelOnInterruption: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			started := make(chan struct{})
			release := make(chan struct{})
			var cancelled atomic.Bool
			frontend := &fakeLLM{name: "fast", fn: delegatingFrontend("{}", "One moment.", "Your order shipped.")}
			backend := &fakeLLM{name: "slow", fn: func(_ context.Context, snap llm.Snapshot) ([]llm.Event, error) {
				if _, ok := toolResult(snap.Messages, "b1"); !ok {
					return call("b1", "lookup_order", `{}`), nil
				}
				return say("The order shipped."), nil
			}}
			w := newPair(frontend, backend, WithDelegateOptions(tc.opts))
			w.Backend().RegisterFunction("lookup_order", func(ctx context.Context, _ llm.FunctionCallParams) (llm.FunctionResult, error) {
				close(started)
				select {
				case <-release:
					return llm.FunctionResult{Value: "shipped"}, nil
				case <-ctx.Done():
					cancelled.Store(true)
					return llm.FunctionResult{}, ctx.Err()
				}
			})
			b := runBot(t, w)

			b.userSays("Where is my order?")
			<-started
			b.queue(frames.NewInterruptionFrame())

			if tc.wantsResult {
				b.waitFor("the interruption to pass", func() bool {
					for _, f := range b.downstream() {
						if _, ok := f.(*frames.InterruptionFrame); ok {
							return true
						}
					}
					return false
				})
				close(release)
				b.waitFor("the relayed answer", func() bool { return strings.Contains(b.said(), "Your order shipped.") })
				if cancelled.Load() {
					t.Fatal("the backend tool was cancelled by the interruption")
				}
				if strings.Count(b.said(), "Your order shipped.") != 1 {
					t.Fatalf("bot said %q", b.said())
				}
				r, _ := b.settled("d1")
				if !strings.Contains(r.Content, "Backend: The order shipped.") {
					t.Fatalf("delegate result = %+v", r)
				}
			} else {
				b.waitFor("the delegation to be cancelled", func() bool { _, ok := b.settled("d1"); return ok })
				r, _ := b.settled("d1")
				if r.Content != llm.ToolResultInterrupted {
					t.Fatalf("delegate result = %+v", r)
				}
				b.waitFor("the backend tool to see the cancellation", cancelled.Load)
				if res, _ := toolResult(w.BackendContext().Messages(), "b1"); res.Content != llm.ToolResultInterrupted {
					t.Fatalf("backend tool result = %+v", res)
				}
				close(release)
				if backend.count() != 1 {
					t.Fatalf("backend ran %d times after the interruption", backend.count())
				}
			}
			b.checkOneCallOneResult("d1")
		})
	}
}

type rateLimited struct{}

func (rateLimited) Error() string                       { return "slow down" }
func (rateLimited) ErrorCategory() frames.ErrorCategory { return frames.ErrorCategoryRateLimit }

func TestBackendErrorIsSurfacedAndAttributedToTheBackend(t *testing.T) {
	frontend := &fakeLLM{name: "fast", fn: delegatingFrontend("{}", "Checking.", "Sorry, I could not finish that.")}
	backend := &fakeLLM{name: "slow", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return nil, rateLimited{}
	}}
	w := newPair(frontend, backend, WithRespondOnDelegate(false))
	b := runBot(t, w)

	b.userSays("Book me a flight")
	b.waitFor("the apology", func() bool { return strings.Contains(b.said(), "could not finish") })

	var errFrame *frames.ErrorFrame
	for _, f := range b.upstream() {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			errFrame = ef
		}
	}
	if errFrame == nil {
		t.Fatal("no ErrorFrame reached upstream")
	}
	var target rateLimited
	if errFrame.Processor != "thinker" || errFrame.Category != frames.ErrorCategoryRateLimit || !errors.As(errFrame.Error, &target) || !strings.Contains(errFrame.Error.Error(), "thinker generation failed") {
		t.Fatalf("error frame = processor %q category %q error %v", errFrame.Processor, errFrame.Category, errFrame.Error)
	}
	r, _ := b.settled("d1")
	if !r.IsError || !strings.Contains(r.Content, "could not be completed") {
		t.Fatalf("delegate result = %+v", r)
	}
	if strings.Contains(b.said(), "Checking.") {
		t.Fatalf("frontend acknowledged although RespondOnDelegate is off: %q", b.said())
	}
	b.checkOneCallOneResult("d1")
}

func TestExplicitStrategySendsTheWordedRequest(t *testing.T) {
	frontend := &fakeLLM{name: "fast", fn: delegatingFrontend(`{"request":"Book a table for two at 7pm"}`, "Sure.", "Booked for 7pm.")}
	backend := &fakeLLM{name: "slow", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return say("The table for two is booked at 7pm."), nil
	}}
	w := newPair(frontend, backend, WithRequestStrategy(NewExplicitStrategy()))
	b := runBot(t, w)

	b.userSays("Hi, um, can you get us a table, two people, seven tonight?")
	b.waitFor("the relayed answer", func() bool { return strings.Contains(b.said(), "Booked for 7pm.") })

	snaps, settings := frontend.requests()
	var tool llm.FunctionSchema
	for _, f := range snaps[0].Tools.Functions {
		if f.Name == DelegateToolName {
			tool = f
		}
	}
	if tool.Description != ExplicitToolDescription || len(tool.Required) != 1 || tool.Required[0] != "request" {
		t.Fatalf("delegate tool = %+v", tool)
	}
	if !strings.Contains(settings[0].SystemInstruction, ExplicitFrontendInstruction) {
		t.Fatalf("frontend instruction lacks the explicit guidance: %q", settings[0].SystemInstruction)
	}
	backendSnaps, _ := backend.requests()
	msgs := backendSnaps[0].Messages
	if len(msgs) != 1 || msgs[0].Role != llm.RoleUser || msgs[0].Text() != "Book a table for two at 7pm\n\n"+ExplicitRequestInstruction {
		t.Fatalf("backend request = %+v", msgs)
	}
}

func TestTranscriptStrategyHandsOverTheConversationSinceTheLastDelegation(t *testing.T) {
	s := NewTranscriptStrategy()
	conversation := llm.NewContext(
		llm.System("You are a test bot."),
		llm.User("Hi"),
		llm.Assistant("Hello! How can I help?"),
		llm.User("What's the weather in Paris?"),
	)
	params := llm.FunctionCallParams{Context: conversation}

	first, ok := s.ComposeRequest(params)
	want := "Voice conversation so far:\nUSER: Hi\nASSISTANT: Hello! How can I help?\nUSER: What's the weather in Paris?\n\n" + TranscriptRequestInstruction
	if !ok || first != want {
		t.Fatalf("first request = %q", first)
	}

	if _, ok := s.ComposeRequest(params); ok {
		t.Fatal("a delegation with nothing new produced a request")
	}

	conversation.Add(
		llm.ToolCalls(llm.ToolCallPart{ID: "d1", Name: DelegateToolName, Arguments: json.RawMessage("{}")}),
		llm.ToolResult("d1", DelegateToolName, `"Backend: 21 degrees"`),
		llm.Developer("Intermediate result of function delegate (call d1): \"Backend (working): get_weather()\""),
		llm.Assistant("It is 21 degrees."),
		llm.User("And in Rome?"),
	)
	second, ok := s.ComposeRequest(params)
	want = "Voice conversation since the previous delegation:\nASSISTANT: It is 21 degrees.\nUSER: And in Rome?\n\n" + TranscriptRequestInstruction
	if !ok || second != want {
		t.Fatalf("second request = %q", second)
	}

	conversation.Set([]llm.Message{llm.User("Start over: book a taxi")})
	third, ok := s.ComposeRequest(params)
	if !ok || !strings.HasPrefix(third, "Voice conversation so far:\nUSER: Start over: book a taxi\n") {
		t.Fatalf("request after a reset = %q", third)
	}
}

func TestTranscriptStrategyDelegationWithNothingNewStaysQuiet(t *testing.T) {
	frontend := &fakeLLM{name: "fast", fn: func(_ context.Context, snap llm.Snapshot) ([]llm.Event, error) {
		if _, ok := toolResult(snap.Messages, "d1"); !ok {
			return call("d1", DelegateToolName, "{}"), nil
		}
		if _, ok := toolResult(snap.Messages, "d2"); !ok {
			return call("d2", DelegateToolName, "{}"), nil
		}
		return say("done"), nil
	}}
	backend := &fakeLLM{name: "slow", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return say("Done."), nil
	}}
	w := newPair(frontend, backend, WithRespondOnDelegate(false))
	b := runBot(t, w)

	b.userSays("Do the thing")
	b.waitFor("both delegations to settle", func() bool {
		_, one := b.settled("d1")
		_, two := b.settled("d2")
		return one && two
	})
	r, _ := b.settled("d2")
	if r.Content != `{"status":"already_delegated"}` {
		t.Fatalf("second delegation result = %+v", r)
	}
	if backend.count() != 1 {
		t.Fatalf("backend ran %d times", backend.count())
	}
	msgs := w.BackendContext().Messages()
	if !strings.HasPrefix(msgs[0].Text(), "Voice conversation so far:\nUSER: Do the thing\n\n") {
		t.Fatalf("backend request = %q", msgs[0].Text())
	}
}

func TestWorksAsAnOrdinaryServiceInsideAnLLMSwitcher(t *testing.T) {
	frontend := &fakeLLM{name: "fast", fn: delegatingFrontend("{}", "Looking.", "Found it.")}
	backend := &fakeLLM{name: "slow", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return say("It is in the drawer."), nil
	}}
	w := newPair(frontend, backend)
	other := llmservice.New(&fakeLLM{name: "other", fn: func(context.Context, llm.Snapshot) ([]llm.Event, error) {
		return say("other"), nil
	}}, llmservice.WithName("other"))
	sw := pipeline.NewLLMSwitcher([]*llmservice.Service{w.Frontend(), other}, pipeline.ManualSwitchStrategy{})
	b := runBot(t, sw)

	b.userSays("Where are my keys?")
	b.waitFor("the relayed answer", func() bool { return strings.Contains(b.said(), "Found it.") })
	if backend.count() != 1 {
		t.Fatalf("backend ran %d times", backend.count())
	}
}
