package strawtest_test

import (
	"context"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/strawtest"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type fakeTB struct {
	testing.TB
	mu       sync.Mutex
	messages []string
	fatal    bool
}

func (f *fakeTB) Helper() {}

func (f *fakeTB) Errorf(format string, args ...any) {
	f.mu.Lock()
	f.messages = append(f.messages, fmt.Sprintf(format, args...))
	f.mu.Unlock()
}

func (f *fakeTB) Fatalf(format string, args ...any) {
	f.mu.Lock()
	f.messages = append(f.messages, fmt.Sprintf(format, args...))
	f.fatal = true
	f.mu.Unlock()
	runtime.Goexit()
}

func (f *fakeTB) output() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return strings.Join(f.messages, "\n")
}

func runWithFake(t *testing.T, fn func(tb testing.TB)) *fakeTB {
	t.Helper()
	tb := &fakeTB{TB: t}
	done := make(chan struct{})
	go func() {
		defer close(done)
		fn(tb)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("helper did not return")
	}
	return tb
}

type echo struct {
	*processors.BaseProcessor
	swallow bool
}

func newEcho(swallow bool) *echo {
	e := &echo{swallow: swallow}
	e.BaseProcessor = processors.NewBaseProcessor("Echo", e)
	return e
}

func (e *echo) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if tf, ok := frame.(*frames.TextFrame); ok && direction == frames.Downstream {
		if err := e.PushFrame(frames.NewTextFrame("ack "+tf.Text), frames.Upstream); err != nil {
			return err
		}
		return e.PushFrame(frames.NewLLMTextFrame(strings.ToUpper(tf.Text)), frames.Downstream)
	}
	if e.swallow && frame.Name() == "StrawtestBarrierFrame" {
		return nil
	}
	return e.PushFrame(frame, direction)
}

func TestRunProcessorsCollectsBothDirections(t *testing.T) {
	down, up := strawtest.RunProcessors(t, []processors.FrameProcessor{newEcho(false)}, []frames.Frame{
		frames.NewTextFrame("hi"),
		strawtest.Sleep(5 * time.Millisecond),
		frames.NewInterruptionFrame(),
		frames.NewTextFrame("there"),
	})

	strawtest.AssertFrameTypesIgnoring(t, down, []frames.Frame{(*frames.InterruptionFrame)(nil)},
		(*frames.LLMTextFrame)(nil), (*frames.LLMTextFrame)(nil))
	strawtest.AssertFrameTypes(t, up, (*frames.TextFrame)(nil), (*frames.TextFrame)(nil))
	var texts []string
	for _, f := range down {
		if tf, ok := f.(*frames.LLMTextFrame); ok {
			texts = append(texts, tf.Text)
		}
	}
	if strings.Join(texts, " ") != "HI THERE" {
		t.Fatalf("downstream texts = %v", texts)
	}
	if got := up[0].(*frames.TextFrame).Text; got != "ack hi" {
		t.Fatalf("first upstream text = %q", got)
	}
}

func TestRunProcessorsUpstreamDirection(t *testing.T) {
	_, up := strawtest.RunProcessors(t, []processors.FrameProcessor{newEcho(false)},
		[]frames.Frame{frames.NewTextFrame("up")}, strawtest.WithDirection(frames.Upstream))
	strawtest.AssertFrameTypes(t, up, (*frames.TextFrame)(nil))
}

func TestAssertFrameTypesReportsExpectedAndReceived(t *testing.T) {
	tb := runWithFake(t, func(tb testing.TB) {
		ok := strawtest.AssertFrameTypesIgnoring(tb,
			[]frames.Frame{frames.NewTextFrame("a"), frames.NewInterruptionFrame()},
			[]frames.Frame{(*frames.InterruptionFrame)(nil)},
			(*frames.TextFrame)(nil), (*frames.LLMTextFrame)(nil))
		if ok {
			t.Error("assertion should fail")
		}
	})
	want := []string{
		"frame types do not match at position 1: expected LLMTextFrame, received nothing",
		"expected: [TextFrame, LLMTextFrame]",
		"received: [TextFrame]",
		"ignoring: [InterruptionFrame]",
	}
	for _, w := range want {
		if !strings.Contains(tb.output(), w) {
			t.Fatalf("message missing %q:\n%s", w, tb.output())
		}
	}
}

func TestRunProcessorsFailsWhenAProcessorSwallowsFrames(t *testing.T) {
	swallower := newEcho(true)
	stoppedAtFailure := make(chan bool, 1)
	tb := runWithFake(t, func(tb testing.TB) {
		defer func() { stoppedAtFailure <- swallower.Context().Err() != nil }()
		strawtest.RunProcessors(tb, []processors.FrameProcessor{swallower},
			[]frames.Frame{frames.NewTextFrame("x")}, strawtest.WithTimeout(300*time.Millisecond))
		t.Error("RunProcessors should have stopped the test")
	})
	out := tb.output()
	if !tb.fatal || !strings.Contains(out, "did not pass through every processor") || !strings.Contains(out, "received down: [LLMTextFrame]") {
		t.Fatalf("unexpected failure message:\n%s", out)
	}
	if !<-stoppedAtFailure {
		t.Fatal("the pipeline was still running when the failure was reported")
	}
}

func quickTurns() turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewVADUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewSpeechTimeoutUserTurnStopStrategy(10 * time.Millisecond)},
	}
}

func bot(provider llm.Provider) func() strawtest.Bot {
	return botWith(provider, nil)
}

func botWith(provider llm.Provider, setup func(task *pipeline.PipelineTask, svc *llmservice.Service)) func() strawtest.Bot {
	return func() strawtest.Bot {
		pair := aggregators.NewContextAggregatorPair(llm.NewContext(llm.System("You are a test bot.")), aggregators.ContextAggregatorPairParams{
			User: aggregators.UserAggregatorParams{Strategies: quickTurns()},
		})
		if provider == nil {
			return strawtest.Bot{Pipeline: pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), pair.Assistant()})}
		}
		svc := llmservice.New(provider)
		b := strawtest.Bot{
			Pipeline: pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), svc, pair.Assistant()}),
			LLM:      svc,
		}
		if setup != nil {
			b.OnTask = func(task *pipeline.PipelineTask) { setup(task, svc) }
		}
		return b
	}
}

func TestScriptThreeTurnsWithFunctionCall(t *testing.T) {
	fake := strawtest.NewFakeLLM(
		strawtest.FakeTurn{Reply: "Hello, how can I help?"},
		strawtest.FakeTurn{
			Calls: []strawtest.FakeCall{{Name: "get_weather", Arguments: map[string]any{"city": "Paris", "days": 1}}},
			Reply: "It is 21 degrees in Paris.",
		},
		strawtest.FakeTurn{Reply: "Goodbye!"},
	)
	res := strawtest.RunScript(t, strawtest.Script{
		Build: bot(fake),
		Steps: []strawtest.Step{
			strawtest.Say("Hi"),
			strawtest.ExpectReply(strawtest.Contains("help")),
			strawtest.Say("What is the weather in Paris?"),
			strawtest.ExpectCall("get_weather", strawtest.ArgsInclude(map[string]any{"city": "Paris", "days": 1})).
				Returning(map[string]any{"temp_c": 21}),
			strawtest.ExpectReply(strawtest.Contains("21 degrees"), strawtest.NotContains("snow")),
			strawtest.Say("Thanks, bye"),
			strawtest.ExpectReply(strawtest.MatchesRegexp(`(?i)^goodbye`), strawtest.Satisfies("is short", func(r string) bool { return len(r) < 20 })),
		},
		StepTimeout: 3 * time.Second,
	})
	if !res.Passed() {
		return
	}

	var roles []string
	for _, m := range res.Transcript {
		roles = append(roles, string(m.Role))
	}
	wantRoles := "user,bot,user,function call,function result,bot,user,bot"
	if strings.Join(roles, ",") != wantRoles {
		t.Fatalf("transcript roles = %v\n%s", roles, res)
	}
	if res.Transcript[4].Text != `get_weather -> {"temp_c":21}` {
		t.Fatalf("function result entry = %q", res.Transcript[4].Text)
	}

	reqs := fake.Requests()
	if len(reqs) != 4 {
		t.Fatalf("FakeLLM got %d requests, want 4", len(reqs))
	}
	sawResult := false
	for _, m := range reqs[2].Messages {
		for _, p := range m.Parts {
			if tr, ok := p.(llm.ToolResultPart); ok && strings.Contains(tr.Content, "21") {
				sawResult = true
			}
		}
	}
	if !sawResult {
		t.Fatalf("the canned result did not reach the LLM context: %+v", reqs[2].Messages)
	}
}

func TestScriptFailingMatcherReportsTranscript(t *testing.T) {
	fake := strawtest.NewFakeLLM(strawtest.FakeTurn{Reply: "Hello there"})
	var res strawtest.Result
	tb := runWithFake(t, func(tb testing.TB) {
		res = strawtest.RunScript(tb, strawtest.Script{
			Build: bot(fake),
			Steps: []strawtest.Step{
				strawtest.Say("Hi"),
				strawtest.ExpectReply(strawtest.Contains("Goodbye")),
				strawtest.Say("never sent"),
			},
			StepTimeout: 3 * time.Second,
		})
	})
	if res.Passed() {
		t.Fatal("script should fail")
	}
	if res.Steps[1].Status != strawtest.StepFailed || res.Steps[1].Got != `"Hello there"` {
		t.Fatalf("step 2 = %+v", res.Steps[1])
	}
	if res.Steps[2].Status != strawtest.StepSkipped {
		t.Fatalf("step 3 should be skipped, got %s", res.Steps[2].Status)
	}
	for _, w := range []string{
		"script failed: 1 of 3 steps failed",
		"user: Hi",
		"bot: Hello there",
		`FAIL 2. bot reply contains "Goodbye"`,
		`reason:   reply does not satisfy: contains "Goodbye"`,
		`got:      "Hello there"`,
		`skip 3. user says "never sent"`,
	} {
		if !strings.Contains(tb.output(), w) {
			t.Fatalf("message missing %q:\n%s", w, tb.output())
		}
	}
}

func TestScriptTimesOutWhenTheBotNeverAnswers(t *testing.T) {
	start := time.Now()
	res := strawtest.Script{
		Build:       bot(nil),
		Steps:       []strawtest.Step{strawtest.Say("Hello?"), strawtest.ExpectReply()},
		StepTimeout: 200 * time.Millisecond,
	}.Run(context.Background())
	if res.Passed() {
		t.Fatal("script should fail")
	}
	if !strings.Contains(res.Steps[1].Reason, "timed out after 200ms waiting for the bot to finish its turn") {
		t.Fatalf("reason = %q", res.Steps[1].Reason)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("run took %v", elapsed)
	}
}

func TestScriptCallsTheJudge(t *testing.T) {
	for _, pass := range []bool{true, false} {
		t.Run(fmt.Sprint(pass), func(t *testing.T) {
			var gotRubric string
			var gotTranscript []strawtest.Message
			judge := strawtest.JudgeFunc(func(ctx context.Context, transcript []strawtest.Message, rubric string) (strawtest.Verdict, error) {
				gotRubric, gotTranscript = rubric, transcript
				return strawtest.Verdict{Passed: pass, Reason: "polite greeting"}, nil
			})
			res := strawtest.Script{
				Build: bot(strawtest.NewFakeLLM(strawtest.FakeTurn{Reply: "Hello!"})),
				Steps: []strawtest.Step{
					strawtest.Say("Hi"),
					strawtest.ExpectJudge("the bot greets the user"),
				},
				Judge:       judge,
				StepTimeout: 3 * time.Second,
			}.Run(context.Background())

			if gotRubric != "the bot greets the user" {
				t.Fatalf("judge rubric = %q", gotRubric)
			}
			if len(gotTranscript) != 2 || gotTranscript[0].Text != "Hi" || gotTranscript[1] != (strawtest.Message{Role: strawtest.RoleBot, Text: "Hello!"}) {
				t.Fatalf("judge transcript = %+v", gotTranscript)
			}
			if res.Passed() != pass {
				t.Fatalf("passed = %v, want %v\n%s", res.Passed(), pass, res)
			}
			if !strings.Contains(res.Steps[1].Got, "polite greeting") {
				t.Fatalf("judge reason missing from %q", res.Steps[1].Got)
			}
		})
	}
}

func userMessages(snap llm.Snapshot) int {
	n := 0
	for _, m := range snap.Messages {
		if m.Role == llm.RoleUser {
			n++
		}
	}
	return n
}

func TestScriptBotFirstGreeting(t *testing.T) {
	fake := strawtest.NewFakeLLM(
		strawtest.FakeTurn{Reply: "Welcome to the weather line."},
		strawtest.FakeTurn{Reply: "Sure, which city?"},
	)
	res := strawtest.RunScript(t, strawtest.Script{
		Build: botWith(fake, func(task *pipeline.PipelineTask, _ *llmservice.Service) {
			_ = task.QueueFrame(frames.NewLLMRunFrame())
		}),
		Steps: []strawtest.Step{
			strawtest.ExpectReply(strawtest.Contains("Welcome")),
			strawtest.Say("Can I get a forecast?"),
			strawtest.ExpectReply(strawtest.Contains("which city")),
		},
		StepTimeout: 3 * time.Second,
	})
	if !res.Passed() {
		return
	}
	reqs := fake.Requests()
	if len(reqs) != 2 || userMessages(reqs[0]) != 0 || userMessages(reqs[1]) != 1 {
		t.Fatalf("requests = %+v", reqs)
	}
	if res.Transcript[0] != (strawtest.Message{Role: strawtest.RoleBot, Text: "Welcome to the weather line."}) {
		t.Fatalf("transcript = %+v", res.Transcript)
	}
}

func TestScriptTaskHookQueuesFrame(t *testing.T) {
	fake := strawtest.NewFakeLLM(strawtest.FakeTurn{Reply: "Hello Ada, how can I help?"})
	var hooked *pipeline.PipelineTask
	res := strawtest.RunScript(t, strawtest.Script{
		Build: botWith(fake, func(task *pipeline.PipelineTask, _ *llmservice.Service) {
			hooked = task
			_ = task.QueueFrame(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.Developer("Greet Ada by name.")}, true))
		}),
		Steps:       []strawtest.Step{strawtest.ExpectReply(strawtest.Contains("Ada"))},
		StepTimeout: 3 * time.Second,
	})
	if hooked == nil {
		t.Fatal("OnTask was not called")
	}
	if !res.Passed() {
		return
	}
	reqs := fake.Requests()
	if len(reqs) != 1 {
		t.Fatalf("FakeLLM got %d requests, want 1", len(reqs))
	}
	last := reqs[0].Messages[len(reqs[0].Messages)-1]
	if last.Role != llm.RoleDeveloper || last.Text() != "Greet Ada by name." {
		t.Fatalf("last message = %+v", last)
	}
}

func TestScriptDeveloperRunBetweenUserTurns(t *testing.T) {
	fake := strawtest.NewFakeLLM(
		strawtest.FakeTurn{Calls: []strawtest.FakeCall{{Name: "lookup", Arguments: map[string]any{"q": "a"}}}, Reply: "First answer."},
		strawtest.FakeTurn{Reply: "Anything else?"},
		strawtest.FakeTurn{Calls: []strawtest.FakeCall{{Name: "lookup", Arguments: map[string]any{"q": "b"}}}, Reply: "Second answer."},
	)
	res := strawtest.RunScript(t, strawtest.Script{
		Build: bot(fake),
		Steps: []strawtest.Step{
			strawtest.Say("Look up a"),
			strawtest.ExpectCall("lookup", strawtest.ArgsInclude(map[string]any{"q": "a"})).Returning("A"),
			strawtest.ExpectReply(strawtest.Contains("First answer")),
			strawtest.Queue(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.Developer("The user is quiet; check in.")}, true)),
			strawtest.ExpectReply(strawtest.Contains("Anything else?"), strawtest.NotContains("First answer")),
			strawtest.Say("Look up b"),
			strawtest.ExpectCall("lookup", strawtest.ArgsInclude(map[string]any{"q": "b"})).Returning("B"),
			strawtest.ExpectReply(strawtest.Contains("Second answer")),
		},
		StepTimeout: 3 * time.Second,
	})
	if !res.Passed() {
		return
	}
	reqs := fake.Requests()
	if len(reqs) != 5 {
		t.Fatalf("FakeLLM got %d requests, want 5", len(reqs))
	}
	ids := map[string]int{}
	for _, m := range reqs[len(reqs)-1].Messages {
		for _, p := range m.Parts {
			if c, ok := p.(llm.ToolCallPart); ok {
				ids[c.ID]++
			}
		}
	}
	if len(ids) != 2 {
		t.Fatalf("call ids in the final context = %v, want two distinct ids", ids)
	}
	for id, n := range ids {
		if n != 1 {
			t.Fatalf("call id %s used %d times", id, n)
		}
	}
}

func TestScriptWaitsForFurtherResponse(t *testing.T) {
	fake := strawtest.NewFakeLLM(strawtest.FakeTurn{
		Calls: []strawtest.FakeCall{{Name: "save_note"}},
		Reply: "Saved. Anything else?",
	})
	res := strawtest.RunScript(t, strawtest.Script{
		Build: botWith(fake, func(task *pipeline.PipelineTask, svc *llmservice.Service) {
			svc.RegisterFunction("save_note", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
				if err := task.QueueFrame(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.Developer("Tell the user the note is saved.")}, true)); err != nil {
					return llm.FunctionResult{}, err
				}
				runLLM := false
				return llm.FunctionResult{Value: "saved", RunLLM: &runLLM}, nil
			})
		}),
		Steps: []strawtest.Step{
			strawtest.Say("Save a note"),
			strawtest.ExpectCall("save_note"),
			strawtest.ExpectReply(strawtest.Contains("Saved.")).Responses(2),
		},
		StepTimeout: 3 * time.Second,
	})
	if res.Passed() && len(fake.Requests()) != 2 {
		t.Fatalf("FakeLLM got %d requests, want 2", len(fake.Requests()))
	}
}
