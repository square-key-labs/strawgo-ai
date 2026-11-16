# Testing your bot

The `strawtest` package (`github.com/square-key-labs/strawgo-ai/src/strawtest`) lets you test a Strawgo bot with `go test`, without a phone, audio, or a paid API. It has two layers:

- **Processor tests** run one or more processors inside a real `pipeline.PipelineTask`, send them frames, and return the frames that came out of each end of the pipeline.
- **Scripted conversations** drive your own pipeline (aggregators, LLM service, and anything else you put in it) turn by turn with text, and check what the bot replied and which functions it called. You can run a script against a real LLM provider or against the deterministic `FakeLLM`.

Both layers are ports of ideas from Pipecat: `run_test` and `SleepFrame` from `pipecat/tests/utils.py`, and the scripted evals in `pipecat/evals`. The simulated caller, audio, and the eval CLI are not ported.

## Before you begin

- You know what a frame, a processor and a pipeline are. See [Core concepts](../getting-started/core-concepts.md).
- For scripted conversations, you know how the context aggregators work. See [Conversation context](conversation-context.md) and [Function calling](function-calling.md).

## Test a processor

`strawtest.RunProcessors` builds a pipeline around the processors you pass, starts it as a `PipelineTask`, waits for the `StartFrame` to reach the end, sends your frames, sends an `EndFrame`, and returns the frames that reached the end of the pipeline (`down`) and the frames that came back out of its start (`up`).

This processor answers every `TextFrame` with an upper-cased `LLMTextFrame` downstream and an acknowledgement upstream:

```go
package shout

import (
	"context"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Shout struct {
	*processors.BaseProcessor
}

func New() *Shout {
	s := &Shout{}
	s.BaseProcessor = processors.NewBaseProcessor("Shout", s)
	return s
}

func (s *Shout) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	tf, ok := frame.(*frames.TextFrame)
	if !ok || direction != frames.Downstream {
		return s.PushFrame(frame, direction)
	}
	if err := s.PushFrame(frames.NewTextFrame("ack"), frames.Upstream); err != nil {
		return err
	}
	return s.PushFrame(frames.NewLLMTextFrame(strings.ToUpper(tf.Text)), frames.Downstream)
}
```

Its test:

```go
package shout_test

import (
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/strawtest"

	"example.com/bot/shout"
)

func TestShout(t *testing.T) {
	down, up := strawtest.RunProcessors(t, []processors.FrameProcessor{shout.New()}, []frames.Frame{
		frames.NewTextFrame("hello"),
		strawtest.Sleep(20 * time.Millisecond),
		frames.NewTextFrame("again"),
	})

	strawtest.AssertFrameTypes(t, down, (*frames.LLMTextFrame)(nil), (*frames.LLMTextFrame)(nil))
	strawtest.AssertFrameTypes(t, up, (*frames.TextFrame)(nil), (*frames.TextFrame)(nil))
	if got := down[0].(*frames.LLMTextFrame).Text; got != "HELLO" {
		t.Fatalf("got %q", got)
	}
}
```

What happens:

- `strawtest.Sleep(d)` is not sent; the runner waits `d` before sending the next frame. Use it when a test needs time to pass, for example to let a timer in your processor fire. It is the equivalent of Pipecat's `SleepFrame`.
- Frames are sent downstream from the start of the pipeline. `strawtest.WithDirection(frames.Upstream)` sends them upstream from the end instead.
- Before the `EndFrame`, the runner sends a marker frame after your frames and waits for it to come out of the other end, so every frame you sent has been handled. This is why a processor must forward frames it does not recognise.
- The `StartFrame` and the runner's own `EndFrame` are left out of `down`. `strawtest.WithStartFrame()` keeps the `StartFrame`.
- An LLM service in the pipeline pushes an `LLMServiceMetadataFrame` upstream when it starts, so it is the first frame in `up`. Use `AssertFrameTypesIgnoring` to skip it.
- If the pipeline does not start, does not pass the frames through, or does not finish within the timeout (5 seconds by default, `strawtest.WithTimeout(d)` to change it), the runner cancels the task, waits for it to stop (up to the task's `CancelTimeout` plus one second), then fails the test with `t.Fatalf`, listing the frames received so far. The test never hangs, and the pipeline's goroutines do not outlive it.

Remember that system frames (for example `InterruptionFrame` or `UserStartedSpeakingFrame`) are handled ahead of queued data frames, so they can come out before data frames you sent earlier. That is how a real pipeline behaves.

### Options

| Option | Effect |
| --- | --- |
| `WithDirection(frames.Upstream)` | send the frames upstream from the end of the pipeline |
| `WithTaskConfig(cfg)` | use your `*pipeline.PipelineTaskConfig` (default: `DefaultPipelineTaskConfig()` with `CancelOnIdleTimeout` off) |
| `WithTimeout(d)` | limit for the whole run (default `strawtest.DefaultTimeout`, 5 s) |
| `WithStartFrame()` | keep the `StartFrame` in `down` |
| `WithoutEndFrame()` | do not send an `EndFrame`; the run ends when the pipeline ends itself, for example after a processor pushes an `EndTaskFrame` |

### Assert frame types

Expected types are given as typed nil pointers, the same convention as `PipelineTask.OnFrameReachedDownstream`.

- `AssertFrameTypes(t, got, want...)` checks that `got` has exactly these types, in this order.
- `AssertFrameTypesIgnoring(t, got, ignore, want...)` does the same after removing every frame whose type is in `ignore`.

Both call `t.Errorf` and return `false` on a mismatch. The message shows the first difference and both lists:

```text
frame types do not match at position 1: expected LLMTextFrame, received nothing
expected: [TextFrame, LLMTextFrame]
received: [TextFrame]
ignoring: [InterruptionFrame]
```

`strawtest.FrameTypes(list)` returns the same `[A, B]` text for your own messages.

## Script a conversation

A `strawtest.Script` builds your pipeline with your function, starts it, and runs a list of steps:

| Step | What it does |
| --- | --- |
| `Say(text)` | The user says `text`. The runner sends what a VAD and an STT service would produce: `UserStartedSpeakingFrame`, a final `TranscriptionFrame`, then `UserStoppedSpeakingFrame`. It waits for the first frame to come out of the end of the pipeline and for the transcription to be handled before it sends the next one, so the frames arrive in that order. |
| `ExpectReply(matchers...)` | Waits for the bot turn to finish, then checks the bot's reply against every matcher. `.Responses(n)` makes it wait until the turn holds at least `n` LLM responses that did not fail. |
| `ExpectCall(name, args)` | Waits for the next function call of the bot turn and checks its name and, optionally, its arguments. `.Returning(value)` makes the call return `value`. |
| `WaitForBot()` | Waits for the bot turn to finish without checking anything. Takes `.Responses(n)` too. |
| `Queue(frames...)` | Starts a new bot turn and sends the frames downstream from the start of the pipeline, for example an `LLMMessagesAppendFrame` with a developer message and `RunLLM` set, to test a run the application triggers between user turns. |
| `ExpectJudge(rubric)` | Waits for the bot turn to finish, then asks `Script.Judge` whether the transcript meets the rubric. |

Reply matchers:

| Matcher | Passes when the reply |
| --- | --- |
| `Contains(text)` | contains `text`, ignoring differences in spacing |
| `NotContains(text)` | does not contain `text` |
| `MatchesRegexp(expr)` | matches the regular expression |
| `Satisfies(description, func(reply string) bool)` | makes your function return true |

Argument matchers for `ExpectCall`:

| Matcher | Passes when |
| --- | --- |
| `ArgsInclude(map[string]any{...})` | every listed key is present with an equal value (compared as JSON, so `1` matches `1.0`); other arguments are ignored |
| `ArgsSatisfy(description, func(args map[string]any) bool)` | your function returns true |

### A complete example with the fake LLM

```go
package bot_test

import (
	"testing"
	"time"

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

func buildBot(provider llm.Provider) func() strawtest.Bot {
	return func() strawtest.Bot {
		pair := aggregators.NewContextAggregatorPair(
			llm.NewContext(llm.System("You are a weather assistant.")),
			aggregators.ContextAggregatorPairParams{
				User: aggregators.UserAggregatorParams{Strategies: turns.UserTurnStrategies{
					StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewVADUserTurnStartStrategy(true)},
					StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewSpeechTimeoutUserTurnStopStrategy(10 * time.Millisecond)},
				}},
			},
		)
		svc := llmservice.New(provider)
		return strawtest.Bot{
			Pipeline: pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), svc, pair.Assistant()}),
			LLM:      svc,
		}
	}
}

func TestWeatherConversation(t *testing.T) {
	fake := strawtest.NewFakeLLM(
		strawtest.FakeTurn{Reply: "Hello, how can I help?"},
		strawtest.FakeTurn{
			Calls: []strawtest.FakeCall{{Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}},
			Reply: "It is 21 degrees in Paris.",
		},
		strawtest.FakeTurn{Reply: "Goodbye!"},
	)

	strawtest.RunScript(t, strawtest.Script{
		Build: buildBot(fake),
		Steps: []strawtest.Step{
			strawtest.Say("Hi"),
			strawtest.ExpectReply(strawtest.Contains("help")),
			strawtest.Say("What is the weather in Paris?"),
			strawtest.ExpectCall("get_weather", strawtest.ArgsInclude(map[string]any{"city": "Paris"})).
				Returning(map[string]any{"temp_c": 21}),
			strawtest.ExpectReply(strawtest.Contains("21 degrees"), strawtest.NotContains("snow")),
			strawtest.Say("Thanks, bye"),
			strawtest.ExpectReply(strawtest.MatchesRegexp(`(?i)^goodbye`)),
		},
		StepTimeout: 5 * time.Second,
	})
}
```

The pipeline is the usual text part of a voice bot: user aggregator, LLM service, assistant aggregator. Leave the transports out; the script plays their role. A TTS service may stay in the pipeline if it can run in the test (the reply is then taken from its `TTSTextFrame`s), but most tests leave it out too.

The short speech timeout in the user aggregator only makes the test faster. With the default strategies each user turn ends 600 ms after `UserStoppedSpeakingFrame`, which also works. See [Turn-taking](turn-taking.md).

### The Bot your function returns

| Field | Meaning |
| --- | --- |
| `Pipeline` | your pipeline, required. The runner wraps it in a `PipelineTask`. |
| `LLM` | the `*llmservice.Service` in the pipeline. Required only when a step uses `Returning`: the runner registers a handler for that function name on it, which returns the canned values in script order. A handler you registered for the same name is replaced; your other handlers are kept. |
| `Config` | the `*pipeline.PipelineTaskConfig` for the task. Default: `DefaultPipelineTaskConfig()` with `CancelOnIdleTimeout` off. |
| `OnTask` | optional `func(*pipeline.PipelineTask)`, called with the task after it is created and before it runs. Use it for what needs the task: a greeting with `task.QueueFrame(frames.NewLLMRunFrame())`, a `flows.NewManager`, an observer with `task.SetObserver`, or your own task callbacks. The runner only subscribes with `OnFrameReachedDownstream` and `OnFrameReachedUpstream`, so `OnStarted`, `OnFinished` and `OnError` are yours. |

`Build` is called once per run, so every run starts with a fresh context.

### When a bot turn is finished

`Say` and `Queue` start a new bot turn; the bot turn before the first step is the greeting turn. The bot turn is finished when the runner has seen at least one `LLMFullResponseEndFrame` (or `n` that did not fail with `.Responses(n)`; a response counts as failed when the LLM service reports an `ErrorFrame` for it), every response that started has ended, every function call (seen through `FunctionCallInProgressFrame`) has its `FunctionCallResultFrame` or `FunctionCallCancelFrame`, one more response has started for every group of function results that asked the LLM to run again, and the bot is not speaking (`BotStartedSpeakingFrame` without `BotStoppedSpeakingFrame`, or `TTSStartedFrame` without `TTSStoppedFrame`). These are counts, so the order in which the frames from the two ends of the pipeline arrive does not matter. Before it declares the turn finished, the runner sends a marker frame up the pipeline from its end and waits for it to come out of the start: by then the LLM service has finished handling the response, so a function call it starts right after a response is not missed. No sleeps are involved.

The reply text is what the bot said in the turn: the `TTSTextFrame` text if there is any, otherwise the `LLMTextFrame` text, with the parts before and after each function call joined by a space.

Each waiting step has its own limit, `Script.StepTimeout` (default `strawtest.DefaultStepTimeout`, 10 s). When the bot never answers, the step fails with `timed out after 10s waiting for the bot to finish its turn`.

The runner only knows that one more response is coming when a function result asks for it. When something else starts another response in the same turn, for example application code that queues an `LLMMessagesAppendFrame` with `RunLLM` once the first answer has ended (through a `task.OnFrameReachedDownstream` hook for `LLMFullResponseEndFrame`), or the flows manager moving to the next node, the turn can look finished after the first response. Declare the extra response with `.Responses(2)` on the step that checks the reply:

```go
package bot_test

import "github.com/square-key-labs/strawgo-ai/src/strawtest"

var steps = []strawtest.Step{
	strawtest.Say("Save a note"),
	strawtest.ExpectReply(strawtest.Contains("saved"), strawtest.Contains("Anything else?")).Responses(2),
}
```

Two consequences of how this works:

- Every processor in your pipeline must forward frames it does not recognise, in both directions. If one does not, steps fail with a message saying that a frame never reached the other end of the pipeline.
- When a function result asks the LLM to run again, the assistant aggregator is what reruns it. Without an assistant aggregator in the pipeline, such a turn never finishes.

### The fake LLM

`strawtest.NewFakeLLM(turns...)` returns a `*strawtest.FakeLLM`, which implements `llm.Provider`, the interface every LLM service in Strawgo uses. Wrap it with `llmservice.New` like a real provider.

It hands out the scripted `FakeTurn`s in order, one per LLM run, whatever started the run: a user turn, a greeting with no user message, a developer message with `RunLLM`, the user aggregator's empty-turn recovery, or an idle prompt. Script one `FakeTurn` for each of them, in the order they happen.

- A turn with `Calls` answers its run with those function calls, then answers with `Reply` when the next request holds a result for one of those calls (the rerun after the function results). A placeholder result (`llm.ToolResultInProgress`) does not count as a result. A request that arrives before the calls are even recorded in the context also gets the `Reply`, because it belongs to the same run. Call IDs are unique across the whole run (`fake_call_1`, `fake_call_2`, ...).
- A turn without `Calls` answers with `Reply`.
- Any other request starts the next `FakeTurn`. Context changes such as summarization do not shift the order.
- A request when no `FakeTurn` is left returns an error, which the LLM service reports as a non-fatal `ErrorFrame`; the bot's reply for that run is empty.

A bot that greets first:

```go
package bot_test

import (
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/strawtest"
)

func greetingBot(build func() strawtest.Bot) func() strawtest.Bot {
	return func() strawtest.Bot {
		b := build()
		b.OnTask = func(task *pipeline.PipelineTask) {
			_ = task.QueueFrame(frames.NewLLMRunFrame())
		}
		return b
	}
}

var greetingFake = strawtest.NewFakeLLM(
	strawtest.FakeTurn{Reply: "Welcome to the weather line."},
	strawtest.FakeTurn{Reply: "Sure, which city?"},
)

var greetingSteps = []strawtest.Step{
	strawtest.ExpectReply(strawtest.Contains("Welcome")),
	strawtest.Say("Can I get a forecast?"),
	strawtest.ExpectReply(strawtest.Contains("which city")),
}
```

`fake.Requests()` returns every context snapshot it received, so you can check what the LLM was sent, for example that a function result reached it.

### Against a real provider

Pass a real provider to the same `Build` function, for example `openai.NewProvider(openai.LLMConfig{...})` (see [LLM providers](../providers/llm.md)), and register your real function handlers on the service. Steps without `Returning` leave your handlers alone. Replies from a real model vary from run to run, so prefer `Contains`, `MatchesRegexp`, `Satisfies` or a judge over exact text. Keep such tests behind a build tag or an environment variable check so that `go test ./...` stays free and offline.

### Plug in a judge

`ExpectJudge(rubric)` calls `Script.Judge`, which is any value implementing:

```go
package strawtest

import "context"

type Message struct {
	Role Role
	Text string
}

type Role string

type Verdict struct {
	Passed bool
	Reason string
}

type Judge interface {
	Judge(ctx context.Context, transcript []Message, rubric string) (Verdict, error)
}
```

The transcript holds the conversation so far, with roles `RoleUser`, `RoleBot`, `RoleFunctionCall` (for example `get_weather({"city":"Paris"})`) and `RoleFunctionResult` (for example `get_weather -> {"temp_c":21}`). The call gets a context limited to `StepTimeout`. `strawtest.JudgeFunc` turns a function into a `Judge`:

```go
package bot_test

import (
	"context"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/strawtest"
)

var politeJudge = strawtest.JudgeFunc(func(ctx context.Context, transcript []strawtest.Message, rubric string) (strawtest.Verdict, error) {
	for _, m := range transcript {
		if m.Role == strawtest.RoleBot && strings.Contains(strings.ToLower(m.Text), "please") {
			return strawtest.Verdict{Passed: true, Reason: "the bot said please"}, nil
		}
	}
	return strawtest.Verdict{Passed: false, Reason: "the bot never said please"}, nil
})
```

Strawgo ships no LLM-backed judge. To use one, implement `Judge` by sending the transcript and the rubric to a model of your choice and parsing its answer. A judge error or a failing verdict fails the step, and the verdict's reason is shown in the result.

### Results

`RunScript(t, script)` runs the script and, if it failed, reports the whole result with `t.Errorf`. It returns the `strawtest.Result` either way. `script.Run(ctx)` does the same without a `testing.TB`, for use outside `go test`.

A `Result` has:

| Field or method | Meaning |
| --- | --- |
| `Steps []StepResult` | one per step: `Index`, `Step` (a description), `Status` (`StepPassed`, `StepFailed`, `StepSkipped`), `Expected`, `Got` (what the bot said or called, or the judge's verdict), `Reason` |
| `Transcript []Message` | the conversation as the runner saw it |
| `Err` | set when the script could not run at all, for example `Build` returned no pipeline or the pipeline did not start |
| `Passed()` | true when `Err` is nil and every step passed |
| `String()` | the readable report below |

The first failing step stops the script; the remaining steps are marked skipped. A failure reads like this:

```text
script failed: 1 of 3 steps failed
transcript:
  user: Hi
  bot: Hello there
steps:
  ok   1. user says "Hi"
  FAIL 2. bot reply contains "Goodbye"
         reason:   reply does not satisfy: contains "Goodbye"
         expected: reply contains "Goodbye"
         got:      "Hello there"
  skip 3. user says "Are you there?"
```

## Limits

- Scripts are text only: no audio, no VAD or STT model, no transport. Test those parts with processor tests.
- The user turn ends when the user aggregator's stop strategy decides so, exactly as in a call; `Say` does not force it.
- Without a transport nothing sends `BotStoppedSpeakingFrame`, so the user aggregator treats the bot as still awaiting a response and each new `Say` after the first broadcasts an `InterruptionFrame`. With nothing in flight this has no effect, but you will see those frames if you observe the pipeline.
