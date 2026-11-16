# LLM with a backend

A fast, cheap model answers quickly but makes poor decisions with tools. A capable model reasons well but keeps the caller waiting. `llmwithbackend` pairs the two: a quick **frontend** LLM keeps the conversation moving, and hands anything that needs tools or careful reasoning to a slower **backend** LLM. The frontend says a short acknowledgement ("Let me check") right away, and relays the backend's answer when it arrives.

The design follows Pipecat's `LLMWithBackend`, ported onto Strawgo's function calling rather than onto a separate worker.

## Before you begin

- You have a working voice bot, for example from [Your first phone bot](../getting-started/first-phone-bot.md).
- You know [function calling](function-calling.md): handlers, `FunctionOptions`, delivery modes and cancellation. The frontend reaches the backend through one function, and all of those rules apply to it.

> [!NOTE]
> The feature is tested with scripted fake models only. No live vendor model has been run against these prompts in Strawgo. Test with your own models before you rely on it.

## A complete example

```go
package main

import (
	"context"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/llmwithbackend"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/anthropic"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

func runCall(ctx context.Context, s *transports.Session) error {
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: os.Getenv("CARTESIA_API_KEY"), VoiceID: os.Getenv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})

	frontend := openai.NewLLMService(openai.LLMConfig{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are the phone assistant of a travel agency. Flight searches and bookings are done for you in the background.",
	})
	backend := anthropic.NewLLMService(anthropic.LLMConfig{
		APIKey:       os.Getenv("ANTHROPIC_API_KEY"),
		Model:        "claude-sonnet-4-6",
		SystemPrompt: "You search and book flights for the caller.",
	})
	backend.RegisterFunction("search_flights", searchFlights)

	backendConversation := llm.NewContext()
	backendConversation.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{
		Name:        "search_flights",
		Description: "Find flights between two cities on a date.",
		Properties: map[string]any{
			"from": map[string]any{"type": "string"},
			"to":   map[string]any{"type": "string"},
			"date": map[string]any{"type": "string", "description": "YYYY-MM-DD"},
		},
		Required: []string{"from", "to", "date"},
	}}})

	llmSvc := llmwithbackend.New(frontend, backend, llmwithbackend.WithBackendContext(backendConversation))

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	return pipeline.NewPipelineTask(pipe).Run(ctx)
}

func searchFlights(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
	return llm.FunctionResult{Value: []map[string]any{
		{"flight": "AF1234", "departs": "08:10", "price_eur": 129},
	}}, nil
}
```

`llmSvc` goes exactly where a single LLM service would. The backend service is not put in the pipeline.

## What happens on a call

1. The caller asks "Is there a flight from Paris to Rome tomorrow morning?". The frontend sees the `delegate` tool and calls it.
2. The `delegate` handler starts. By default it reports a progress update that runs the frontend again, which says a short acknowledgement such as "Let me look into that".
3. The backend runs with its own conversation: the request, its tools, and its own tool loop. It calls `search_flights`, reads the result and writes its answer.
4. The answer becomes the result of the `delegate` call, marked `Backend: `. The frontend runs again and says it in its own words.

The conversation context stays valid throughout: one `delegate` call, one result. While the backend works, the result is `IN_PROGRESS`, exactly as for any slow function.

| Step | Where it shows |
| --- | --- |
| Delegation acknowledged | Intermediate result `{"status":"delegated"}`, which runs the frontend. |
| Backend text written beside a tool call | Silent intermediate result `Backend (note): ...`. |
| Backend tool call starting | Silent intermediate result `Backend (working): search_flights(date="2026-11-17", from="Paris", to="Rome")`. |
| Progress reported by a backend tool | Intermediate result `{"backend_function": "<name>", "progress": <value>}`, with the tool's own `RunLLM` and `Delivery`. |
| Backend answer | Final result `Backend: <answer>` followed by `BackendMessageRelayInstruction`. |

Silent updates are in the context, so if the caller asks "how is it going?" the frontend can say what the backend is doing without delegating again.

A frontend that already handles a question itself, such as small talk, never calls `delegate`, and the backend does not run.

## Choose what the backend receives

The request strategy decides what a `delegate` call sends. The backend keeps its own conversation across delegations, so each request only needs what is new.

| Strategy | The frontend calls `delegate` with | The backend receives |
| --- | --- | --- |
| `NewTranscriptStrategy()` (default) | no arguments | The user and assistant lines since the previous delegation, as `USER: ...` and `ASSISTANT: ...`, under "Voice conversation so far:" the first time. Tool messages and the backend's own updates are left out. |
| `NewExplicitStrategy()` | `request`, worded by the frontend model | That request as it stands. |

```go
llmSvc := llmwithbackend.New(frontend, backend, llmwithbackend.WithRequestStrategy(llmwithbackend.NewExplicitStrategy()))
```

The transcript strategy needs no wording skill from the frontend model and never drops a detail the caller said. The explicit strategy keeps backend requests short. When a transcript delegation finds nothing new since the previous one, it sends nothing and returns `{"status":"already_delegated"}` without running the frontend.

Each strategy's texts are fields you can change: `Instruction` (the guidance placed in the frontend's pairing instruction), `Description` (the tool description) and `RequestInstruction` (what is put after the request for the backend). You can also write your own `RequestStrategy`.

## Delivery, interruptions and cancellation

The `delegate` function is an ordinary function on the frontend. Set its options with `WithDelegateOptions`; the default is `llm.FunctionOptions{CancellableByLLM: true}`.

| Situation | What happens |
| --- | --- |
| The answer arrives while the bot speaks or the caller talks | The `Delivery` mode decides, as described in [Choose when a result is delivered](function-calling.md#choose-when-a-result-is-delivered). |
| The caller interrupts while the backend works | By default nothing is cancelled: the backend finishes and its answer is delivered. With `CancelOnInterruption: true`, the interruption cancels the delegation, the backend's request and its running tools are cancelled, and the result becomes `INTERRUPTED: ...`. |
| The caller says "never mind" | The frontend model calls `cancel_delegate`. The backend stops, its running tools see their `ctx` cancelled, and the result becomes `CANCELLED: ...`. |
| A second delegation arrives while the backend works | It waits for the first to finish, then runs with the backend's conversation up to date. |
| The delegation's `Timeout` expires | The backend is cancelled and the frontend gets an error result. |

When the backend is cancelled mid-turn, each tool call it had started is recorded in its conversation with `llm.ToolResultInterrupted`, so the backend's own context also stays valid. Each backend tool call runs exactly once; nothing is retried.

```go
llmSvc := llmwithbackend.New(frontend, backend, llmwithbackend.WithDelegateOptions(llm.FunctionOptions{
	CancellableByLLM: true,
	Delivery:         llm.ResultDeliveryWhenIdle,
}))
```

Turn off the immediate acknowledgement with `WithRespondOnDelegate(false)` when your frontend model already says something in the same response as its tool call.

## Backend tools and settings

- Register the backend's functions on the backend service with `RegisterFunction`, as usual. Their `Timeout` is honoured; the service-wide `WithFunctionCallTimeout` is not.
- Put the backend's tool schemas in the backend conversation (`WithBackendContext`), not in the caller's conversation. The frontend only gets `delegate`.
- The backend uses its service's settings, with `BackendPairingInstruction` appended to its system instruction for each request. Replace that text with `WithBackendPairingInstruction`.
- Limit runaway tool loops with `WithMaxBackendSteps` (default 10 model turns per delegation).

## Prompts

`New` appends `FrontendPairingInstruction` to the frontend's system instruction with `AppendSystemInstruction`, and adds the `delegate` tool with `AddBuiltinTool`. Neither changes the frontend's settings or the context's tool list, so `LLMSetToolsFrame` and `LLMUpdateSettingsFrame` keep them. The pairing instruction tells the model when to delegate, to speak as one assistant, to acknowledge briefly, and to relay each backend answer once. Your own system prompt should say, in plain words, what the backend is for.

Replace the pairing instruction with `WithFrontendPairingInstruction`. In your text, `{strategy}` is replaced by the strategy's guidance and `{stop}` by `StopInstruction` (or by nothing when the model cannot cancel). `w.FrontendInstruction()` returns the final text.

## Metrics and errors

Backend requests report TTFB, processing time and token usage in `MetricsFrame`s whose `Processor` is the backend service's name and `Model` its model, so you can tell them from the frontend's in [observability](observability.md) tooling.

A backend failure pushes a non-fatal `ErrorFrame` upstream with `Processor` set to the backend's name and the usual category (`rate_limit`, `content_filtered` for a refusal, and so on). The `delegate` call then returns an error result, `Backend: The work could not be completed.`, and the frontend tells the caller.

## Limits

- The frontend must be an `*llmservice.Service`. Speech-to-speech services (OpenAI Realtime, Gemini Live) are not supported as a frontend.
- Under an `LLMSwitcher`, pass `w.Frontend()` as one of the LLMs. Delegation is available only while that LLM is active; the other LLMs do not get the `delegate` tool.
- The backend handles one delegation at a time.
- If the backend finishes before the acknowledgement run starts, both frontend runs see the answer, and the model's relay instruction ("relay each message once") is what keeps it from being said twice.
- Predictive delegation from Pipecat (a classifier racing the frontend to delegate early) is not ported. Strawgo has no classifier stage, and eager end-of-turn speculation drops any speculative response that wants a tool call, so it never delegates speculatively.

## Next steps

- [Function calling](function-calling.md): delivery modes, progress and cancellation in detail.
- [Switching services](switching-services.md): `LLMSwitcher` and failover.
- [Pipeline reference](../reference/pipeline.md#llmwithbackend): every option and constant.
