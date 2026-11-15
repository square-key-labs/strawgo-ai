# Structured output and refusals

This guide covers three things you need when you treat LLM output as data rather than speech: asking for JSON that matches a schema, handling a model or provider that refuses to answer, and finding the request and response ids that providers ask for when you report a problem.

## Before you begin

- You have an LLM service from [LLM providers](../providers/llm.md), for example `openai.NewLLMService`.
- You know what a **pipeline** (the ordered list of processors that frames flow through) and a **frame** (a typed message in that pipeline) are. See [Core concepts](../getting-started/core-concepts.md).

> [!NOTE]
> Provider integrations are tested against mock servers at best, some not at all; none against a live vendor API (see the [status table](../providers/README.md)). Check structured output and refusal handling against your provider's live API before you rely on them.

## Structured output

`llm.Settings.ResponseFormat` asks the provider to return JSON instead of free text.

### A complete example: extract call notes after the call

Spoken replies should not be JSON, so the usual place for structured output in a voice bot is a side request: classify something, or extract data from the transcript. `RunInference` sends one request outside the pipeline and returns the text.

```go
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
)

type callNotes struct {
	Name      string `json:"name"`
	Callback  bool   `json:"callback_requested"`
	Sentiment string `json:"sentiment"`
}

var notesFormat = llm.JSONSchemaFormat("call_notes", map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name":               map[string]any{"type": "string"},
		"callback_requested": map[string]any{"type": "boolean"},
		"sentiment":          map[string]any{"type": "string", "enum": []string{"positive", "neutral", "negative"}},
	},
	"required":             []string{"name", "callback_requested", "sentiment"},
	"additionalProperties": false,
})

func main() {
	svc := openai.NewLLMService(openai.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-4.1-mini"})

	transcript := llm.NewContext(
		llm.System("Extract call notes from this phone conversation."),
		llm.User("Hi, it's Marta. My order never arrived and I'm pretty annoyed. Can someone call me back tomorrow?"),
	)

	ctx := context.Background()
	reply, err := svc.RunInference(ctx, transcript.Snapshot(), llm.Settings{ResponseFormat: notesFormat})
	if refusal, ok := llm.AsRefusal(err); ok {
		log.Fatalf("refused by %s: %s (%s)", refusal.Provider, refusal.Category, refusal.Message)
	}
	if errors.Is(err, llm.ErrResponseFormatUnsupported) {
		log.Fatal("this provider cannot do this response format")
	}
	if err != nil {
		log.Fatal(err)
	}

	var notes callNotes
	if err := json.Unmarshal([]byte(reply), &notes); err != nil {
		log.Fatalf("model returned invalid JSON: %v", err)
	}
	fmt.Printf("%+v\n", notes)
}
```

After a real call, build the snapshot from your conversation context instead: `svc.RunInference(ctx, conversation.Snapshot(), llm.Settings{ResponseFormat: notesFormat})`.

The settings you pass to `RunInference` are merged over the service's settings for that one request only. The service keeps its normal settings for the spoken conversation.

### The response format

| Constructor or field | Meaning |
| --- | --- |
| `llm.JSONSchemaFormat(name, schema)` | `Type: "json_schema"` with `Strict: true`; the usual choice |
| `&llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}` | any valid JSON object, no schema |
| `&llm.ResponseFormat{Type: llm.ResponseFormatText}` | plain text; the same as no format |
| `Name`, `Description` | sent to providers that accept them; the name defaults to `response` |
| `Schema` | a JSON Schema as `map[string]any` |
| `Strict` | asks the provider to follow the schema exactly |

To make every response of a service structured, set `ResponseFormat` in its settings, for example `openai.LLMConfig{Settings: llm.Settings{ResponseFormat: ...}}`, or during a call with `frames.NewLLMUpdateSettingsFrame`. Remember that TTS will then speak the JSON unless something in the pipeline consumes it.

### Provider support

| Provider | JSON object | JSON schema | How it is sent |
| --- | --- | --- | --- |
| OpenAI Chat Completions and the other OpenAI-compatible providers (Azure OpenAI, Groq, Together, Cerebras, DeepSeek, Fireworks, Mistral, NVIDIA, Ollama, OpenRouter, Perplexity, Qwen, SambaNova, Sarvam, xAI) | yes | yes | `response_format` |
| OpenAI Responses API | yes | yes | the response's text format |
| Google Gemini, Vertex AI | yes | yes | `responseMimeType: application/json`, plus `responseJsonSchema` for a schema |
| Anthropic | no | yes | `output_config.format` |
| AWS Bedrock | no | yes | the Converse output configuration |
| OpenAI Realtime, Gemini Live | no | no | not sent |

"Yes" means Strawgo sends the request field. Whether a given model honours it is up to the vendor; the OpenAI-compatible providers send the field even when their backend ignores it. Anthropic and Bedrock return an error wrapping `llm.ErrResponseFormatUnsupported` for a JSON object format, or a schema format with an empty schema. Check with `errors.Is`.

Always validate the JSON you get back, as in the example. Even with `Strict`, a refusal or a truncated response (stop reason `length`) does not contain the JSON you asked for.

## Refusals

A refusal is a response the model or the provider's safety system declined to produce. Strawgo turns every kind of refusal into one error type, `*llm.RefusalError`, and `llm.AsRefusal(err)` extracts its details:

```go
if refusal, ok := llm.AsRefusal(err); ok {
	log.Printf("%s refused: category=%s reason=%s detail=%s message=%s",
		refusal.Provider, refusal.Category, refusal.Reason, refusal.Detail, refusal.Message)
}
```

| `Category` | Meaning | Reported by |
| --- | --- | --- |
| `llm.RefusalContentFilter` | a content filter stopped the response | OpenAI-compatible providers (including Azure content filters), Gemini, Bedrock |
| `llm.RefusalPromptBlocked` | the prompt itself was blocked | Gemini, OpenAI-compatible providers (a filtered prompt) |
| `llm.RefusalModel` | the model declined | OpenAI-compatible providers (a `refusal` message), Anthropic (stop reason `refusal`) |
| `llm.RefusalRecitation` | the response was stopped for reciting source material | Gemini |
| `llm.RefusalGuardrail` | a guardrail intervened | Bedrock |

`Reason` holds the provider's own code, `Detail` lists the filter categories when the provider gives them, and `Message` holds any text the provider sent with the refusal.

### Refusals during a call

In the pipeline, the LLM service reports a refusal as a non-fatal `ErrorFrame`, so the call keeps going but the caller hears nothing for that turn. Catch it with `task.OnError` and say something:

```go
task.OnError(func(err error) {
	if _, ok := llm.AsRefusal(err); ok {
		if qerr := task.QueueFrame(frames.NewTTSSpeakFrame("Sorry, I can't help with that. Is there anything else?")); qerr != nil {
			log.Printf("queue fallback: %v", qerr)
		}
	}
})
```

`task.OnError` also receives every other error, fatal or not. See the [errors reference](../reference/errors.md).

### Stop reasons

Every response ends with a normalized stop reason:

| Constant | Value | Meaning |
| --- | --- | --- |
| `llm.StopReasonStop` | `stop` | the model finished normally |
| `llm.StopReasonLength` | `length` | the output token limit (`Settings.MaxTokens`) cut it off |
| `llm.StopReasonToolCalls` | `tool_calls` | the model called functions |
| `llm.StopReasonContentFilter` | `content_filter` | a filter stopped it |
| `llm.StopReasonRefusal` | `refusal` | the model refused |
| `llm.StopReasonOther` | `other` | anything else |

The LLM service logs a warning with the response and request ids when a response is cut off by the token limit. The stop reason, and the provider's raw value, are in the response metrics below.

## Request and response ids in metrics

With `EnableMetrics` on, the LLM service sends an `frames.LLMResponseMetricsData` in a `MetricsFrame` after each response:

| Field | Meaning |
| --- | --- |
| `Processor`, `Model` | the service name and the configured model |
| `ResponseID` | the provider's id for the response |
| `RequestID` | the HTTP request id (for the OpenAI-compatible providers, the `x-request-id` header) |
| `ServedModel` | the model the provider says actually served the request |
| `StopReason`, `RawStopReason` | the normalized and the provider's stop reason |

Providers fill in the fields they have. Log them with an observer, a component that sees every frame in the pipeline:

```go
package main

import (
	"log"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type responseLogger struct{}

func (*responseLogger) OnProcessFrame(pipeline.ProcessFrameEvent) {}
func (*responseLogger) OnPipelineStarted()                        {}
func (*responseLogger) OnPipelineStopped()                        {}

func (*responseLogger) OnPushFrame(e pipeline.PushFrameEvent) {
	m, ok := e.Frame.(*frames.MetricsFrame)
	if !ok {
		return
	}
	for _, d := range m.Data {
		if r, ok := d.(frames.LLMResponseMetricsData); ok {
			log.Printf("llm response=%s request=%s model=%s stop=%s (%s)",
				r.ResponseID, r.RequestID, r.ServedModel, r.StopReason, r.RawStopReason)
		}
	}
}

func attach(pipe *pipeline.Pipeline) *pipeline.PipelineTask {
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)
	obs := pipeline.NewTaskObserver()
	obs.AddObserver(&responseLogger{})
	task.SetObserver(obs)
	return task
}

func main() {
	_ = attach
}
```

Include the request id when you open a support ticket with the provider. See [Observability](observability.md) for other metrics and for exporting them.

## Next steps

- [Errors reference](../reference/errors.md): every error the pipeline reports and what the task does with it.
- [LLM reference](../reference/llm.md): `Settings`, `ResponseFormat`, `Refusal` and the provider interface.
- [Observability](observability.md): metrics, observers and tracing.
- [Switching services](switching-services.md): fail over to another LLM when one errors.
