# LLM reference

This page documents the `llm` package (messages, context, tools, settings, the `Provider` interface and its events), the `llmservice.Service` processor that runs any provider, and the `openaicompat` package that most HTTP providers are built on.

An **LLM** (large language model) produces the bot's replies. The **context** is the list of messages so far. A **provider** is the adapter that talks to one vendor's API. **Function calling** (also called tool use) lets the model ask your Go code to run a named function.

Import paths: `github.com/square-key-labs/strawgo-ai/src/llm`, `.../src/services/llmservice`, `.../src/services/openaicompat`.

## Quick example

This complete program implements a toy provider and runs one inference through `llmservice.Service`. It needs no API key and prints `You said: hello`.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

type echoProvider struct{}

func (echoProvider) Name() string { return "Echo" }

func (echoProvider) Stream(ctx context.Context, snap llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	last := ""
	if n := len(snap.Messages); n > 0 {
		last = snap.Messages[n-1].Text()
	}
	emit(llm.ResponseInfo{Model: settings.Model})
	emit(llm.TextDelta{Text: "You said: " + last})
	emit(llm.UsageEvent{Usage: llm.Usage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7}})
	emit(llm.Done{StopReason: llm.StopReasonStop})
	return ctx.Err()
}

func main() {
	svc := llmservice.New(echoProvider{}, llmservice.WithSettings(llm.Settings{Model: "echo-1"}))
	conversation := llm.NewContext(llm.System("Be brief."), llm.User("hello"))
	reply, err := svc.RunInference(context.Background(), conversation.Snapshot(), llm.Settings{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(reply)
}
```

## Messages

```text
type Message struct {
	Role  Role
	Name  string
	Parts []Part
}
```

| Role constant | Value | Use |
| --- | --- | --- |
| `RoleSystem` | `"system"` | Instructions. |
| `RoleDeveloper` | `"developer"` | Instructions that some providers treat separately from system. |
| `RoleUser` | `"user"` | What the caller said. |
| `RoleAssistant` | `"assistant"` | What the bot said, or its tool calls. |
| `RoleTool` | `"tool"` | A tool result. |

Constructors:

| Function | Result |
| --- | --- |
| `Text(role Role, text string) Message` | One `TextPart` with the given role. |
| `System(text)`, `Developer(text)`, `User(text)`, `Assistant(text)` | `Text` with that role. |
| `ToolCalls(calls ...ToolCallPart) Message` | An assistant message holding tool calls. |
| `ToolResult(toolCallID, name, content string) Message` | A tool message with one `ToolResultPart`. |

Accessors: `m.Text()` joins all `TextPart`s; `m.ToolCalls()` returns the `ToolCallPart`s; `m.ToolResult()` returns the first `ToolResultPart` and whether there is one.

Part types (each implements the sealed `Part` interface):

| Part | Fields |
| --- | --- |
| `TextPart` | `Text string` |
| `ImagePart` | `URL string`, `Data []byte`, `MIME string`, `Detail string` |
| `AudioPart` | `Data []byte`, `MIME string`, `SampleRate int` |
| `FilePart` | `URL string`, `Data []byte`, `MIME string`, `Name string` |
| `ToolCallPart` | `ID string`, `Name string`, `Arguments json.RawMessage`, `Signature string` |
| `ToolResultPart` | `ToolCallID string`, `Name string`, `Content string`, `IsError bool` |
| `ThoughtPart` | `Text string`, `Signature string`, `Redacted bool` |
| `ProviderPart` | `Provider string`, `Raw json.RawMessage` (passed through only to the named provider) |

Which parts a provider sends depends on the provider; see [LLM providers](../providers/llm.md).

## Context and Snapshot

`Context` is the shared, mutable conversation. It is safe for concurrent use, and every change increments a version number. Providers never see the `Context` itself: they get an immutable `Snapshot`.

```text
func NewContext(messages ...Message) *Context
func (c *Context) Add(messages ...Message)
func (c *Context) Set(messages []Message)
func (c *Context) Transform(fn func([]Message) []Message)
func (c *Context) Messages() []Message
func (c *Context) Len() int
func (c *Context) SetTools(tools Tools)
func (c *Context) Tools() Tools
func (c *Context) SetToolChoice(choice ToolChoice)
func (c *Context) ReplaceToolResult(toolCallID, content string, isError bool) bool
func (c *Context) Snapshot() Snapshot
```

| Method | Description |
| --- | --- |
| `Add` | Append messages. |
| `Set` | Replace all messages. |
| `Transform` | Replace messages with `fn(copy)`; `fn` gets copies it may edit. |
| `Messages` | The current messages. Do not modify the returned slice. |
| `SetTools`, `SetToolChoice` | Replace the tool list or tool choice. |
| `ReplaceToolResult` | Overwrite the most recent tool result for `toolCallID`. Returns false if there is none. |
| `Snapshot` | A consistent copy of messages, tools, tool choice and version. |

```text
type Snapshot struct {
	Messages   []Message
	Tools      Tools
	ToolChoice ToolChoice
	Version    uint64
}
```

### Saving and restoring a context

```text
const ContextFormatVersion = 1
func MarshalContext(c *Context, opts ...MarshalOption) ([]byte, error)
func MarshalSnapshot(s Snapshot, opts ...MarshalOption) ([]byte, error)
func UnmarshalContext(data []byte) (*Context, error)
func OmitBinary() MarshalOption
func StripProviderState() MarshalOption
const ToolResultInProgress = "IN_PROGRESS"
var ErrNilContext error
var ErrUnsupportedContextVersion error
var ErrUnknownPartType error
```

| Name | Description |
| --- | --- |
| `MarshalContext` | Encode the context's current snapshot (messages, tools, tool choice) as versioned JSON. |
| `MarshalSnapshot` | The same for a `Snapshot` you already hold. |
| `UnmarshalContext` | Build a new `Context` from that JSON. Its version counter starts at 0. |
| `OmitBinary()` | Drop the bytes of image, audio and file parts. A part with a URL keeps the URL; a part with only bytes becomes a `TextPart` such as `[image/png image omitted, 48213 bytes]`. |
| `StripProviderState()` | Drop `ThoughtPart`s and `ProviderPart`s, clear `ToolCallPart.Signature`, and leave out messages left with no parts. Use it when the restored context goes to a different provider, model or API key, which can reject another provider's opaque data. |
| `ToolResultInProgress` | The placeholder content of a tool result whose call has not finished. On save it is written as `ToolResultInterrupted`, since nothing completes it after a restore. |
| `ErrNilContext` | Returned by `MarshalContext(nil)`. |
| `ErrUnsupportedContextVersion` | Returned (wrapped) when `version` is missing or newer than `ContextFormatVersion`. |
| `ErrUnknownPartType` | Returned (wrapped) when a part's `type` is not one listed below. |

Every part is an object with a `type` field:

| `type` | Go part | JSON fields |
| --- | --- | --- |
| `text` | `TextPart` | `text` |
| `image` | `ImagePart` | `url`, `data` (base64), `mime`, `detail` |
| `audio` | `AudioPart` | `data` (base64), `mime`, `sample_rate` |
| `file` | `FilePart` | `url`, `data` (base64), `mime`, `name` |
| `tool_call` | `ToolCallPart` | `id`, `name`, `arguments` (the raw arguments as a string), `signature` |
| `tool_result` | `ToolResultPart` | `tool_call_id`, `name`, `content`, `is_error` |
| `thought` | `ThoughtPart` | `text`, `signature`, `redacted` |
| `provider` | `ProviderPart` | `provider`, `raw` (the raw JSON as a string) |

Ids, signatures and provider raw bytes are kept exactly unless you pass `StripProviderState()`, so by default a restored context is only safe to send to the provider and model that produced it. The top-level object is `{"version":1,"messages":[{"role","name","parts"}],"tools":{"functions","custom"},"tool_choice":{"mode","name"}}`. See [the guide](../guides/conversation-context.md#save-and-restore-a-conversation) for an example.

Inside a pipeline you rarely call these directly: the [context aggregators](aggregators.md) add user and assistant turns, and frames such as `LLMMessagesAppendFrame` change the context in order with the conversation. See the [conversation context guide](../guides/conversation-context.md).

## Tools

```text
type FunctionSchema struct {
	Name        string
	Description string
	Properties  map[string]any
	Required    []string
	Strict      bool
}

type Tools struct {
	Functions []FunctionSchema
	Custom    map[string][]json.RawMessage
}

type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}
```

| Item | Meaning |
| --- | --- |
| `FunctionSchema.Properties` | JSON Schema properties of the arguments object. |
| `FunctionSchema.Required` | Names of required properties. |
| `FunctionSchema.Strict` | Ask providers that support it to enforce the schema exactly. |
| `FunctionSchema.JSONSchema()` | The arguments schema as `{"type":"object","properties":...,"required":[...]}`; nil values become empty. |
| `Tools.Custom` | Raw provider-specific tool definitions, keyed by provider. |
| `Tools.Empty()` | True when there are no functions and no custom tools. |

| `ToolChoiceMode` | Value | Meaning |
| --- | --- | --- |
| `ToolChoiceUnset` | `""` | Provider default. |
| `ToolChoiceAuto` | `"auto"` | The model decides. |
| `ToolChoiceNone` | `"none"` | Do not call tools. |
| `ToolChoiceRequired` | `"required"` | Must call some tool. |
| `ToolChoiceFunction` | `"function"` | Must call the tool in `Name`. |

```go
func weatherTools() llm.Tools {
	return llm.Tools{Functions: []llm.FunctionSchema{{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Properties: map[string]any{
			"city": map[string]any{"type": "string", "description": "City name"},
		},
		Required: []string{"city"},
	}}}
}
```

## Settings

```text
type Settings struct {
	Model             string
	SystemInstruction string
	Temperature       *float64
	TopP              *float64
	MaxTokens         *int
	Seed              *int
	ResponseFormat    *ResponseFormat
	Extra             map[string]any
}
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Model` | `string` | provider default | Model ID. |
| `SystemInstruction` | `string` | none | System prompt the provider adds before the messages. |
| `Temperature` | `*float64` | nil (provider default) | Sampling temperature. |
| `TopP` | `*float64` | nil | Nucleus sampling. |
| `MaxTokens` | `*int` | nil | Output token limit. |
| `Seed` | `*int` | nil | Sampling seed, where supported. |
| `ResponseFormat` | `*ResponseFormat` | nil (plain text) | Structured output. See below. |
| `Extra` | `map[string]any` | none | Provider-specific request fields. In `openaicompat`, each key is set on the request body, and the keys of an `"extra_body"` map are merged in. |

`s.Merge(delta)` returns `s` with every non-empty string and non-nil pointer from `delta` applied, and `Extra` maps merged key by key. This is how `LLMUpdateSettingsFrame`, `RunInference` and `UpdateSettings` apply changes.

### ResponseFormat

```text
type ResponseFormat struct {
	Type        string
	Name        string
	Description string
	Schema      map[string]any
	Strict      bool
}
func JSONSchemaFormat(name string, schema map[string]any) *ResponseFormat
```

| `Type` constant | Value | Meaning |
| --- | --- | --- |
| `ResponseFormatText` | `"text"` | Plain text. |
| `ResponseFormatJSONObject` | `"json_object"` | Any JSON object. |
| `ResponseFormatJSONSchema` | `"json_schema"` | JSON matching `Schema`. |

`JSONSchemaFormat` sets `Type` to `json_schema` and `Strict` to true. `Structured()` reports whether the format is JSON object or JSON schema; `SchemaName()` returns `Name` or `"response"`. Providers that cannot honour a structured format return an error wrapping `llm.ErrResponseFormatUnsupported` (the Anthropic and Bedrock providers do this for `json_object`, or for `json_schema` with an empty schema). See the [structured output guide](../guides/structured-output-and-refusals.md).

## Provider interface and events

```text
type Provider interface {
	Name() string
	Stream(ctx context.Context, snapshot Snapshot, settings Settings, emit func(Event)) error
}
```

`Stream` sends one request and calls `emit` for each event, in order, on the calling goroutine. It returns when the response is complete, with nil, an error, or `ctx.Err()`. To write your own provider, see [custom providers](../providers/custom.md).

| Event type | Fields | What `llmservice.Service` does with it |
| --- | --- | --- |
| `TextDelta` | `Text string` | Pushes an `LLMTextFrame` (empty text is skipped). |
| `ThoughtDelta` | `Text string`, `Signature string` | Pushes `LLMThoughtStartFrame` (first one), `LLMThoughtTextFrame`; the thought ends with `LLMThoughtEndFrame` carrying the last signature. |
| `ToolCall` | `ID string`, `Name string`, `Arguments []byte`, `Signature string` | Collects it; calls run after the stream ends. |
| `UsageEvent` | `Usage Usage` | Reported as `LLMUsageMetricsData` when usage metrics are on. |
| `ResponseInfo` | `ResponseID string`, `RequestID string`, `Model string` | Merged (non-empty fields win) and reported in `LLMResponseMetricsData`. |
| `Refusal` | `Category RefusalCategory`, `Provider`, `Reason`, `Detail`, `Message string` | The first one becomes a `*RefusalError` if `Stream` returns nil. |
| `Done` | `StopReason string`, `RawStopReason string` | Stop reason for metrics; `StopReasonLength` logs a truncation warning. |

`Usage` fields: `PromptTokens`, `CompletionTokens`, `TotalTokens`, `CacheReadTokens`, `CacheCreationTokens`, `ReasoningTokens` (all `int`).

### Stop reasons

| Constant | Value |
| --- | --- |
| `StopReasonStop` | `"stop"` |
| `StopReasonLength` | `"length"` |
| `StopReasonToolCalls` | `"tool_calls"` |
| `StopReasonContentFilter` | `"content_filter"` |
| `StopReasonRefusal` | `"refusal"` |
| `StopReasonOther` | `"other"` |

`RawStopReason` keeps the vendor's own string.

### Refusals

A **refusal** is when the model or the vendor's safety system declines to answer.

| `RefusalCategory` | Value | Meaning |
| --- | --- | --- |
| `RefusalContentFilter` | `"content_filter"` | The output was filtered. |
| `RefusalPromptBlocked` | `"prompt_blocked"` | The input was rejected. |
| `RefusalModel` | `"model_refusal"` | The model itself declined. |
| `RefusalRecitation` | `"recitation"` | Blocked for reproducing source material. |
| `RefusalGuardrail` | `"guardrail"` | A configured guardrail intervened. |

`RefusalError{Refusal Refusal; Err error}` is the error form; `Unwrap` returns `Err`. `AsRefusal(err) (Refusal, bool)` finds one anywhere in an error chain. See [errors](errors.md#refusals).

## llmservice.Service

`llmservice.Service` is the processor that runs a provider in a pipeline. Every built-in LLM constructor returns one.

```text
func New(provider llm.Provider, opts ...Option) *Service
func WithName(name string) Option
func WithSettings(settings llm.Settings) Option
func WithSequentialFunctionCalls() Option
func WithFunctionCallTimeout(d time.Duration) Option
```

| Option | Default | Effect |
| --- | --- | --- |
| `WithName` | `provider.Name()` | Processor name, used in logs, metrics and `Service` matching on settings frames. |
| `WithSettings` | zero `llm.Settings` | Initial settings. |
| `WithSequentialFunctionCalls` | off (parallel) | Run the function calls from one response one after another, in order. |
| `WithFunctionCallTimeout` | 0 (none) | Timeout for function calls whose own `FunctionOptions.Timeout` is 0. |

| Method | Signature | Description |
| --- | --- | --- |
| `Provider` | `func (s *Service) Provider() llm.Provider` | The wrapped provider. |
| `Settings` | `func (s *Service) Settings() llm.Settings` | Current settings. |
| `UpdateSettings` | `func (s *Service) UpdateSettings(delta llm.Settings)` | Merge `delta` into the settings. |
| `RegisterFunction` | `func (s *Service) RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions)` | Register a handler. Only the first `opts` value is used. An empty `name` registers a catch-all for every unregistered name. Registering an existing name replaces it. |
| `UnregisterFunction` | `func (s *Service) UnregisterFunction(name string)` | Remove a handler (`""` removes the catch-all). |
| `HasFunction` | `func (s *Service) HasFunction(name string) bool` | Whether a call to `name` would find a handler. |
| `RunInference` | `func (s *Service) RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error)` | One out-of-band request with `Settings().Merge(delta)`. Returns the concatenated text. Pushes no frames and runs no functions. |

Frames it handles:

| Frame | Action |
| --- | --- |
| `LLMContextFrame` (downstream, non-nil context) | Snapshot the context and stream a response. Emits `LLMFullResponseStartFrame`, text and thought frames, `LLMFullResponseEndFrame`, then metrics, then runs any tool calls. |
| `LLMUpdateSettingsFrame` | If `Service` is nil or names this service, merge `Delta` and consume the frame; otherwise forward it. |
| `InterruptionFrame` | Cancel running calls registered with `CancelOnInterruption`. The running stream is cancelled by the processor engine. |
| `CancelFrame` | Cancel every running call. |

On a stream error the service pushes a non-fatal `ErrorFrame` ("generation failed", or "response refused" for a refusal). A cancelled context is not an error. `Stop` cancels running calls and waits up to 2 s for them.

### Function handlers

```text
type FunctionHandler func(ctx context.Context, params FunctionCallParams) (FunctionResult, error)

type FunctionCallParams struct {
	ToolCallID     string
	FunctionName   string
	Arguments      map[string]any
	RawArguments   json.RawMessage
	Context        *Context
	ReportProgress func(FunctionResult) bool
}

type FunctionResult struct {
	Value    any
	RunLLM   *bool
	Delivery ResultDelivery
}

type FunctionOptions struct {
	CancelOnInterruption bool
	Timeout              time.Duration
	CancellableByLLM     bool
	Delivery             ResultDelivery
}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `FunctionResult.Value` | nil | The result written to the context. Strings are JSON-quoted, `json.RawMessage` and `[]byte` are used as is, other values are JSON-encoded, nil becomes `"COMPLETED"`. |
| `FunctionResult.RunLLM` | nil (run) | Set to a pointer to false to store the result without asking the LLM to respond. |
| `FunctionOptions.CancelOnInterruption` | false | Cancel the handler's `ctx` when the user interrupts. |
| `FunctionOptions.Timeout` | 0 (use the service timeout) | Per-call timeout. |
| `FunctionOptions.CancellableByLLM` | false | While a call runs, offer the model a `cancel_<name>` tool that stops it. Ignored when `CancelOnInterruption` is true. |
| `FunctionOptions.Delivery` | `ResultDeliveryDefault` | When a result is acted on. See the table below. |
| `FunctionResult.Delivery` | `ResultDeliveryDefault` (use the function's) | Overrides `FunctionOptions.Delivery` for this result. |
| `FunctionCallParams.ReportProgress` | | Sends an intermediate result while the call runs. Silent unless the value's `RunLLM` points to true. Returns false once the call has settled. |

| `ResultDelivery` | Value | Effect |
| --- | --- | --- |
| `ResultDeliveryDefault` | `""` | Run the LLM after the user stops speaking and the bot finishes speaking (the behaviour before delivery modes existed). |
| `ResultDeliveryInterrupt` | `"interrupt"` | Interrupt the bot's speech or reply in progress and run the LLM now. Falls back to the default when interruptions are disabled. |
| `ResultDeliveryWhenIdle` | `"when_idle"` | Run the LLM once the bot is silent and no user turn is in progress. A reply to a user turn that started before the result arrived counts as the delivery. |
| `ResultDeliverySilent` | `"silent"` | Write the result into the context only. Overrides `RunLLM`. |

Cancel tool helpers: `CancelToolPrefix` (`"cancel_"`), `CancelToolName(name)` and `CancelToolSchema(name)` build the tool the service offers for a `CancellableByLLM` function. `ToolResultCancelled` is the tool result recorded for a call the model cancelled.

`DefaultFunctionOptions()` returns the zero value.

How a call runs:

1. After the stream ends, the service broadcasts `FunctionCallsStartedFrame`, then one `FunctionCallInProgressFrame` per call.
2. Each handler runs in its own goroutine (or in order with `WithSequentialFunctionCalls`), with a `ctx` derived from the processor's lifetime, not from the interruption context.
3. On return the service broadcasts `FunctionCallResultFrame`. If the handler returned an error, the frame's `Error` is set and a nil `Value` becomes `{"error": "<message>"}`.
4. If the handler's `ctx` was cancelled (interruption, `CancelFrame`, `Stop`), it broadcasts `FunctionCallCancelFrame` instead; the context then records `llm.ToolResultInterrupted`. If the model cancelled the call through its cancel tool, the frame's `Result` is `llm.ToolResultCancelled` and the context records that.
5. The result frame carries the effective `Delivery`. A silent result has `RunLLM` set to false. Progress reported with `ReportProgress` is broadcast as a `FunctionCallResultFrame` whose `Properties.IsFinal` is false.
6. While a `CancellableByLLM` call runs, each request the service sends includes `cancel_<name>` in its tools. The context's tool list is not changed.
7. A timeout, an unregistered name, invalid JSON arguments or a panic all produce a result frame with `Error` set.

The assistant aggregator writes results into the context and re-runs the LLM. See the [function calling guide](../guides/function-calling.md).

```go
func registerWeather(svc *llmservice.Service) {
	svc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		city, _ := p.Arguments["city"].(string)
		if city == "" {
			return llm.FunctionResult{}, errors.New("city is required")
		}
		return llm.FunctionResult{Value: map[string]any{"city": city, "forecast": "sunny"}}, nil
	}, llm.FunctionOptions{CancelOnInterruption: true, Timeout: 10 * time.Second})
}
```

## openaicompat

`openaicompat` implements `llm.Provider` for any API that speaks the OpenAI Chat Completions streaming format. Most built-in LLM providers are thin wrappers around it.

```text
func NewService(cfg ServiceConfig, d Defaults, opts ...Option) *llmservice.Service
func New(cfg Config, opts ...Option) *Provider
func Wrap(p llm.Provider, settings llm.Settings, opts ...llmservice.Option) *llmservice.Service
func (cfg ServiceConfig) Resolve(d Defaults) (Config, llm.Settings)
```

`NewService` resolves `cfg` against `d`, builds a `Provider`, and wraps it in an `llmservice.Service` with the resolved settings and `cfg.ServiceOptions`.

`ServiceConfig` (what you pass):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent as `Authorization: Bearer <key>` unless `WithAuthHeader` changes it. |
| `BaseURL` | `string` | `Defaults.BaseURL` | API root; `/chat/completions` is appended. |
| `Model` | `string` | `Defaults.Model` | Overrides the model. |
| `SystemPrompt` | `string` | none | Sets `Settings.SystemInstruction`. |
| `Settings` | `llm.Settings` | zero | Merged over `Defaults.Settings`. |
| `Headers` | `map[string]string` | none | Extra request headers. |
| `HTTPClient` | `*http.Client` | a shared client | HTTP client. |
| `Retry` | `RetryPolicy` | zero (no retry) | Connection retry policy. |
| `ServiceOptions` | `[]llmservice.Option` | none | Passed to `llmservice.New`. |

Settings resolve in this order: `Defaults.Settings`, then `Defaults.Model`, then `cfg.Settings` merged, then `cfg.Model`, then `cfg.SystemPrompt`.

`Defaults` (set by a provider package): `Name string`, `BaseURL string`, `Model string`, `Settings llm.Settings`.

`Config` (for `New`): `Name string` (default `"OpenAICompatible"`), `BaseURL string`, `APIKey string`, `Headers map[string]string`, `HTTPClient *http.Client`, `Retry RetryPolicy`.

`RetryPolicy` is `httpretry.Policy`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `MaxAttempts` | `int` | 0 | Total attempts. Retrying is on only when greater than 1. |
| `FirstByteTimeout` | `time.Duration` | 0 (none) | Give up on an attempt that has not received its first response byte in this time. |
| `Backoff` | `time.Duration` | 200 ms (`DefaultBackoff`) | First wait between attempts; doubles each time. |
| `MaxBackoff` | `time.Duration` | 2 s (`DefaultMaxBackoff`) | Cap on the wait. |

Only dial errors, network timeouts and first-byte timeouts (`httpretry.ErrFirstByteTimeout`) are retried. HTTP error responses are not.

Options:

| Option | Default | Effect |
| --- | --- | --- |
| `WithPath(path)` | `/chat/completions` | Endpoint path appended to `BaseURL`. |
| `WithQuery(key, value)` | none | Add a query parameter. |
| `WithURLFunc(fn func(llm.Settings) string)` | none | Compute the full URL per request. |
| `WithAuthHeader(header, prefix)` | `Authorization`, `Bearer ` | Auth header name and value prefix. |
| `WithoutStreamOptions()` | sends `stream_options.include_usage` | Omit `stream_options`. |
| `WithoutDeveloperRole()` | sends `developer` | Send developer messages as `user`. |
| `WithoutTools()` | sends tools | Never send `tools` or `tool_choice`. |
| `WithReasoningContent()` | off | Send thoughts back as `reasoning_content` on assistant messages. |
| `WithSkipAnsweredToolCalls()` | off | Drop streamed tool calls whose ID already has a result in the context. |
| `WithMaxTokensField(field)` | `max_tokens` | Body field for `MaxTokens`; empty omits it. |
| `WithSeedField(field)` | `seed` | Body field for `Seed`; empty omits it. |
| `WithMessageTransform(fn)` | none | Rewrite the wire messages before sending. |
| `WithBodyHook(fn func(body map[string]any, settings llm.Settings))` | none | Edit the request body; runs before `Extra` is applied. |
| `WithUsageHook(fn func(*llm.Usage))` | none | Adjust reported usage. |
| `WithRetry(policy)` | `Config.Retry` | Override the retry policy. |
| `WithThoughtTags(open, close)` | none | Treat text between the tags as thoughts. |
| `WithInlineThoughts()` | none | `WithThoughtTags("<think>", "</think>")`. |

Helpers: `ReadAPIError`, `ParseStreamError`, `RequestID` (reads `x-request-id`), `NormalizeStopReason`, `ResponseFormatBody`, `DataURL`, `DefaultHTTPClient`, `NewSSEReader`, `(*Provider).BuildRequestBody`. Errors are `*openaicompat.APIError`, or a `*llm.RefusalError` wrapping one when the vendor code is `content_filter`. See [errors](errors.md#typed-errors).

## Next steps

- [Function calling guide](../guides/function-calling.md): tools end to end.
- [LLM providers](../providers/llm.md): every built-in provider and its config.
- [Aggregators reference](aggregators.md): how the context is built during a call.
- [Errors reference](errors.md): refusals and API errors.
