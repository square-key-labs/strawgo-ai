# LLM providers

This page shows how to configure every LLM (large language model) integration in Strawgo: the config fields and their defaults, a constructor example, the quirks the code handles for you, and how to change the model while a call is running.

> [!NOTE]
> No LLM integration has been verified against the live vendor API. See the status column in the [provider matrix](README.md#llm).

An LLM service is a **processor**: a pipeline stage that receives **frames** (small typed messages) and pushes new ones. It reads the conversation **context** (the list of messages so far), streams the model's reply as `LLMTextFrame`s, and runs any functions the model calls.

## Choosing a provider

Every row below is a fact about the integration code, not about the model behind it.

| If you need | Use | Why |
| --- | --- | --- |
| The widest choice of hosted models through one API shape | any `openaicompat`-based package (OpenAI, Groq, Cerebras, Together, Fireworks, OpenRouter, DeepSeek, Mistral, Qwen, SambaNova, NVIDIA, xAI, Sarvam, Ollama) | one shared client, same config struct, same tool handling |
| A server you host yourself | `ollama`, or `openaicompat.NewService` with your own `BaseURL` | no API key is sent when `APIKey` is empty |
| Extended thinking and prompt caching on Claude | `anthropic` | `Thinking`, `EnablePromptCaching` and `Betas` fields |
| AWS IAM credentials instead of an API key | `aws/bedrock` | uses the AWS SDK credential chain |
| Google service-account credentials | `vertex` | uses Application Default Credentials or a JSON key |
| An Azure OpenAI deployment | `azureopenai` | builds the deployment URL and `api-key` header |
| The OpenAI Responses API | `openai.NewResponsesLLMService` | separate provider, same config struct |
| No function calling | `perplexity` | the integration never sends tools |

## Complete example

This program asks OpenAI one question and prints the streamed answer. It uses the same pipeline shape as a voice bot, minus the audio: a user context aggregator, the LLM service, a small printer, and an assistant context aggregator.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
)

type printer struct {
	*processors.BaseProcessor
	task *pipeline.PipelineTask
}

func newPrinter() *printer {
	p := &printer{}
	p.BaseProcessor = processors.NewBaseProcessor("Printer", p)
	return p
}

func (p *printer) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.LLMTextFrame:
		fmt.Print(f.Text)
	case *frames.LLMFullResponseEndFrame:
		fmt.Println()
		_ = p.task.StopWhenDone()
	case *frames.ErrorFrame:
		fmt.Fprintln(os.Stderr, "error:", f)
		_ = p.task.StopWhenDone()
	}
	return p.PushFrame(frame, direction)
}

func main() {
	logger.SetLevel(logger.WARN)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required")
	}
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       apiKey,
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You answer in one short sentence.",
	})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	out := newPrinter()
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{
		pair.User(), llmSvc, out, pair.Assistant(),
	}))
	out.task = task

	question := []llm.Message{llm.User("What is a voice agent?")}
	if err := task.QueueFrame(frames.NewLLMMessagesAppendFrame(question, true)); err != nil {
		log.Fatal(err)
	}
	if err := task.Run(context.Background()); err != nil {
		log.Fatal(err)
	}
}
```

Run it:

```sh
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
go run .
```

How it works:

1. `openai.NewLLMService` returns an `*llmservice.Service`, which is a processor, so it goes straight into `pipeline.NewPipeline`.
2. `NewLLMMessagesAppendFrame(question, true)` adds the user message to the context and asks for a reply. The `true` means "run the LLM now".
3. The service streams `LLMTextFrame`s downstream. The printer writes them and stops the task at `LLMFullResponseEndFrame`.
4. In a voice bot you replace the printer with a TTS service and a transport output, as in the [quickstart](../getting-started/quickstart.md).

To use another provider, change only the constructor line. The rest of the program stays the same.

## Shared config for OpenAI-compatible providers

These packages all define `type LLMConfig = openaicompat.ServiceConfig`: `openai`, `groq`, `cerebras`, `deepseek`, `fireworks`, `mistral`, `nvidia`, `ollama`, `openrouter`, `perplexity`, `qwen`, `sambanova`, `together`, `xai` and `sarvam`. `azureopenai.LLMConfig` embeds it.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent as `Authorization: Bearer <key>`. When empty, no auth header is sent. |
| `BaseURL` | `string` | the package's `DefaultBaseURL` | API root; the client appends `/chat/completions`. |
| `Model` | `string` | the package's `DefaultModel` | Model name sent in each request. |
| `SystemPrompt` | `string` | none | Sent as the first `system` message of every request. |
| `Settings` | `llm.Settings` | zero | Initial temperature, top-p, max tokens, seed, response format and `Extra` body fields. `Model` and `SystemPrompt` above win over the same fields here. |
| `Headers` | `map[string]string` | none | Extra HTTP headers on every request. |
| `HTTPClient` | `*http.Client` | a shared client with connect and header timeouts | Replace to add a proxy or tracing. |
| `Retry` | `httpretry.Policy` | zero (no retry) | Retries connection errors and slow first bytes. Enabled when `MaxAttempts` is greater than 1; `Backoff` defaults to 200 ms and `MaxBackoff` to 2 s. |
| `ServiceOptions` | `[]llmservice.Option` | none | Options for the wrapping service: `WithName`, `WithSequentialFunctionCalls`, `WithFunctionCallTimeout`. |

`Settings.Extra` entries are copied into the JSON request body as top-level fields. An `Extra` key named `extra_body` whose value is a `map[string]any` is merged field by field.

The library never reads environment variables for keys. The `*_API_KEY` names on this page are the convention the examples use; you pass the key in the config yourself.

## OpenAI

Chat Completions against `https://api.openai.com/v1`. Default model `gpt-4.1`. Conventional key variable: `OPENAI_API_KEY`.

```go
func newOpenAI() *llmservice.Service {
	return openai.NewLLMService(openai.LLMConfig{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a helpful phone assistant.",
	})
}
```

Quirks the code handles:

- `MaxTokens` is sent as `max_completion_tokens`, which newer OpenAI models require.
- Streaming requests ask for usage with `stream_options.include_usage`, so usage metrics arrive at the end of each response.

### OpenAI Responses API

`openai.NewResponsesLLMService(cfg)` takes the same `LLMConfig` and calls `POST {BaseURL}/responses` instead. It sends `store: false`, streams reasoning summaries as thought frames, and copies `Settings.Extra` into the request body.

```go
func newOpenAIResponses() *llmservice.Service {
	return openai.NewResponsesLLMService(openai.LLMConfig{
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model:  "gpt-4.1-mini",
	})
}
```

## Other OpenAI-compatible providers

Each package below has `NewLLMService(cfg LLMConfig) *llmservice.Service`, `NewProvider(cfg)` for the bare provider, and the constants `DefaultBaseURL` and `DefaultModel`. Only the defaults and the wire quirks differ.

| Package | Default base URL | Default model | Conventional key variable | Quirks the code handles |
| --- | --- | --- | --- | --- |
| `groq` | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` | `GROQ_API_KEY` | none |
| `cerebras` | `https://api.cerebras.ai/v1` | `gpt-oss-120b` | `CEREBRAS_API_KEY` | none |
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-flash` | `DEEPSEEK_API_KEY` | sends `"thinking": {"type": "disabled"}` unless you override `Extra["thinking"]`; `developer` messages become `user`; no `seed`; earlier reasoning is sent back as `reasoning_content` |
| `fireworks` | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/nemotron-3-ultra-nvfp4` | `FIREWORKS_API_KEY` | no `stream_options`; no `seed` |
| `mistral` | `https://api.mistral.ai/v1` | `mistral-small-latest` | `MISTRAL_API_KEY` | `developer` becomes `user`; no `stream_options`; seed sent as `random_seed`; a system message after the first non-system message becomes `user`; an assistant message is inserted after a trailing tool result; tool calls the context already answered are dropped |
| `nvidia` | `https://integrate.api.nvidia.com/v1` | `nvidia/nemotron-3.5-super-vl-120b-a12b` | `NVIDIA_API_KEY` | text inside `<think>…</think>` is emitted as thought frames, not spoken text |
| `ollama` | `http://localhost:11434/v1` | `llama3.2` | none needed | `developer` becomes `user` |
| `openrouter` | `https://openrouter.ai/api/v1` | `openai/gpt-4.1` | `OPENROUTER_API_KEY` | `developer` becomes `user`; for model names containing `gemini`, every system message after the first becomes `user` |
| `perplexity` | `https://api.perplexity.ai` | `sonar` | `PERPLEXITY_API_KEY` | `developer` becomes `user`; no `stream_options`; tools are never sent; no `seed` |
| `qwen` | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | `DASHSCOPE_API_KEY` | `developer` becomes `user` |
| `sambanova` | `https://api.sambanova.ai/v1` | `Meta-Llama-3.3-70B-Instruct` | `SAMBANOVA_API_KEY` | `developer` becomes `user` |
| `sarvam` | `https://api.sarvam.ai/v1` | `sarvam-m` | `SARVAM_API_KEY` | `developer` becomes `user`; constructors are `NewLLMService` and `NewLLMProvider` |
| `together` | `https://api.together.xyz/v1` | `zai-org/GLM-5.2` | `TOGETHER_API_KEY` | `developer` becomes `user` |
| `xai` | `https://api.x.ai/v1` | `grok-4.20-non-reasoning` | `XAI_API_KEY` | when the reported total counts reasoning tokens separately, they are added to completion tokens |

All of them stream `reasoning_content` or `reasoning` deltas as thought frames when the server sends them.

The constructor looks the same for every package. For example:

```go
func newGroq() *llmservice.Service {
	return groq.NewLLMService(groq.LLMConfig{APIKey: os.Getenv("GROQ_API_KEY")})
}

func newTogether() *llmservice.Service {
	return together.NewLLMService(together.LLMConfig{APIKey: os.Getenv("TOGETHER_API_KEY")})
}

func newLocalOllama() *llmservice.Service {
	return ollama.NewLLMService(ollama.LLMConfig{Model: "llama3.2"})
}

func newSarvam() *llmservice.Service {
	return sarvam.NewLLMService(sarvam.LLMConfig{APIKey: os.Getenv("SARVAM_API_KEY")})
}
```

### Any other OpenAI-compatible server

`openaicompat.NewService(cfg, defaults, opts...)` works with any server that speaks Chat Completions. The `openaicompat.Option` values let you match a server's quirks: `WithoutDeveloperRole`, `WithoutStreamOptions`, `WithoutTools`, `WithSeedField`, `WithMaxTokensField`, `WithAuthHeader`, `WithPath`, `WithQuery`, `WithURLFunc`, `WithMessageTransform`, `WithBodyHook`, `WithUsageHook`, `WithReasoningContent`, `WithInlineThoughts`, `WithThoughtTags`, `WithSkipAnsweredToolCalls` and `WithRetry`.

```go
func newVLLM() *llmservice.Service {
	return openaicompat.NewService(
		openaicompat.ServiceConfig{BaseURL: "http://localhost:8000/v1"},
		openaicompat.Defaults{Name: "VLLM", Model: "meta-llama/Llama-3.1-8B-Instruct"},
		openaicompat.WithoutDeveloperRole(),
	)
}
```

## Azure OpenAI

`azureopenai.NewLLMService(cfg)` talks to an Azure OpenAI resource. Conventional key variable: `AZURE_OPENAI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Endpoint` | `string` | none | Resource URL, for example `https://YOUR_RESOURCE.openai.azure.com`. Used as the base URL when `BaseURL` is empty. |
| `APIVersion` | `string` | `2025-04-01-preview` | `api-version` query value for deployment URLs. |
| embedded `ServiceConfig` | | | as in the shared table; `Model` is the deployment name and defaults to `gpt-4.1` |

The code picks one of two URL shapes:

- If `Endpoint` ends in `/openai/v1`, requests go to `{Endpoint}/chat/completions` with a bearer token.
- Otherwise requests go to `{Endpoint}/openai/deployments/{Model}/chat/completions?api-version={APIVersion}` with the key in the `api-key` header.

`MaxTokens` is sent as `max_completion_tokens`.

```go
func newAzure() *llmservice.Service {
	return azureopenai.NewLLMService(azureopenai.LLMConfig{
		ServiceConfig: openaicompat.ServiceConfig{
			APIKey: os.Getenv("AZURE_OPENAI_API_KEY"),
			Model:  "my-gpt-deployment",
		},
		Endpoint: os.Getenv("AZURE_OPENAI_ENDPOINT"),
	})
}
```

## Anthropic

`anthropic.NewLLMService(cfg, opts...)` streams from the Messages API. Conventional key variable: `ANTHROPIC_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent in the `x-api-key` header. |
| `Model` | `string` | `claude-sonnet-4-6` | Model ID. |
| `SystemPrompt` | `string` | none | Sent as the `system` block. |
| `Temperature` | `float64` | 0 (not sent) | Sent only when greater than 0. |
| `MaxTokens` | `int` | 4096 | `max_tokens`; required by the API. |
| `BaseURL` | `string` | `https://api.anthropic.com/v1` | API root; the client appends `/messages`. |
| `EnablePromptCaching` | `bool` | false | Adds `cache_control` markers to the system prompt, the last tool and recent messages. |
| `PromptCacheTTL` | `string` | none | `ttl` value inside the system prompt's cache marker. |
| `Thinking` | `*anthropic.Thinking` | nil | `Type`, `BudgetTokens` and `Display`, sent as the `thinking` object. |
| `Effort` | `string` | none | Sent as `output_config.effort`. |
| `Betas` | `[]string` | none | Joined into the `anthropic-beta` header. |
| `HTTPClient` | `*http.Client` | a client with connect and header timeouts | |
| `Retry` | `httpretry.Policy` | zero (no retry) | as in the shared table |

Quirks the code handles: the `anthropic-version: 2023-06-01` header; tool results moved into the shape the API expects; JSON-schema response formats sent as `output_config.format` (a format without a schema returns an error); `Settings.Extra["enable_prompt_caching"]` turns caching on or off per request, and every other `Extra` key is copied into the body.

```go
func newAnthropic() *llmservice.Service {
	return anthropic.NewLLMService(anthropic.LLMConfig{
		APIKey:       os.Getenv("ANTHROPIC_API_KEY"),
		SystemPrompt: "You are a helpful phone assistant.",
		Thinking:     &anthropic.Thinking{Type: "enabled", BudgetTokens: 1024},
		MaxTokens:    2048,
	})
}
```

## Google Gemini

`gemini.NewLLMService(cfg, opts...)` streams from `generateContent` on the Gemini API. Conventional key variable: `GEMINI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent in the `x-goog-api-key` header. |
| `Model` | `string` | none | Required. There is no default model for this constructor. |
| `SystemPrompt` | `string` | none | Sent as the system instruction. |
| `Temperature` | `float64` | 0 (not sent) | Sent only when greater than 0. |
| `MaxTokens` | `int` | 0 (not sent) | Sent only when greater than 0. |
| `BaseURL` | `string` | `https://generativelanguage.googleapis.com/v1beta` | API root. |
| `Thinking` | `*gemini.Thinking` | nil | `Budget` (`thinkingBudget`), `Level` (`thinkingLevel`), `IncludeThoughts`. |
| `SafetySettings` | `[]map[string]any` | none | Sent as `safetySettings`. |
| `HTTPClient` | `*http.Client` | a client with connect and header timeouts | |
| `Retry` | `httpretry.Policy` | zero (no retry) | as in the shared table |

`Settings.Extra` is ignored by this provider.

```go
func newGemini() *llmservice.Service {
	return gemini.NewLLMService(gemini.LLMConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
		Model:  "gemini-2.5-flash",
	})
}
```

## Google Vertex AI

`vertex.NewLLMService(cfg, opts...)` calls Gemini models through Vertex AI. It returns an error because it loads credentials at construction.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ProjectID` | `string` | none | Required. |
| `Location` | `string` | none | Required. `global` uses `https://aiplatform.googleapis.com`; any other value uses `https://{Location}-aiplatform.googleapis.com`. |
| `Model` | `string` | `gemini-2.5-flash` | |
| `SystemPrompt`, `Temperature`, `MaxTokens`, `Thinking`, `SafetySettings`, `HTTPClient`, `Retry` | | | as for Gemini |
| `CredentialsJSON` | `[]byte` | none | Service-account JSON. When empty, Google Application Default Credentials are used (for example `GOOGLE_APPLICATION_CREDENTIALS`). |
| `BaseURL` | `string` | derived from `Location` | |

```go
func newVertex() (*llmservice.Service, error) {
	return vertex.NewLLMService(vertex.LLMConfig{
		ProjectID: os.Getenv("GOOGLE_CLOUD_PROJECT"),
		Location:  "us-central1",
	})
}
```

## AWS Bedrock

`bedrock.NewLLMService(ctx, cfg, opts...)` uses the Bedrock `ConverseStream` API through the AWS SDK. It returns an error because it loads the AWS config. `NewLLMServiceWithClient` takes a `*bedrockruntime.Client` you built yourself.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AWS` | `awsauth.Config` | zero | `Region` (default `us-east-1`), `AccessKeyID`, `SecretAccessKey`, `SessionToken`, `Profile`, `Endpoint`, `HTTPClient`, or a prepared `AWSConfig`. Empty fields fall back to the AWS SDK default credential chain. |
| `Model` | `string` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Model or inference profile ID. |
| `SystemPrompt` | `string` | none | |
| `Temperature`, `TopP` | `*float64` | nil (not sent) | |
| `MaxTokens` | `int` | 4096 | |
| `StopSequences` | `[]string` | none | Overridable per request with `Extra["stop_sequences"]`. |
| `Latency` | `string` | none | Performance config latency; overridable with `Extra["latency"]`. |
| `EnablePromptCaching` | `bool` | false | Overridable with `Extra["enable_prompt_caching"]`. |
| `AdditionalModelRequestFields` | `map[string]any` | none | Model-specific fields; merged with `Extra["additional_model_request_fields"]`. |

```go
func newBedrock(ctx context.Context) (*llmservice.Service, error) {
	return bedrock.NewLLMService(ctx, bedrock.LLMConfig{
		AWS: awsauth.Config{Region: "us-west-2"},
	})
}
```

## Change settings during a call

Queue an `LLMUpdateSettingsFrame` on the task. The service merges the delta into its current `llm.Settings`: non-empty strings and non-nil pointers replace the old value, and `Extra` keys are merged. The next request uses the new settings.

```go
func switchToFastModel(task *pipeline.PipelineTask) error {
	temperature := 0.2
	return task.QueueFrame(frames.NewLLMUpdateSettingsFrame(llm.Settings{
		Model:             "gpt-4.1-nano",
		SystemInstruction: "Answer in at most ten words.",
		Temperature:       &temperature,
	}))
}
```

Things to know:

- The frame is uninterruptible, so a barge-in does not drop it. **Barge-in** (interruption) is when the caller speaks while the bot is talking.
- Set the frame's `Service` field to a specific service to target only that one. Without it, the first LLM service that receives the frame applies it and does not pass it on.
- Fields fixed at construction, such as Anthropic `Thinking` or Gemini `SafetySettings`, cannot be changed this way.
- You can also call `UpdateSettings(delta)` on the `*llmservice.Service` directly.

## Next steps

- [Function calling](../guides/function-calling.md): give the model tools.
- [Switching services](../guides/switching-services.md): fail over between two LLMs.
- [LLM reference](../reference/llm.md): every type in the `llm` and `llmservice` packages.
- [Writing a custom provider](custom.md): add an LLM that is not listed.
