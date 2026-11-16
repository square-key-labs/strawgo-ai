# Errors reference

This page explains how errors travel through a Strawgo pipeline: the `ErrorFrame`, the error category that says what kind of failure it was, the difference between fatal and non-fatal errors, what the task does with each, the typed errors you can test for, and how to observe and handle them.

A **frame** is one typed message in the pipeline. A **processor** is one stage in the pipeline. The **task** runs the pipeline for one call. **Upstream** means back toward the start of the pipeline, where the task listens.

## Quick example

This function logs every pipeline error with a reason you can act on, and tells the caller about refusals.

```go
func watchErrors(task *pipeline.PipelineTask) {
	task.OnError(func(err error) {
		switch {
		case errors.Is(err, llm.ErrResponseFormatUnsupported):
			log.Printf("config error, the model cannot do structured output: %v", err)
		case frames.ClassifyError(err) == frames.ErrorCategoryRateLimit:
			log.Printf("rate limited: %v", err)
		case frames.ClassifyError(err).IsPermanent():
			log.Printf("provider rejected the credentials, account or configuration: %v", err)
		case errors.Is(err, httpretry.ErrFirstByteTimeout):
			log.Printf("LLM did not answer in time: %v", err)
		default:
			if r, ok := llm.AsRefusal(err); ok {
				log.Printf("%s refused (%s): %s", r.Provider, r.Category, r.Reason)
				_ = task.QueueFrame(frames.NewTTSSpeakFrame("Sorry, I can't help with that."))
				return
			}
			log.Printf("pipeline error: %v", err)
		}
	})
}
```

`OnError` runs on a pipeline goroutine. Keep it short; `QueueFrame` is safe to call from it.

## ErrorFrame

```text
type ErrorFrame struct {
	*SystemFrame
	Error     error
	Fatal     bool
	Processor string
	Category  ErrorCategory
}
func NewErrorFrame(err error) *ErrorFrame
func NewCategorizedErrorFrame(err error, category ErrorCategory) *ErrorFrame
```

| Field | Meaning |
| --- | --- |
| `Error` | The error. When built by `PushError`, it is `fmt.Errorf("<message>: %w", err)`, so `errors.Is` and `errors.As` see the original. |
| `Fatal` | Whether the call cannot continue. |
| `Processor` | Name of the processor that reported it. |
| `Category` | What kind of failure it was (see [error categories](#error-categories)). Empty when the reporter did not set one; `NewErrorFrame` and `PushError` leave it empty. |

`PushError` also sets metadata: `"processor"`, `"file"` and `"line"` (where `PushError` was called) and, for fatal errors, `"fatal": true`.

`ErrorFrame` is a system frame: it overtakes queued audio and text and is never dropped by an interruption. Processors push it upstream, and every built-in processor forwards it, so it reaches the task's source.

## Reporting an error from a processor

```text
func (p *BaseProcessor) PushError(errorMsg string, err error, fatal bool) error
func (p *BaseProcessor) PushErrorFrame(errorFrame *frames.ErrorFrame) error
func (p *BaseProcessor) SetOnError(handler ErrorHandler)
```

| Call | Effect |
| --- | --- |
| `PushError(msg, err, fatal)` | Log at error level, call the processor's `SetOnError` handler, push an `ErrorFrame` upstream. `err` may be nil. |
| `PushErrorFrame(f)` | Same for a frame you built; fills `Processor` if empty. Logs as non-fatal. |
| Returning an error from `HandleFrame` | Logged and passed to the `SetOnError` handler, but **not** pushed as an `ErrorFrame`. `context.Canceled` is ignored. Call `PushError` if the task should know. |
| A panic in `HandleFrame` | Recovered and pushed as a non-fatal `ErrorFrame` ("processor panic"). |

```go
func reportLookupFailure(p *processors.BaseProcessor, err error) {
	_ = p.PushError("CRM lookup failed", err, false)
}
```

## Error categories

A category says why an operation failed, independent of the provider: a rejected API key versus a rate limit versus a network drop. It does not say whether the call should go on; the task and the switcher decide that from it.

```text
type ErrorCategory string
func (c ErrorCategory) IsPermanent() bool
func ClassifyHTTPStatus(code int) ErrorCategory
func ClassifyError(err error) ErrorCategory
```

| Category | Value | Meaning | Permanent |
| --- | --- | --- | --- |
| `ErrorCategoryUnknown` | `unknown` | The cause could not be determined. | No |
| `ErrorCategoryAuthentication` | `authentication` | Credentials are missing or invalid (HTTP 401). | Yes |
| `ErrorCategoryAuthorization` | `authorization` | Credentials are valid but lack access (HTTP 403). | Yes |
| `ErrorCategoryInvalidRequest` | `invalid_request` | This request was rejected (HTTP 400, 404, 422): a malformed tool call, a context that is too long, an image that could not be fetched. The next request may well succeed. | No |
| `ErrorCategoryConfiguration` | `configuration` | The service is set up with something the provider does not have, such as an unknown model or deployment. Only set from a provider error code known to mean this (see below), never from a bare status code. | Yes |
| `ErrorCategoryRateLimit` | `rate_limit` | Too many requests in too short a window (HTTP 429). | No |
| `ErrorCategoryQuota` | `quota` | The account's credit or usage allowance is used up (HTTP 402, or OpenAI's `insufficient_quota`). | Yes |
| `ErrorCategoryConnectivity` | `connectivity` | The service could not be reached: a dial failure, DNS failure, timeout or `context.DeadlineExceeded`. | No |
| `ErrorCategoryServer` | `server` | The provider reported an internal failure (HTTP 5xx). | No |
| `ErrorCategoryApplication` | `application` | Your own code failed, not the provider. | No |
| `ErrorCategoryContentFiltered` | `content_filtered` | The provider's content policy rejected the request, or the model refused. | No |

A **permanent** category means retrying will give the same result until something changes (new credentials, more credit, a different model). A service that reports one is called **unusable**: the failover strategy will not switch back to it, and the task applies its `ProcessorUnusablePolicy`.

`ClassifyHTTPStatus` maps a status code as in the table; any other code is `unknown`. `ClassifyError` walks the error chain with `errors.As` and returns the first match of:

1. an error with an `ErrorCategory() frames.ErrorCategory` method, which decides for itself;
2. an error with an `HTTPStatusCode() int` method, classified by `ClassifyHTTPStatus`;
3. `context.DeadlineExceeded` or any `net.Error` (including `*net.OpError` and `*net.DNSError`), which is `connectivity`.

`context.Canceled`, nil and anything else are `unknown`.

Where categories are set:

| Source | Category |
| --- | --- |
| `llmservice.Service` "generation failed" | `ClassifyError` of the provider's error. |
| `llmservice.Service` "response refused" | `content_filtered`. |
| STT and TTS bases (connect, request, stream, transcription, settings update and `Sink.Error` errors) | `ClassifyError` of the provider's error. |
| `*openaicompat.APIError` | By `Code` first: `content_filter` is `content_filtered`, `invalid_api_key` is `authentication`, `insufficient_quota` (also as `Type`) is `quota`, `model_not_found` and Azure's `DeploymentNotFound` are `configuration`. Otherwise its status code. |
| `*anthropic.APIError`, `*gemini.APIError` | Their `Status`, so an unknown model (HTTP 404) is `invalid_request`, not `configuration`. |
| AWS SDK errors (Bedrock, Polly, Transcribe) | The SDK's `HTTPStatusCode()`. |
| `*reconnect.HandshakeError` (a websocket handshake answered with an HTTP status) | Its `StatusCode`. |
| `httpretry.ErrFirstByteTimeout` | `connectivity`. |
| Errors that only put the status in the message text (`"<vendor> HTTP 401: ..."`, used by most HTTP TTS and segmented STT providers) | `unknown`, unless the underlying error is a network error. |
| "transcription backlog full", "processor panic", anything sent with `PushError` | Not set. |

To classify your own provider's errors, return an error that has an `HTTPStatusCode() int` or `ErrorCategory() frames.ErrorCategory` method. A processor of your own can set the category directly:

```go
func reportRejectedKey(p *processors.BaseProcessor, err error) {
	_ = p.PushErrorFrame(frames.NewCategorizedErrorFrame(fmt.Errorf("CRM login failed: %w", err), frames.ErrorCategoryAuthentication))
}
```

## Fatal versus non-fatal

| | Non-fatal | Fatal |
| --- | --- | --- |
| Meaning | This request failed; the call can continue. | The call cannot continue (for example a connection is gone for good). |
| Task action | `OnError` callback. If the category is permanent, `PipelineTaskConfig.ProcessorUnusablePolicy` applies, once per processor: keep running (the default), end the pipeline gracefully, or cancel it. | `OnError` callback, then `Cancel()`, unless `PipelineTaskConfig.KeepRunningOnFatal` is true. |
| `ServiceSwitcher` | From the active service: passed to the strategy, which may switch services (`FailoverSwitchStrategy` does) and drop the error. From an inactive service: dropped. A permanent category marks the reporting service unusable. | Always passed upstream. |
| `MuteUntilFirstBotCompleteUserMuteStrategy` | Any `ErrorFrame` before the bot has spoken unmutes the user. | Same. |

Built-in sources:

| Source | Errors | Fatal? |
| --- | --- | --- |
| `llmservice.Service` | "generation failed" (stream error), "response refused" (refusal) | No |
| STT base | "STT connect failed", "settings update failed", "transcription failed", "transcription backlog full"; provider errors through `Sink.Error` | No, except providers built on `reconnect.Session` once reconnection gives up |
| TTS base | "TTS connect failed", "TTS request failed", "TTS stream failed", "settings update failed"; provider errors through `Sink.Error` | No, except providers built on `reconnect.Session` once reconnection gives up |
| `reconnect.Session` | "connection failed after N attempts", "connection dropped N times in a row" | Yes; each earlier failed attempt is reported as non-fatal. A handshake rejected with an HTTP status is a `*reconnect.HandshakeError`, so its category comes from the status. |
| OpenAI Realtime, Gemini Live | "connection lost" | Yes |
| WebRTC output | "webrtc output encoder" | Yes |
| WebSocket output, `AudioConverterProcessor` | "output audio conversion failed", "audio conversion failed" | No |
| Any processor | "processor panic" | No |

When a fatal error cancels the call, or `ProcessorUnusableCancel` cancels it, `PipelineTask.Run` returns a `*pipeline.FatalError` that wraps the original `ErrorFrame.Error`; test it with `errors.Is` or `errors.As`. With `KeepRunningOnFatal`, `Run` does not report it. See [pipeline](pipeline.md#lifecycle-and-shutdown).

## Typed errors

Test for these with `errors.Is` and `errors.As` on the error from `OnError` or `ErrorFrame.Error`.

| Error | Package | Fields or meaning |
| --- | --- | --- |
| `*llm.RefusalError` | `llm` | `Refusal Refusal` (`Category`, `Provider`, `Reason`, `Detail`, `Message`), `Err error` (the underlying error, if any). Find it with `llm.AsRefusal(err)`. |
| `llm.ErrResponseFormatUnsupported` | `llm` | The provider cannot honour `Settings.ResponseFormat` (Anthropic and Bedrock return it for `json_object`, or for `json_schema` with an empty schema). |
| `*openaicompat.APIError` | `services/openaicompat` | `Provider`, `StatusCode`, `Type`, `Code`, `Param`, `Message`, `Body`, `RequestID`. Every OpenAI-compatible provider returns it for HTTP errors. When `Code` is `content_filter`, it is wrapped in a `*llm.RefusalError`. `ErrorCategory()` classifies it. |
| `*anthropic.APIError` | `services/anthropic` | `Status`, `Type`, `Message`, `RequestID`. `HTTPStatusCode()` returns `Status`. |
| `*gemini.APIError` | `services/gemini` | `Status`, `Code`, `Message`. `HTTPStatusCode()` returns `Status`. |
| `*reconnect.HandshakeError` | `services/reconnect` | `StatusCode`, `Err`: the websocket server answered the handshake with an HTTP error. |
| `httpretry.ErrFirstByteTimeout` | `services/openaicompat/httpretry` | No response byte within `RetryPolicy.FirstByteTimeout`. After retries the message ends with `(after N attempts)`; `errors.Is` still matches. |
| `*deepgram.FluxError` | `services/deepgram` | `Code`, `Description` from Deepgram Flux. |
| `reconnect.ErrQueueFull` | `services/reconnect` | `SendJSON` could not queue a message. |
| `*flows.ValidationError` | `flows` | `Problems []Problem` from building a JSON flow. |
| `flows.ErrNotInitialized`, `ErrAlreadyInitialized`, `ErrClosed` | `flows` | Manager misuse or shutdown. |
| `webrtc.ErrBadRequest`, `ErrNotFound`, `ErrShuttingDown`, `ErrNoSupportedCodec`, `ErrSessionClosed` | `transports/webrtc` | Signalling errors; mapped to HTTP status codes by the handler. |
| `*classify.PanicError` | `audio/classify` | `Value`: a recovered classifier panic. |
| `wav.ErrFormat` | `audio/wav` | The WAV file is not 16-bit PCM. |

### Refusals

A **refusal** is the model or the vendor's safety system declining to answer. The LLM service turns it into a non-fatal `ErrorFrame` whose error is (or wraps) a `*llm.RefusalError`. The bot says nothing for that turn, so decide what it should say: queue a `TTSSpeakFrame` as in the quick example, or append a message and run the LLM again. Categories and stop reasons are in the [LLM reference](llm.md#refusals); the [structured output and refusals guide](../guides/structured-output-and-refusals.md) shows patterns.

## Observing errors

| Where | How | Sees |
| --- | --- | --- |
| Task | `task.OnError(func(error))` | Every `ErrorFrame` that reaches either end of the pipeline, fatal or not. One callback; a second call replaces it. |
| Task | `task.OnFrameReachedUpstream(fn, (*frames.ErrorFrame)(nil))` | The full `ErrorFrame`, with `Fatal`, `Processor`, `Category` and metadata. Adds to other handlers. |
| Processor | `p.SetOnError(func(processor, err, file, line))` | That processor's `PushError` calls and returned errors, including ones never pushed as frames. |
| Observer | An `Observer` interested in `*frames.ErrorFrame` push events | Every hop of every `ErrorFrame`, asynchronously. See [observers](observers.md). |
| Logs | The `logger` package | Every `PushError` at error level, with file and line. |

## Recommended handling

1. **Log every error with its processor and category.** Subscribe with `OnFrameReachedUpstream` for `ErrorFrame` when you need `Processor`, `Fatal` and `Category`.
2. **Let fatal errors end the call** (the default) unless you have a recovery path; a caller left in silence is worse than a clean hang-up. With `KeepRunningOnFatal`, you must recover yourself.
3. **Fail over instead of failing.** Wrap interchangeable services in a `ServiceSwitcher` with `FailoverSwitchStrategy` so a non-fatal error moves to the next vendor. Set `SwitchOnlyWhenUnusable` to ride out rate limits and network blips on the same vendor. See [switching services](../guides/switching-services.md).
4. **Decide what an unusable service means for the call.** A rejected API key will not fix itself. Set `ProcessorUnusablePolicy` to `ProcessorUnusableEnd` or `ProcessorUnusableCancel` if the call should stop rather than carry on without that service.
5. **Speak on refusals and LLM failures.** A failed generation produces no audio; queue a short apology so the caller knows to rephrase.
6. **Retry connections, not requests.** Set `RetryPolicy` (`MaxAttempts`, `FirstByteTimeout`) on OpenAI-compatible providers to retry dial errors and slow first bytes. HTTP error responses are not retried.
7. **Return errors from function handlers.** A handler error becomes the tool result, and the LLM explains it to the caller. It is not an `ErrorFrame`.
8. **Fix configuration errors at startup.** `ErrResponseFormatUnsupported` and `flows.ValidationError` mean the setup is wrong; they will not go away on retry.

```go
func watchUnusable(task *pipeline.PipelineTask, onUnusable func(processor string, category frames.ErrorCategory)) {
	task.OnFrameReachedUpstream(func(f frames.Frame) {
		if ef := f.(*frames.ErrorFrame); ef.Category.IsPermanent() {
			onUnusable(ef.Processor, ef.Category)
		}
	}, (*frames.ErrorFrame)(nil))
}
```

```go
func watchFatal(task *pipeline.PipelineTask, onFatal func(processor string, err error)) {
	task.OnFrameReachedUpstream(func(f frames.Frame) {
		if ef := f.(*frames.ErrorFrame); ef.Fatal {
			onFatal(ef.Processor, ef.Error)
		}
	}, (*frames.ErrorFrame)(nil))
}
```

## Next steps

- [Troubleshooting guide](../guides/troubleshooting.md): symptoms, causes and fixes.
- [Pipeline reference](pipeline.md): task callbacks and shutdown.
- [LLM reference](llm.md): refusals, stop reasons and `openaicompat` retry settings.
- [Switching services guide](../guides/switching-services.md): failover on errors.
