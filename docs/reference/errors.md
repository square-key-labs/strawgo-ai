# Errors reference

This page explains how errors travel through a Strawgo pipeline: the `ErrorFrame`, the difference between fatal and non-fatal errors, what the task does with each, the typed errors you can test for, and how to observe and handle them.

A **frame** is one typed message in the pipeline. A **processor** is one stage in the pipeline. The **task** runs the pipeline for one call. **Upstream** means back toward the start of the pipeline, where the task listens.

## Quick example

This function logs every pipeline error with a reason you can act on, and tells the caller about refusals.

```go
func watchErrors(task *pipeline.PipelineTask) {
	task.OnError(func(err error) {
		var apiErr *openaicompat.APIError
		switch {
		case errors.Is(err, llm.ErrResponseFormatUnsupported):
			log.Printf("config error, the model cannot do structured output: %v", err)
		case errors.As(err, &apiErr) && apiErr.StatusCode == http.StatusTooManyRequests:
			log.Printf("rate limited by %s (request %s)", apiErr.Provider, apiErr.RequestID)
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
}
func NewErrorFrame(err error) *ErrorFrame
```

| Field | Meaning |
| --- | --- |
| `Error` | The error. When built by `PushError`, it is `fmt.Errorf("<message>: %w", err)`, so `errors.Is` and `errors.As` see the original. |
| `Fatal` | Whether the call cannot continue. |
| `Processor` | Name of the processor that reported it. |

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

## Fatal versus non-fatal

| | Non-fatal | Fatal |
| --- | --- | --- |
| Meaning | This request failed; the call can continue. | The call cannot continue (for example a connection is gone for good). |
| Task action | `OnError` callback. | `OnError` callback, then `Cancel()`, unless `PipelineTaskConfig.KeepRunningOnFatal` is true. |
| `ServiceSwitcher` | From the active service: passed to the strategy, which may switch services (`FailoverSwitchStrategy` does) and drop the error. From an inactive service: dropped. | Always passed upstream. |
| `MuteUntilFirstBotCompleteUserMuteStrategy` | Any `ErrorFrame` before the bot has spoken unmutes the user. | Same. |

Built-in sources:

| Source | Errors | Fatal? |
| --- | --- | --- |
| `llmservice.Service` | "generation failed" (stream error), "response refused" (refusal) | No |
| STT base | "STT connect failed", "settings update failed", "transcription failed", "transcription backlog full"; provider errors through `Sink.Error` | No, except providers built on `reconnect.Session` once reconnection gives up |
| TTS base | "TTS connect failed", "TTS request failed", "TTS stream failed", "settings update failed"; provider errors through `Sink.Error` | No, except providers built on `reconnect.Session` once reconnection gives up |
| `reconnect.Session` | "connection failed after N attempts", "connection dropped N times in a row" | Yes; each earlier failed attempt is reported as non-fatal |
| OpenAI Realtime, Gemini Live | "connection lost" | Yes |
| WebRTC output | "webrtc output encoder" | Yes |
| WebSocket output, `AudioConverterProcessor` | "output audio conversion failed", "audio conversion failed" | No |
| Any processor | "processor panic" | No |

When a fatal error cancels the call, `PipelineTask.Run` returns an error that wraps the original `ErrorFrame.Error`; test it with `errors.Is` or `errors.As`. With `KeepRunningOnFatal`, `Run` does not report it. See [pipeline](pipeline.md#lifecycle-and-shutdown).

## Typed errors

Test for these with `errors.Is` and `errors.As` on the error from `OnError` or `ErrorFrame.Error`.

| Error | Package | Fields or meaning |
| --- | --- | --- |
| `*llm.RefusalError` | `llm` | `Refusal Refusal` (`Category`, `Provider`, `Reason`, `Detail`, `Message`), `Err error` (the underlying error, if any). Find it with `llm.AsRefusal(err)`. |
| `llm.ErrResponseFormatUnsupported` | `llm` | The provider cannot honour `Settings.ResponseFormat` (Anthropic and Bedrock return it for `json_object`, or for `json_schema` with an empty schema). |
| `*openaicompat.APIError` | `services/openaicompat` | `Provider`, `StatusCode`, `Type`, `Code`, `Param`, `Message`, `Body`, `RequestID`. Every OpenAI-compatible provider returns it for HTTP errors. When `Code` is `content_filter`, it is wrapped in a `*llm.RefusalError`. |
| `*anthropic.APIError` | `services/anthropic` | `Status`, `Type`, `Message`, `RequestID`. |
| `*gemini.APIError` | `services/gemini` | `Status`, `Code`, `Message`. |
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
| Task | `task.OnFrameReachedUpstream(fn, (*frames.ErrorFrame)(nil))` | The full `ErrorFrame`, with `Fatal`, `Processor` and metadata. Adds to other handlers. |
| Processor | `p.SetOnError(func(processor, err, file, line))` | That processor's `PushError` calls and returned errors, including ones never pushed as frames. |
| Observer | An `Observer` interested in `*frames.ErrorFrame` push events | Every hop of every `ErrorFrame`, asynchronously. See [observers](observers.md). |
| Logs | The `logger` package | Every `PushError` at error level, with file and line. |

## Recommended handling

1. **Log every error with its processor.** Subscribe with `OnFrameReachedUpstream` for `ErrorFrame` when you need `Processor` and `Fatal`.
2. **Let fatal errors end the call** (the default) unless you have a recovery path; a caller left in silence is worse than a clean hang-up. With `KeepRunningOnFatal`, you must recover yourself.
3. **Fail over instead of failing.** Wrap interchangeable services in a `ServiceSwitcher` with `FailoverSwitchStrategy` so a non-fatal error moves to the next vendor. See [switching services](../guides/switching-services.md).
4. **Speak on refusals and LLM failures.** A failed generation produces no audio; queue a short apology so the caller knows to rephrase.
5. **Retry connections, not requests.** Set `RetryPolicy` (`MaxAttempts`, `FirstByteTimeout`) on OpenAI-compatible providers to retry dial errors and slow first bytes. HTTP error responses are not retried.
6. **Return errors from function handlers.** A handler error becomes the tool result, and the LLM explains it to the caller. It is not an `ErrorFrame`.
7. **Fix configuration errors at startup.** `ErrResponseFormatUnsupported` and `flows.ValidationError` mean the setup is wrong; they will not go away on retry.

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
