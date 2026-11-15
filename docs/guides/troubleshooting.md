# Troubleshooting

This page lists common symptoms, each under its own heading with its likely causes and fixes, shows how to turn on debug logging, and lists what to include in a bug report.

## Before you begin

- Reproduce the problem with one call if you can. Most problems show up in the log of a single call.
- Terms used on this page: **STT** (speech-to-text), **LLM** (language model), **TTS** (text-to-speech), **VAD** (voice activity detection, which decides when the caller is speaking), **interruption** (the caller talking over the bot), **frame** (one unit of data in the pipeline), and **serializer** (the piece that translates a provider's messages into frames).

## Turn on debug logging first

Strawgo's logger reads `LOG_LEVEL` the first time it is used:

```bash
LOG_LEVEL=DEBUG LOG_COLOR=false go run .
```

Valid levels are `DEBUG`, `INFO` (the default), `WARN` and `ERROR`. `LOG_COLOR=false` (or `LOG_COLOR=0`) removes colour codes, which makes logs easier to search and share.

You can also change the level from code, for example behind an admin flag:

```go
func enableDebugLogs() {
	if err := logger.SetLogLevel("debug"); err != nil {
		log.Print(err)
	}
}
```

`logger.SetLevel(logger.DEBUG)` does the same with a typed level. Child loggers (each session, service and processor has a prefixed one) follow the change.

For frame-level detail, add a debug observer for the frames you care about (see [Observability](observability.md#built-in-observers)):

```go
func frameTrace() *pipeline.TaskObserver {
	obs := pipeline.NewTaskObserver()
	obs.AddObserver(observers.NewDebugLogObserverFor(
		frames.NewInterruptionFrame(),
		frames.NewBotStartedSpeakingFrame(),
		frames.NewBotStoppedSpeakingFrame(),
		frames.NewTranscriptionFrame("", true),
	))
	obs.AddObserver(observers.NewTranscriptionLogObserver())
	obs.AddObserver(observers.NewLLMLogObserver())
	return obs
}
```

Useful log lines to look for:

| Log line | Meaning |
| --- | --- |
| `[WebSocketServer] listening on [::]:8080/media` | The server is up, with its listen address and path. |
| `[WebSocketSession ws-N] connected (provider=... stream=... call=...)` | A call arrived and the handshake succeeded. |
| `[WebSocketServer] upgrade: ...` / `handshake: ...` / `no serializer for ...` | A connection was rejected before your handler ran. |
| `[WebSocketSession ws-N] handler: ...` | Your handler returned an error. |
| `client disconnected` | The caller or provider closed the connection. |
| `session timeout after ...` | `SessionTimeout` ended the call. |
| `hangup: ...` | The provider's hang-up request failed. |
| `Pipeline finishing: end frame` / `cancel frame` | Why the pipeline stopped. |
| `[WebRTCHandler] request failed: ...` | A WebRTC offer failed on the server side. |

## Symptoms

Each symptom below lists its likely causes, most common first, each with its fix.

### No audio heard from the bot

- **Cause:** the provider stream is one-way (for example Twilio `<Start><Stream>`), or bidirectional audio is not enabled (Jambonz `bidirectionalAudio`, Plivo `bidirectional="true"`). **Fix:** use a bidirectional stream such as Twilio `<Connect><Stream>`; see [Twilio](telephony-twilio.md#answer-inbound-calls-with-twiml) and [other providers](telephony-providers.md).
- **Cause:** the TTS service is failing. **Fix:** look for TTS errors in the log (wrong key, voice id or model) and test the TTS service alone.
- **Cause:** `s.Output()` is missing from the pipeline, or placed before the TTS. **Fix:** put `s.Output()` after the TTS service.
- **Cause:** Genesys has the stream paused. **Fix:** audio is dropped while paused; check the `paused` and `resumed` messages.
- **Cause:** WebRTC: the browser did not attach or play the remote track. **Fix:** set `srcObject` on an `<audio autoplay>` element in `ontrack`, after a user click.

### Choppy or robotic audio

- **Cause:** the machine is CPU-starved, so the real-time output pacing slips. **Fix:** check CPU on the host during calls and reduce calls per instance; see [Deployment: resource sizing](deployment.md#resource-sizing).
- **Cause:** network loss between your server and the provider or browser. **Fix:** host closer to the provider's media region; for WebRTC, use TURN over a better path.
- **Cause:** native passthrough with mismatched settings: the TTS produces µ-law at a rate other than the wire rate. **Fix:** set the TTS service to 8000 Hz when using `AudioOutNative`, or turn native output off. See [Audio](audio.md#native-passthrough).
- **Cause:** a mixer sound file is clipping. **Fix:** lower `mixer.WithVolume`.

### Bot talks over the user

- **Cause:** the user's turn is judged finished too early, during a pause. **Fix:** lengthen the stop strategy's speech timeout (`NewSpeechTimeoutUserTurnStopStrategy`, 600 ms by default). Keep VAD `StopSecs` at 0.2: the VAD analyzer logs a warning for any other value, and the speech timeout already waits after the pause. See [Turn-taking: stop strategies](turn-taking.md#stop-strategies).
- **Cause:** callers pause mid-thought ("my number is... 555..."). **Fix:** use Smart Turn with the stop strategy shown in [Turn-taking: Smart Turn](turn-taking.md#smart-turn); with the default stop strategy, Smart Turn does not hold the turn open.
- **Cause:** there is no VAD, so turn detection relies on STT timing alone. **Fix:** add `vad.NewVADInputProcessor` after `s.Input()`.

### Bot never stops talking when interrupted

- **Cause:** interruptions are disabled (`PipelineTaskConfig.AllowInterruptions = false`, or start strategies created with interruptions off). **Fix:** enable them; see [Turn-taking: interruptions](turn-taking.md#interruptions).
- **Cause:** a mute strategy is muting the user while the bot speaks. **Fix:** check `MuteStrategies` on the user aggregator.
- **Cause:** VAD never fires, so nothing detects the user speaking. **Fix:** see [VAD never triggers](#vad-never-triggers).
- **Cause:** the provider keeps playing its buffer because it never got the clear message. **Fix:** check that the serializer supports clear (see the [feature table](telephony-providers.md#feature-summary)), and look for write errors in the log.

### Bot goes silent after a tool call

- **Cause:** the function handler returned `RunLLM: false`, so the LLM is not asked to respond. **Fix:** leave `RunLLM` nil (the default) when the bot should speak after the tool.
- **Cause:** the handler returned a nil `Value` with a nil error. The assistant aggregator records `COMPLETED` but does not run the LLM again, even when `RunLLM` is nil or true. **Fix:** return a non-nil `Value`, such as a short status string. See [Function calling](function-calling.md).
- **Cause:** the handler is blocked (waiting on a slow API or a channel) with no timeout. **Fix:** give the function a timeout with `llm.FunctionOptions{Timeout: ...}` when you register it, and return when `ctx` is done.
- **Cause:** the function is not registered under the exact name in the tool schema, so it fails with `function "..." is not registered`. **Fix:** match the names in `llm.Tools` and `RegisterFunction`.
- **Cause:** the user is muted during function calls (`user_mute.NewFunctionCallUserMuteStrategy`) and the call never finishes. **Fix:** fix the stuck handler; the mute lifts when the call completes.

### High latency

- **Cause:** one stage dominates: endpointing wait, STT, LLM or TTS. **Fix:** add `observers.NewUserBotLatencyObserver` with `OnLatencyBreakdown` and read which contribution is largest. See [Observability](observability.md).
- **Cause:** a long speech timeout. **Fix:** shorten it; see [Turn-taking: stop strategies](turn-taking.md#stop-strategies).
- **Cause:** the LLM or TTS model, or its region, is slow. **Fix:** try a faster model or a closer region, and compare TTFB metrics.
- **Cause:** the TTS produces a high sample rate that is then downsampled for the phone. **Fix:** set `PipelineTaskConfig.AudioOutSampleRate = 8000` for phone bots.

### Call never ends

- **Cause:** nothing pushes an `EndFrame`, and the caller is waiting. **Fix:** give the LLM an `end_call` tool that queues an `EndFrame` (see [Twilio](telephony-twilio.md#auto-hang-up)), and set `SessionTimeout`.
- **Cause:** the `EndFrame` was sent but the provider did not hang up: missing credentials (Twilio, Telnyx, Plivo), `DisableAutoHangUp` set, or a provider that only closes the socket (Exotel, Vonage). **Fix:** add credentials and look for `hangup:` warnings; configure the provider's flow to hang up when the stream ends.
- **Cause:** `OnClientDisconnected` is set, which replaces the default cancel on disconnect. **Fix:** cancel or end the pipeline in your callback.
- **Cause:** idle calls are not timed out. **Fix:** set `PipelineTaskConfig.IdleTimeout` (with `CancelOnIdleTimeout`) or `SessionTimeout`.

### WebSocket closes immediately

- **Cause:** the handshake failed: the provider did not send its start message within `HandshakeTimeout`, or the first message did not parse. **Fix:** look for `handshake:` in the log, and confirm the provider and serializer match.
- **Cause:** `NewSerializer` returned `nil`. **Fix:** look for `no serializer for` in the log.
- **Cause:** your handler returned right away, often because building the pipeline failed (for example the VAD socket is missing). **Fix:** look for `handler:` in the log, and return errors from the handler instead of ignoring them.
- **Cause:** the server is draining (`ShutdownContext` was called) and answers `503`. **Fix:** this is expected during shutdown; route new calls to another instance.
- **Cause:** wrong URL: `ws://` instead of `wss://`, or a path that does not match `ServerConfig.Path` (the server answers `404`). **Fix:** correct the URL in the provider configuration.

### 401 or 403 from a provider

- **Cause:** an API key is missing, wrong, or for another region or project. Services log the HTTP status in their errors. **Fix:** check each key's environment variable at startup, and test the key with the provider's own tools.
- **Cause:** a Twilio hang-up `401`: wrong Auth Token, or an API key from another account. **Fix:** correct the `TwilioConfig` credentials.
- **Cause:** `403` on your own WebSocket upgrade: your `CheckOrigin` rejected the request. **Fix:** check the origin or signature logic (see [Deployment: security checklist](deployment.md#security-checklist)).

### VAD never triggers

- **Cause:** the ONNX worker is not running; VAD then logs `[SileroVAD] onnx-worker error` and reports zero confidence for every chunk. **Fix:** start the worker and check `ONNX_WORKER_SOCK` (see [Deployment](deployment.md#run-the-onnx-worker-sidecar)).
- **Cause:** `AudioInNative` is on, so `VADInputProcessor` receives µ-law bytes instead of PCM. **Fix:** turn native input off, or use the transport VAD hook. See [Audio](audio.md#where-vad-goes).
- **Cause:** the noise gate threshold is too high and mutes speech. **Fix:** lower `ThresholdDB` (for example to `-50`), or disable the filter to test.
- **Cause:** VAD parameters are too strict (`Confidence`, `MinVolume`, `StartSecs`). **Fix:** start from `vad.DefaultVADParams()` and adjust one at a time.
- **Cause:** the analyzer rate is not 8000 or 16000 Hz (`silero VAD requires 8000 or 16000 Hz`). **Fix:** create the analyzer at 16000 and keep the pipeline input rate at 8000 or 16000.

### Worker not found

- **Cause:** `onnx-worker binary unavailable`: `worker.Start("")` could not find or download the binary, or there is no prebuilt binary for your platform (Intel Mac, Windows). **Fix:** build it with `cd onnx-worker && cargo build --release` and pass the path to `worker.Start`.
- **Cause:** `failed to connect to onnx-worker at ...`: the socket path is wrong or the worker is not running. **Fix:** check the path and that the worker process is alive; with a sidecar, check the shared volume.
- **Cause:** `onnx-worker socket not ready after 15s`: the worker started but did not create the socket. **Fix:** read the worker's own log (it prints to your stdout and stderr); usually a bad model path.
- **Cause:** `ensure VAD model: ...`: the model download failed. **Fix:** pre-download the models into `~/.cache/strawgo/models/` (see the [Dockerfile](deployment.md#docker-image)).

### `undefined: transports.NewServer` or a missing Strawgo package

- **Cause:** your module depends on an older tagged release (`v0.1.0` to `v0.1.3`), which has the old API. You see `undefined: transports.NewServer` (or another new name), or `module github.com/square-key-labs/strawgo-ai@latest found (v0.1.3), but does not contain package .../src/llm`. This happens when `go get` ran without `@main`, or when `go mod tidy` ran before any code imported Strawgo and dropped the requirement. **Fix:** in your module folder, after your program file exists, run `go get github.com/square-key-labs/strawgo-ai@main && go mod tidy`. Check the result with `go list -m github.com/square-key-labs/strawgo-ai`: a pseudo-version such as `v0.1.4-0.2026...` means you are on `main`. See [Installation](../getting-started/installation.md#add-strawgo-to-your-module).

### Go version errors

- **Cause:** `go: go.mod requires go >= 1.25.9 (running go ...; GOTOOLCHAIN=local)`. Go 1.21 and newer download the required toolchain automatically, but `GOTOOLCHAIN=local` turns that off. **Fix:** unset `GOTOOLCHAIN` (or set it to `auto`), or install Go 1.25.9 or newer.
- **Cause:** your Go is 1.20 or older, which cannot download toolchains. **Fix:** install Go 1.25.9 or newer.

### Other build errors

- **Cause:** `no required module provides package github.com/pion/webrtc/v4` (or another dependency) in your own module. **Fix:** run `go mod tidy` after adding imports.
- **Cause:** errors mentioning `opus` or `pkg-config`: you built with `-tags opus` without the libopus development files. **Fix:** build without the tag (G.711 works in every browser), or install libopus and pkg-config. The `opus` build is unverified.
- **Cause:** `undefined: ...` after upgrading Strawgo: an API changed. **Fix:** check `CHANGELOG.md` and the [migration guide](../migration-v0.0.12.md).

## Provider-specific problems

- **Twilio**: see [Twilio troubleshooting](telephony-twilio.md#twilio-troubleshooting).
- **Jambonz**: if the log says `dropping audio: mixType=... is not usable`, set `"mixType": "mono"` on the `listen` verb.
- **Asterisk**: the bot's audio pauses while Asterisk sends `MEDIA_XOFF` and resumes on `MEDIA_XON`. Long pauses mean Asterisk's buffer is full.
- **Genesys**: `s.CallInfo()` is empty; read call data from the serializer (see [Genesys AudioHook](telephony-providers.md#genesys-audiohook)).
- **WebRTC**: `400 no supported audio codec offered` means the browser did not offer PCMU or PCMA; `404` on `PATCH` means the `pc_id` is unknown on this instance (see [Deployment: scaling](deployment.md#scaling-and-load-balancing)).

## What to include in a bug report

1. **Versions.** Strawgo (`go list -m github.com/square-key-labs/strawgo-ai`), Go (`go version`), OS and CPU architecture, and the onnx-worker version if you use VAD.
2. **Setup.** Transport and serializer, each service (STT, LLM, TTS) with its model, and the `TransportParams` and `PipelineTaskConfig` values you changed from the defaults.
3. **A minimal program** that shows the problem, ideally under 100 lines, with keys read from the environment.
4. **Logs for one call at `DEBUG` level** with `LOG_COLOR=false`, from the `connected` line to `closed`. Remove API keys, phone numbers and personal data.
5. **What you expected and what happened**, with times from the log if it is about timing. For latency, include the `UserBotLatencyObserver` breakdown lines.
6. **Raw provider messages** if the problem is at the transport level, for example the provider's start message with ids redacted.
7. **How often it happens**: every call, some calls, or only under load (and at what concurrency).

## Next steps

- [Observability](observability.md): metrics, latency breakdowns and tracing.
- [Turn-taking](turn-taking.md): interruptions and end-of-turn detection.
- [Deployment](deployment.md)
- [Transports overview](transports-overview.md)
