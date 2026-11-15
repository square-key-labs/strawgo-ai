# RTVI

RTVI (Real-Time Voice Inference) is a small JSON protocol between a voice bot and its client app. This guide wires Strawgo's RTVI support into a bot so a browser can see live transcripts, bot text, speaking events and metrics, and can send text and custom messages back.

## Before you begin

- Build the [WebRTC](webrtc.md) server and page first. RTVI rides on top of a transport that carries messages.
- Know the basic terms from [Core concepts](../getting-started/core-concepts.md). In short: a **frame** is one unit of data in the pipeline, an **observer** watches frames without changing them, **STT** turns speech into text, **TTS** turns text into speech, and an **interruption** is the user talking over the bot.

> [!NOTE]
> The RTVI package has not been run against an official RTVI client SDK. The message shapes below come from the code; test your client against them.

## What RTVI gives you

Every RTVI message is a JSON object with `"label": "rtvi-ai"`, a `type`, an optional `id` and a `data` object:

```json
{"label": "rtvi-ai", "type": "user-transcription", "data": {"text": "what's the weather", "user_id": "", "timestamp": "2026-01-01T12:00:00.000+00:00", "final": true}}
```

Strawgo has two pieces:

- `rtvi.Processor` sits in the pipeline. It reads messages from the client, answers them, and owns the `Send...` helpers.
- `rtvi.Observer` watches the pipeline and turns what happens (the user speaks, the bot generates text, metrics arrive) into RTVI messages for the client.

RTVI needs a transport that carries messages both ways:

| Transport | Works with RTVI |
| --- | --- |
| WebRTC | yes, over the data channel |
| WebSocket with `NewProtobufFrameSerializer` | yes, as JSON inside protobuf message frames |
| Telephony serializers (Twilio and others) | no; they drop RTVI messages |

## Wire it in

This is the WebRTC server from the [WebRTC guide](webrtc.md) with RTVI added.

```go
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	pwebrtc "github.com/pion/webrtc/v4"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/processors/rtvi"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports/webrtc"
)

func main() {
	logger.Init()
	handler, err := webrtc.NewHandler(webrtc.Config{
		ICEServers: []pwebrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
		Handler:    runPeer,
	})
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.Handle("/api/offer", handler)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "index.html")
	})
	srv := &http.Server{Addr: ":7860", Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = handler.Shutdown(shutdownCtx)
		_ = srv.Shutdown(shutdownCtx)
	}()
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func runPeer(ctx context.Context, s *webrtc.Session) error {
	rtviProc := rtvi.New(rtvi.Params{
		OnClientReady: func(p *rtvi.Processor) {
			log.Printf("peer %s: client ready, version %v", s.ID(), p.ClientVersion())
		},
		OnClientMessage: func(p *rtvi.Processor, msg rtvi.ClientMessage) {
			switch msg.Type {
			case "get-time":
				p.SendServerResponse(msg, map[string]any{"time": time.Now().Format(time.RFC3339)})
			default:
				p.SendErrorResponse(msg, "unknown message "+msg.Type)
			}
		},
	})

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: mustEnv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: mustEnv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a helpful voice assistant. Keep answers short."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: mustEnv("CARTESIA_API_KEY"), VoiceID: mustEnv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), rtviProc, stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)

	obs := pipeline.NewTaskObserver()
	obs.AddObserver(rtviProc.NewObserver(nil))
	task.SetObserver(obs)

	return task.Run(ctx)
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s is required", key)
	}
	return v
}
```

### How it works

1. `rtvi.New` creates the processor. Put it **right after `s.Input()`** so it sees client messages before anything else.
2. `rtviProc.NewObserver(nil)` creates the observer with `rtvi.DefaultObserverParams()` and links it to the processor, which sends the observer's messages. Pass a `*rtvi.ObserverParams` to change the defaults.
3. `OutputSources` names the processors whose text and interruptions count as "what the user heard". The default lists both built-in output processors, `WebSocketOutput` (`rtvi.DefaultOutputSource`) and `WebRTCOutput` (`rtvi.WebRTCOutputSource`), so it works over either transport without changes. Each frame is reported once even if it passes more than one listed output. Set the field only if you write your own output processor.
4. `cfg.EnableMetrics = true` makes services emit metrics, which the observer forwards as `metrics` messages.

## The browser side

In the page from the [WebRTC guide](webrtc.md), replace the data channel's `onopen` and `onmessage` handlers with these. The page now sends `client-ready` and prints transcripts and bot text.

```js
dc.onopen = () => {
  dc.send(JSON.stringify({
    label: "rtvi-ai",
    type: "client-ready",
    id: "ready-1",
    data: { version: "2.3.0", about: { library: "hand-written-page" } }
  }));
  document.getElementById("hello").disabled = false;
};
dc.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.label !== "rtvi-ai") { return; }
  if (msg.type === "bot-ready") { log("bot ready, protocol " + msg.data.version); }
  if (msg.type === "user-transcription" && msg.data.final) { log("you: " + msg.data.text); }
  if (msg.type === "bot-tts-text") { log("bot: " + msg.data.text); }
  if (msg.type === "server-response") { log("response to " + msg.data.t + ": " + JSON.stringify(msg.data.d)); }
};
document.getElementById("hello").onclick = () => dc.send(JSON.stringify({
  label: "rtvi-ai", type: "client-message", id: "msg-1", data: { t: "get-time", d: {} }
}));
```

## The handshake

1. The client sends `client-ready` with its protocol version:

   ```json
   {"label": "rtvi-ai", "type": "client-ready", "id": "ready-1", "data": {"version": "2.3.0", "about": {"library": "hand-written-page"}}}
   ```

2. The processor records the version (read it with `ClientVersion()`), pushes an `InputTransportStartAudioStreamingFrame`, and calls `OnClientReady`. If the version is missing, malformed, or has a major version other than 2 (or the legacy 1), it also sends an `error-response`, but carries on.
3. Unless `Params.DisableAutoBotReady` is set, it replies with `bot-ready`, using the client's `id`:

   ```json
   {"label": "rtvi-ai", "type": "bot-ready", "id": "ready-1", "data": {"version": "2.3.0", "about": {"library": "strawgo-ai", "library_version": "v0.1.0"}}}
   ```

   `about` comes from `Params.About` if set; otherwise it names the library and the Strawgo module version found in your binary's build info. `Params.Capabilities` adds a `capabilities` object. With `DisableAutoBotReady`, call `rtviProc.SetBotReady(about)` yourself when your bot is ready, for example after loading user data.

`IsClientReady()` and `IsBotReady()` report the state. `OnBotStarted` runs once when the pipeline starts.

## Messages the bot sends

`rtvi.DefaultObserverParams()` turns most messages on. Each row names the `ObserverParams` switch.

| Type | When | Switch (default) |
| --- | --- | --- |
| `bot-ready` | Reply to `client-ready` | always |
| `error` | An `ErrorFrame` passed the processor; `data.fatal` says whether the pipeline stops | always |
| `error-response` | A client message was invalid or unsupported | always |
| `server-message` | You called `SendServerMessage` or pushed `rtvi.NewServerMessageFrame` | always |
| `server-response` | You called `SendServerResponse` or pushed `rtvi.NewServerResponseFrame` | always |
| `user-started-speaking`, `user-stopped-speaking` | The user's turn started or ended | `UserSpeakingEnabled` (on) |
| `vad-user-started-speaking`, `vad-user-stopped-speaking` | Voice activity detection fired | `VADUserSpeakingEnabled` (off) |
| `user-mute-started`, `user-mute-stopped` | User input was muted or unmuted | `UserMuteEnabled` (on) |
| `user-transcription` | Interim and final STT results | `UserTranscriptionEnabled` (on) |
| `user-input` | Transcriptions, `send-text` messages and backchannels, with `input_type` | `UserInputEnabled` (on) |
| `user-llm-text` | The user text the LLM was given | `UserLLMEnabled` (on) |
| `bot-llm-started`, `bot-llm-text`, `bot-llm-stopped` | The LLM's response as it streams | `BotLLMEnabled` (on) |
| `bot-transcription` | LLM text grouped into sentences | `BotLLMEnabled` (on) |
| `bot-llm-marker` | A response marker from the LLM | `BotLLMMarkerEnabled` (off) |
| `bot-tts-started`, `bot-tts-text`, `bot-tts-stopped` | TTS activity and the text being spoken | `BotTTSEnabled` (on) |
| `bot-output` | Spoken text with progress (`spoken_status`, `spoken_progress`) | `BotOutputEnabled` (on) |
| `bot-started-speaking`, `bot-stopped-speaking`, `bot-interrupted` | Bot audio started, stopped, or was interrupted | `BotSpeakingEnabled` (on) |
| `metrics` | TTFB, processing time, token usage, STT seconds, TTS characters | `MetricsEnabled` (on) |
| `llm-function-call-started`, `llm-function-call-in-progress`, `llm-function-call-stopped` | Function call lifecycle | `FunctionCallReportLevel` (see below) |
| `stt-raw-text`, `llm-raw-text`, `tts-raw-text` | Raw text streams, for clients 2.2.0 or newer | `STTRawTextEnabled`, `LLMRawTextEnabled`, `TTSRawTextEnabled` (off) |
| `user-audio-level`, `bot-audio-level` | Smoothed volume every `AudioLevelPeriod` (150 ms) | `UserAudioLevelEnabled`, `BotAudioLevelEnabled` (off) |

Other observer settings: `IgnoredSources` (processor names to ignore), `SkipTextTypes` (text types to leave out of `bot-output`), and `AudioLevelPeriod`.

### Function-call reporting

`FunctionCallReportLevel` maps a function name (or `"*"` for all) to how much the client sees:

| Level | Client sees |
| --- | --- |
| `FunctionCallReportDisabled` | nothing |
| `FunctionCallReportNone` (default) | that a call happened, without the name |
| `FunctionCallReportName` | the function name |
| `FunctionCallReportArguments` | name and arguments |
| `FunctionCallReportFull` | name, arguments and result |

```go
func observerParams() rtvi.ObserverParams {
	p := rtvi.DefaultObserverParams()
	p.FunctionCallReportLevel = map[string]rtvi.FunctionCallReportLevel{
		"*":           rtvi.FunctionCallReportName,
		"get_weather": rtvi.FunctionCallReportFull,
	}
	return p
}
```

You can change some settings mid-call by pushing an `rtvi.ConfigureObserverFrame` with `FunctionCallReportLevel`, `VADUserSpeakingEnabled` or `BotLLMMarkerEnabled` set.

## Messages the client sends

| Type | `data` | What the processor does |
| --- | --- | --- |
| `client-ready` | `version`, `about` | The handshake above. |
| `disconnect-bot` | none | Pushes an `EndTaskFrame`, ending the call gracefully. |
| `client-message` | `t` (string, required), `d` (any) | Calls `OnClientMessage` and pushes an `rtvi.ClientMessageFrame`. |
| `send-text` | `content`, optional `options` | Adds `content` as a user message and runs the LLM. |
| `send-file` | `file` (`format`, `name`, `source`), optional `content`, `options` | Adds an image or file to the conversation. |
| `llm-function-call-result` | `function_name`, `tool_call_id`, `arguments`, `result` | Pushes a `FunctionCallResultFrame`. |
| `raw-audio`, `raw-audio-batch` | `base64Audio` or `base64AudioBatch`, `sampleRate`, `numChannels` | Pushes 16-bit PCM audio frames. |
| `dtmf` | `buttons` (array of `"0"`-`"9"`, `"*"`, `"#"`) | Pushes `InputDTMFFrame`s. |

Unknown types get an `error-response` with `Unsupported type ...`.

### Send text

```json
{"label": "rtvi-ai", "type": "send-text", "id": "t-1", "data": {"content": "Summarize what we said", "options": {"run_immediately": true, "audio_response": true}}}
```

`run_immediately` (default true) interrupts the bot and runs the LLM now; set it false to add the message without a response. `audio_response` (default true) set to false makes the bot answer in text only (`bot-llm-text`), without TTS.

### Answer a client message

`OnClientMessage` receives an `rtvi.ClientMessage` with `MsgID`, `Type` (the `t` field) and `Data` (the `d` field). Reply with:

- `p.SendServerResponse(msg, data)`: sends `server-response` with `data: {"t": <type>, "d": <data>}` and the client's `id`.
- `p.SendErrorResponse(msg, "reason")`: sends `error-response`.

If another processor should answer, let it handle the `rtvi.ClientMessageFrame` and push `rtvi.NewServerResponseFrame(clientMsg, data)`; the observer sends it.

To push a message the client did not ask for, call `rtviProc.SendServerMessage(data)`.

## Functions handled by the client

Sometimes the client should run a function, for example reading the browser's location. Strawgo's LLM services run only functions registered in Go, and fail immediately for unknown names. So register a Go handler that asks the client and waits for its answer:

```go
type clientCalls struct {
	mu      sync.Mutex
	waiting map[string]chan any
}

func (c *clientCalls) handler(p *rtvi.Processor) llm.FunctionHandler {
	return func(ctx context.Context, call llm.FunctionCallParams) (llm.FunctionResult, error) {
		ch := make(chan any, 1)
		c.mu.Lock()
		c.waiting[call.ToolCallID] = ch
		c.mu.Unlock()
		defer func() {
			c.mu.Lock()
			delete(c.waiting, call.ToolCallID)
			c.mu.Unlock()
		}()
		p.SendServerMessage(map[string]any{
			"type":         "run-function",
			"tool_call_id": call.ToolCallID,
			"name":         call.FunctionName,
			"arguments":    call.Arguments,
		})
		select {
		case result := <-ch:
			return llm.FunctionResult{Value: result}, nil
		case <-ctx.Done():
			return llm.FunctionResult{}, ctx.Err()
		}
	}
}

func (c *clientCalls) onClientMessage(p *rtvi.Processor, msg rtvi.ClientMessage) {
	if msg.Type != "function-result" {
		return
	}
	d, _ := msg.Data.(map[string]any)
	id, _ := d["tool_call_id"].(string)
	c.mu.Lock()
	ch := c.waiting[id]
	c.mu.Unlock()
	if ch == nil {
		p.SendErrorResponse(msg, "no pending call "+id)
		return
	}
	ch <- d["result"]
}
```

Create one `clientCalls{waiting: map[string]chan any{}}` per session, pass `calls.onClientMessage` as `Params.OnClientMessage`, and register the function with `llmSvc.RegisterFunction("get_location", calls.handler(rtviProc))`. The client listens for the `server-message` and answers with:

```json
{"label": "rtvi-ai", "type": "client-message", "id": "f-1", "data": {"t": "function-result", "d": {"tool_call_id": "call_abc123", "result": {"city": "Lisbon"}}}}
```

Use a `client-message` rather than `llm-function-call-result` for this. The processor turns `llm-function-call-result` into a `FunctionCallResultFrame` directly, which the LLM service does not match to its own pending call, so the Go handler would still be waiting. Give the function a timeout (`llm.FunctionOptions{Timeout: ...}` as the third argument of `RegisterFunction`) so a client that never answers does not stall the bot. See [Function calling](function-calling.md) for the rest of the function API.

## Next steps

- [WebRTC](webrtc.md)
- [Browser voice widget recipe](../cookbook/browser-voice-widget.md)
- [Function calling](function-calling.md)
- [Observability](observability.md): the metrics behind the `metrics` message.
