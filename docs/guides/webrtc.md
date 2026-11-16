# WebRTC

This guide lets people talk to your bot from a web browser. You build a Go server that accepts WebRTC connections and a small HTML page that connects to it with the browser's built-in `RTCPeerConnection`.

## Before you begin

- Read [Transports overview](transports-overview.md). WebRTC sessions work like WebSocket sessions: one session per peer, one handler call per session, one pipeline per handler.
- API keys for Deepgram (STT), OpenAI (LLM) and Cartesia (TTS), or any other providers from [Providers](../providers/README.md).
- A browser on the same machine (browsers only allow microphone access on `https://` pages or `http://localhost`).

Terms used on this page:

- **WebRTC** is the browser standard for real-time audio and video between two peers.
- **SDP offer and answer**: the browser describes what it can send and receive (the offer); the server replies with what it accepts (the answer). Exchanging them is called **signalling**.
- **ICE** is how the two peers find a network path to each other. **STUN** servers tell a peer its public address; **TURN** servers relay media when no direct path exists.
- A **data channel** carries text or binary messages alongside the audio.
- **G.711** (PCMU/PCMA) is the 8 kHz telephone codec. **Opus** is the higher-quality codec browsers prefer.

> [!NOTE]
> The WebRTC transport carries audio and data-channel messages only; there is no video. By default it negotiates G.711 (PCMU, then PCMA), which every browser supports. Opus is available only when you build with the `opus` build tag and have libopus installed, and that build is unverified. There is no Daily or LiveKit transport.

## The complete server

Save this as `main.go`. It serves the page below at `/` and the signalling endpoint at `/api/offer`.

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
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
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
		Params:     webrtc.Params{SessionTimeout: 10 * time.Minute},
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
	log.Printf("open http://localhost:7860")
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func runPeer(ctx context.Context, s *webrtc.Session) error {
	log.Printf("peer %s connected, request data %s", s.ID(), s.RequestData())
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: mustEnv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       mustEnv("OPENAI_API_KEY"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a friendly voice assistant in a web page. Keep answers short.",
	})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: mustEnv("CARTESIA_API_KEY"), VoiceID: mustEnv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	task := pipeline.NewPipelineTask(pipe)
	task.OnFrameReachedDownstream(func(f frames.Frame) {
		msg := f.(*frames.InputTransportMessageFrame).Message
		log.Printf("peer %s says over the data channel: %v", s.ID(), msg)
		if err := s.SendMessage(ctx, map[string]any{"type": "echo", "received": msg}); err != nil {
			log.Printf("send: %v", err)
		}
	}, (*frames.InputTransportMessageFrame)(nil))
	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hello from Strawgo. What can I do for you?")); err != nil {
		return err
	}
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

## The browser page

Save this as `index.html` next to `main.go`.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Strawgo voice</title>
</head>
<body>
  <button id="start">Start</button>
  <button id="stop" disabled>Stop</button>
  <button id="hello" disabled>Send "hello" on the data channel</button>
  <pre id="log"></pre>
  <audio id="bot" autoplay></audio>
  <script>
    const logEl = document.getElementById("log");
    const log = (line) => { logEl.textContent += line + "\n"; };
    let pc;
    let dc;
    let mic;

    async function start() {
      pc = new RTCPeerConnection({ iceServers: [{ urls: "stun:stun.l.google.com:19302" }] });
      dc = pc.createDataChannel("messages");
      dc.onopen = () => { document.getElementById("hello").disabled = false; };
      dc.onmessage = (event) => log("bot: " + event.data);
      pc.ontrack = (event) => { document.getElementById("bot").srcObject = new MediaStream([event.track]); };
      pc.onconnectionstatechange = () => log("connection: " + pc.connectionState);

      mic = await navigator.mediaDevices.getUserMedia({ audio: true });
      mic.getTracks().forEach((track) => pc.addTrack(track, mic));

      await pc.setLocalDescription(await pc.createOffer());
      await new Promise((resolve) => {
        if (pc.iceGatheringState === "complete") { resolve(); return; }
        pc.addEventListener("icegatheringstatechange", () => {
          if (pc.iceGatheringState === "complete") { resolve(); }
        });
      });

      const response = await fetch("/api/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          request_data: { user: "demo" }
        })
      });
      if (!response.ok) {
        log("offer failed: " + response.status + " " + await response.text());
        return;
      }
      const answer = await response.json();
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      log("peer id: " + answer.pc_id);
      document.getElementById("start").disabled = true;
      document.getElementById("stop").disabled = false;
    }

    function stop() {
      if (dc) { dc.close(); }
      if (pc) { pc.close(); }
      if (mic) { mic.getTracks().forEach((track) => track.stop()); }
      document.getElementById("start").disabled = false;
      document.getElementById("stop").disabled = true;
      document.getElementById("hello").disabled = true;
    }

    document.getElementById("start").onclick = () => start().catch((err) => log("error: " + err));
    document.getElementById("stop").onclick = stop;
    document.getElementById("hello").onclick = () => dc.send(JSON.stringify({ type: "hello", text: "hello" }));
  </script>
</body>
</html>
```

Run the server and open the page:

```bash
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export CARTESIA_API_KEY=YOUR_CARTESIA_API_KEY
export CARTESIA_VOICE_ID=YOUR_CARTESIA_VOICE_ID
go run .
```

Open `http://localhost:7860`, press **Start**, allow the microphone, and talk. For a styled widget built on the same pieces, see the [browser voice widget recipe](../cookbook/browser-voice-widget.md).

### How it works

1. The page creates a peer connection, a data channel and a microphone track, then creates an offer.
2. It waits until ICE gathering is complete so the offer already contains its network candidates, then `POST`s the offer to `/api/offer`.
3. `webrtc.Handler` picks the first codec from its list that the browser offered (PCMU by default), creates a `*webrtc.Session`, answers, and starts your `runPeer` handler on a new goroutine. The answer also contains all of the server's candidates, so no further signalling is needed.
4. When the connection reaches "connected", the session pushes a `ClientConnectedFrame` and starts delivering audio and data-channel messages.
5. When the peer goes away, the session cancels the pipeline, `runPeer` returns, and the handler forgets the peer.

## Signalling endpoints

`Handler.ServeHTTP` serves both methods on whatever path you mount it. If you need custom routing or authentication, call `HandleOffer(ctx, req, r)` and `HandleCandidates(ctx, req)` yourself.

### `POST`: send an offer

Request body (`webrtc.OfferRequest`):

```json
{
  "sdp": "v=0\r\no=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n...",
  "type": "offer",
  "pc_id": "",
  "restart_pc": false,
  "request_data": {"user": "demo"}
}
```

| Field | Meaning |
| --- | --- |
| `sdp`, `type` | The browser's offer. `type` must be `offer`. |
| `pc_id` | Empty for a new connection. Send the id from a previous answer to renegotiate that connection. |
| `restart_pc` | With `pc_id`, replace the peer connection instead of renegotiating it. |
| `request_data` (or `requestData`) | Any JSON. Your handler reads it with `s.RequestData()`. |

Response body (`webrtc.Answer`):

```json
{"sdp": "v=0\r\n...", "type": "answer", "pc_id": "pc-3f1c2a9e8b7d4c6a9e0f1b2c3d4e5f60"}
```

### `PATCH`: trickle ICE candidates

If your client sends candidates as it finds them instead of waiting for gathering to finish, send them after you have the answer's `pc_id`:

```json
{
  "pc_id": "pc-3f1c2a9e8b7d4c6a9e0f1b2c3d4e5f60",
  "candidates": [
    {"candidate": "candidate:1 1 udp 2122260223 192.168.1.20 54400 typ host", "sdp_mid": "0", "sdp_mline_index": 0}
  ]
}
```

The response is `{"status":"success"}`. Note the field names: the browser's `sdpMid` and `sdpMLineIndex` become `sdp_mid` and `sdp_mline_index`.

### Error responses

| Status | Cause |
| --- | --- |
| `400` | Bad JSON, `type` is not `offer`, invalid SDP, no supported audio codec offered, or a second connection in `SingleConnection` mode |
| `404` | `pc_id` not found or already closed |
| `405` | Any method other than `POST` or `PATCH` |
| `503` | The handler is shutting down |

## Configuration

`webrtc.Config`:

| Field | Default | Meaning |
| --- | --- | --- |
| `Handler` | required | Your `func(ctx context.Context, s *webrtc.Session) error`. |
| `ICEServers` | none | STUN and TURN servers given to every peer connection. Change them at runtime with `UpdateICEServers`. |
| `Codecs` | `webrtc.DefaultCodecs()` (PCMU, PCMA) | Codecs in order of preference. |
| `Params` | see below | Per-session behaviour. |
| `ConnectionMode` | `MultipleConnections` | `SingleConnection` allows only one peer at a time. |
| `GatherTimeout` | 10 s | How long to wait for the server's own ICE gathering before answering. |
| `SettingEngine` | none | A Pion `SettingEngine` for advanced network settings (port ranges, NAT 1:1 IPs). |

`webrtc.Params`:

| Field | Default | Meaning |
| --- | --- | --- |
| `AudioInSampleRate` | task input rate | Rate the input resamples caller audio to. |
| `ConnectTimeout` | 60 s | Give up if the peer never reaches "connected". |
| `DisconnectTimeout` | 5 s | How long a "disconnected" peer may recover before the session ends. |
| `DataChannelTimeout` | 10 s | How long outgoing messages wait for the data channel to open before being dropped. |
| `KeepAliveTimeout` | off | End the session if no `ping` message arrives on the data channel within this time. |
| `SessionTimeout`, `OnSessionTimeout` | off | Maximum session length. |
| `InterruptionFade` | 0 | Fade out instead of cutting on interruption. |
| `BotStopTimeout` | 3 s | Silence before the bot counts as stopped speaking. |
| `DisableAudioOutSilence` | false | Stop sending silence when the bot is quiet. |
| `NewAudioInAnalyzer`, `NewOutputMixer`, `NewInputFilter` | none | Build a voice activity analyzer, an output mixer and an input filter for each session. They override `AudioInAnalyzer`, `AudioOutMixer` and `InputFilter`. |
| `AudioInAnalyzer`, `AudioOutMixer`, `InputFilter` | none | One analyzer, mixer and filter shared by every peer. All keep per-call state, so prefer the factories when more than one peer can connect at a time. |
| `OnClientConnected`, `OnClientDisconnected` | none | Callbacks. Setting `OnClientDisconnected` replaces the default cancel. |

### ICE, STUN and TURN

A STUN server is enough when the browser and server can reach each other directly or through simple NAT. Behind strict firewalls or symmetric NAT you need a TURN server. Give the same servers to the browser and to the handler:

```go
func iceServers() []pwebrtc.ICEServer {
	return []pwebrtc.ICEServer{
		{URLs: []string{"stun:stun.l.google.com:19302"}},
		{
			URLs:       []string{"turn:turn.example.com:3478?transport=udp"},
			Username:   os.Getenv("TURN_USERNAME"),
			Credential: os.Getenv("TURN_PASSWORD"),
		},
	}
}
```

If your TURN credentials expire, refresh them with `handler.UpdateICEServers(iceServers())`. New peer connections use the new list.

On a server with a public IP behind a cloud NAT, a Pion `SettingEngine` with `SetNAT1To1IPs` (passed as `Config.SettingEngine`) lets the server advertise its public address.

### Codecs

| Codec | How to get it | Rate |
| --- | --- | --- |
| PCMU (G.711 µ-law) | default | 8000 Hz |
| PCMA (G.711 A-law) | default | 8000 Hz |
| Opus | build with `go build -tags opus` and install libopus | 48000 Hz |

With the `opus` tag, `webrtc.Opus` is put first in `DefaultCodecs()`. That build path has not been compiled or tested in this repository, so treat it as experimental. You can also implement the `webrtc.Codec` interface for your own codec and pass it in `Config.Codecs`.

## Data-channel messages

The browser must create the data channel (the server does not create one). Messages flow both ways:

| Direction | How | Format |
| --- | --- | --- |
| Browser to bot | `dc.send(JSON.stringify(...))` | Must be JSON. Each message becomes an `InputTransportMessageFrame` whose `Message` is the decoded value (`map[string]any` for objects). Non-JSON messages are ignored. Messages that arrive before the session is connected are held and delivered once it is. |
| Bot to browser | `s.SendMessage(ctx, msg)`, or push `frames.NewOutputTransportMessageFrame(msg)` (in order with audio) or `frames.NewOutputTransportMessageUrgentFrame(msg)` (immediately) | Strings are sent as text, `[]byte` as binary, anything else as JSON. |
| Keep-alive | Browser sends text starting with `ping` | Resets the `KeepAliveTimeout` timer. Not delivered to the pipeline. |

The server also sends two signalling messages of its own:

```json
{"type": "signalling", "message": {"type": "peerLeft"}}
```

is sent when the session closes, and

```json
{"type": "signalling", "message": {"type": "renegotiate"}}
```

is sent when you call `s.RequestRenegotiation(ctx)`. The browser should answer it by creating a new offer and posting it with the same `pc_id`. Messages from the browser with `"type": "signalling"` are logged and not delivered to the pipeline.

To speak the RTVI client protocol over this data channel, add the RTVI processor; see [RTVI](rtvi.md). The RTVI observer's default `OutputSources` already includes `WebRTCOutput`, so you do not need to set it.

## The WebRTC session

`*webrtc.Session` has the same `Input()`, `Output()`, `ID()`, `Request()`, `Close()`, `Done()` and `Wait()` methods as a WebSocket session, plus:

| Method | What it does |
| --- | --- |
| `RequestData()` | The raw `request_data` JSON from the offer. |
| `Codec()` | The negotiated codec, or `nil` if the offer had no audio. |
| `SendMessage(ctx, msg)` | Send a data-channel message. |
| `RequestRenegotiation(ctx)` | Ask the browser to renegotiate. |
| `Disconnected()` | A channel closed when the peer disconnects. |

There is no `CallInfo()`; use `RequestData()` and `Request()` to identify the user. Keypad tones (DTMF) are not supported over WebRTC.

## Limits

- Audio only. Video tracks from the browser are accepted and discarded.
- One audio track in each direction.
- No DTMF in or out.
- Opus requires the unverified `opus` build.

## Next steps

- [RTVI](rtvi.md): a structured protocol for browser clients on top of this transport.
- [Browser voice widget recipe](../cookbook/browser-voice-widget.md)
- [Deployment](deployment.md): TURN, scaling and shutdown in production.
- [Transports reference](../reference/transports.md)
