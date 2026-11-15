# Transports

A transport connects a pipeline to the outside world. Every connection (a phone call, a browser peer) gets its own session with an input processor and an output processor, and the application builds one pipeline per session.

| Transport | Package | Use |
| --- | --- | --- |
| WebSocket server | `transports` (`NewServer`) | telephony media streams (Twilio, Telnyx, Plivo, Exotel, Vonage, Asterisk, Genesys, Jambonz) and WebSocket clients (Protobuf) |
| WebSocket client | `transports` (`Dial`) | the bot connects out to a WebSocket endpoint |
| WebRTC | `transports/webrtc` (`NewHandler`) | browsers and other WebRTC peers, offer/answer over HTTP |

There is no Daily or LiveKit transport and no video support.

## WebSocket server

```
transports.NewServer(transports.ServerConfig{
	Addr          string
	Path          string
	NewSerializer func(r *http.Request) serializers.FrameSerializer
	Handler       transports.SessionHandler
	Params        transports.TransportParams
	CheckOrigin   func(r *http.Request) bool
})

type SessionHandler func(ctx context.Context, s *transports.Session) error
```

For each upgraded connection the server:

1. calls `NewSerializer(r)` to get a fresh serializer (return `nil` to reject);
2. if the serializer implements `serializers.Handshaker`, reads the provider's opening messages (for example Twilio `connected`/`start`) until the handshake completes, within `Params.HandshakeTimeout`;
3. starts the session and calls `Handler(ctx, session)` on the connection's goroutine;
4. closes the session when the handler returns and waits for its goroutines.

`ListenAndServe(ctx)` serves on `Addr` until `ctx` is cancelled. `Serve(ctx, ln)` uses your listener; `Server` is also an `http.Handler`, so it can be mounted on an existing mux. `Shutdown()` closes every session.

`*transports.Session`:

| Method | Purpose |
| --- | --- |
| `Input()`, `Output()` | the processors to put first and (near) last in the pipeline |
| `CallInfo()` | `serializers.CallInfo` from the handshake: provider, stream id, call id, account, from, to, encoding, sample rate, custom parameters |
| `Request()` | the HTTP upgrade request (query string, headers) |
| `ID()`, `Serializer()` | identifiers |
| `Close()`, `Done()`, `Wait()` | lifecycle |

The handler usually builds a pipeline around `s.Input()` and `s.Output()` and returns `task.Run(ctx)`; see the [README quickstart](../README.md#quickstart) and [examples/twilio](../examples/twilio/main.go).

Behaviour of the session processors:

- Input resamples the serializer's PCM audio to the pipeline's input rate (native-encoded audio from `AudioInNative` is passed through), runs the optional input filter and VAD hook, forwards control frames from the serializer, turns an `InputTransportMessageFrame` into a broadcast, and blocks delivery until the `StartFrame` has passed.
- Output resamples and encodes TTS audio for the wire, paces it in real time in `AudioOutChunk` pieces (default 20 ms), emits `BotStartedSpeakingFrame`/`BotStoppedSpeakingFrame`, and on interruption drops queued audio and writes the serializer's clear message.
- When the serializer implements `PlaybackAckSerializer` and `Params.PlaybackAck` is set, `BotStoppedSpeakingFrame` waits for the client's playback acknowledgement (Twilio `mark`, Asterisk queue-drained), falling back after `PlaybackAckTimeout`.
- On `EndFrame` the output drains audio, sends the serializer's end message, calls `Hangup` if the serializer implements `serializers.Hangupper`, then closes the connection.
- When the client disconnects, input pushes `ClientDisconnectedFrame` and a `CancelTaskFrame` upstream, unless `Params.OnClientDisconnected` is set.

Selected `TransportParams` fields:

| Field | Meaning |
| --- | --- |
| `AudioInSampleRate`, `AudioOutSampleRate` | override the rates from the `StartFrame` / serializer |
| `AudioOutChunk` | size of each paced output write |
| `AudioInAnalyzer` / `NewAudioInAnalyzer` | VAD run inside the input processor (`transports.NewVADAnalyzerHook`) |
| `InputFilter` / `NewInputFilter`, `OutputMixer` / `NewOutputMixer` | `audio/filter` and `audio/mixer` hooks; the `New...` variants build one per session |
| `AudioInNative`, `AudioOutNative` | pass the provider's native encoding (for example mu-law) through without converting to PCM, when the serializer implements `NativeAudioSerializer` |
| `PlaybackAck`, `PlaybackAckTimeout`, `BotStopTimeout`, `AudioOutEndSilence` | bot-stopped-speaking detection and end-of-call padding |
| `SessionTimeout`, `OnSessionTimeout` | maximum session length |
| `WriteTimeout`, `PingInterval`, `HandshakeTimeout`, `CloseTimeout`, `HangupTimeout` | connection timeouts |
| `OnClientConnected`, `OnClientDisconnected` | callbacks |

`transports.Dial(ctx, transports.ClientConfig{URL, Header, Serializer, Params, Dialer})` returns the same `*Session` for an outgoing connection.

## Serializers

```
type FrameSerializer interface {
	SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error)
	Serialize(dst []Message, frame frames.Frame) ([]Message, error)
	Deserialize(msg Message) (frames.Frame, error)
}
```

`Message` is a typed WebSocket message (`serializers.Text(...)`, `serializers.Binary(...)`). Serializers append to `dst` so the output path can reuse buffers. A serializer instance belongs to one session.

Optional interfaces: `Handshaker`, `PlaybackAckSerializer`, `Hangupper`, `NativeAudioSerializer`, `DTMFSerializer`, `DTMFSender`, `WireFormat`, `LabeledMessage`.

| Serializer | Constructor | Handshake | Playback ack | Hangup | Outbound DTMF |
| --- | --- | --- | --- | --- | --- |
| Twilio Media Streams | `NewTwilioFrameSerializer(TwilioConfig)` | yes | yes | yes (REST, needs credentials) | |
| Telnyx | `NewTelnyxFrameSerializer(TelnyxConfig)` | yes | yes | yes | yes |
| Plivo | `NewPlivoFrameSerializer(PlivoConfig)` | yes | yes | yes | yes |
| Exotel | `NewExotelFrameSerializer(ExotelConfig)` | yes | yes | | |
| Vonage | `NewVonageFrameSerializer(VonageConfig)` | yes | yes | | |
| Asterisk WebSocket | `NewAsteriskFrameSerializer(AsteriskSerializerConfig)` | yes | yes | | |
| Genesys AudioHook | `NewGenesysAudioHookSerializer(GenesysConfig)` | | | | |
| Jambonz | `NewJambonzFrameSerializer(JambonzConfig)` | yes | | | |
| Protobuf (Pipecat client format) | `NewProtobufFrameSerializer()` | | | | |
| RTVI client JSON | `NewRTVIClientSerializer()` | | | | |

Only the Twilio and Asterisk playback-ack paths have tests; none of these has been run against the real provider.

## WebRTC

`transports/webrtc` implements the SmallWebRTC-style signalling used by Pipecat clients on top of Pion:

- `POST` with an `OfferRequest` JSON body (`sdp`, `type`, `pc_id`, `restart_pc`, `request_data`) returns an `Answer` (`sdp`, `type`, `pc_id`);
- `PATCH` with a `PatchRequest` (`pc_id`, `candidates`) adds trickled ICE candidates.

`Handler.ServeHTTP` handles both methods; `HandleOffer` and `HandleCandidates` are available for custom routing. Each new peer connection gets a `*webrtc.Session` and the handler's `SessionHandler` runs for it, exactly like the WebSocket server.

Audio codecs: `webrtc.DefaultCodecs()` is PCMU and PCMA (G.711, 8 kHz). Opus is only registered when built with `-tags opus`, which needs libopus; that build has never been compiled, so treat Opus as unavailable. Browsers negotiate G.711 without it.

`*webrtc.Session` adds `SendMessage(ctx, msg)` (data channel), `RequestData()` (the offer's `request_data`), `Codec()`, `Disconnected()` and `RequestRenegotiation(ctx)` to the WebSocket session's methods. Data-channel messages from the peer arrive in the pipeline as `InputTransportMessageFrame`.

A WebRTC server with the RTVI protocol enabled:

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/llm"
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
	handler, err := webrtc.NewHandler(webrtc.Config{Handler: runPeer})
	if err != nil {
		log.Fatal(err)
	}
	http.Handle("/api/offer", handler)
	log.Fatal(http.ListenAndServe(":7860", nil))
}

func runPeer(ctx context.Context, s *webrtc.Session) error {
	rtviProc := rtvi.New(rtvi.Params{})
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a helpful assistant."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: os.Getenv("CARTESIA_API_KEY"), VoiceID: os.Getenv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})

	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), rtviProc, stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	task := pipeline.NewPipelineTask(pipe)

	obs := pipeline.NewTaskObserver()
	obs.AddObserver(rtviProc.NewObserver(nil))
	task.SetObserver(obs)
	return task.Run(ctx)
}
```

`webrtc.Config` also takes `ICEServers`, `Params` (timeouts, VAD hook, mixer, callbacks), `Codecs`, `SettingEngine`, `ConnectionMode` (`MultipleConnections` or `SingleConnection`) and `GatherTimeout`. `Handler.Shutdown(ctx)` closes every peer.

## RTVI

`processors/rtvi` implements the server side of the RTVI client protocol (version constant `rtvi.ProtocolVersion`).

- `rtvi.New(rtvi.Params{...})` creates the processor. Put it right after the transport input. It parses client messages from `InputTransportMessageFrame` (`client-ready`, `disconnect-bot`, `client-message`, `llm-function-call-result`, `send-text`, `send-file`, `raw-audio`, `raw-audio-batch`, `dtmf`), answers through `OutputTransportMessageUrgentFrame`, and sends `bot-ready` automatically when the client is ready unless `DisableAutoBotReady` is set (then call `SetBotReady`). Callbacks: `OnBotStarted`, `OnClientReady`, `OnClientMessage`.
- `rtviProc.NewObserver(&params)` (or `rtvi.NewObserver`) returns a `pipeline.Observer` that turns pipeline activity into RTVI server messages: user and bot speaking, transcriptions, LLM and TTS text, function-call events, metrics, audio levels. `rtvi.DefaultObserverParams()` lists the switches; function-call reporting is off unless `FunctionCallReportLevel` enables it.
- Client messages need a transport that carries them: the WebRTC data channel, or a WebSocket session whose serializer decodes them (`NewProtobufFrameSerializer`). Telephony serializers do not.

The RTVI package has no tests and has not been run against an RTVI client.
