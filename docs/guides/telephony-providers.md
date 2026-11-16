# Telephony providers

Strawgo talks to every telephony provider through the same WebSocket server. Only the serializer changes. This page shows, for each supported provider, what the provider must send, how to construct the serializer, and which call-control features work.

## Before you begin

- Read [Transports overview](transports-overview.md). This page assumes you know `transports.NewServer`, sessions and `TransportParams`.
- [Twilio](telephony-twilio.md) is the most detailed walkthrough. The ideas there (handshake, call info, DTMF, hang-up, clear, playback acknowledgements) apply to every provider below.

Terms used on this page:

- A **serializer** translates one provider's WebSocket messages into Strawgo frames and back.
- **DTMF** is the tone a phone keypad makes. "DTMF in" means the bot receives key presses; "DTMF out" means the bot sends them.
- **Clear** is the message that tells the provider to drop audio it has buffered, used when the caller interrupts the bot.
- A **playback acknowledgement** is a marker the bot sends after its audio; the provider echoes it once the caller has heard everything before it.
- **Native µ-law passthrough** sends 8 kHz µ-law audio from a TTS service straight to the wire without converting it. See [Audio](audio.md#native-passthrough).

> [!NOTE]
> Every serializer here is unit-tested against the message shapes in this repository's tests, and the Twilio, Telnyx and Plivo REST hang-ups are tested against a mock server. None has been run against the live provider. Treat the provider-side configuration snippets as starting points and confirm them against your provider's documentation.

## Feature summary

| Serializer | DTMF in | DTMF out | Hang-up on `EndFrame` | Clear on interruption | Playback ack | Native µ-law passthrough |
| --- | --- | --- | --- | --- | --- | --- |
| Twilio | yes | in-band tones | REST API (needs credentials) | `clear` | `mark` | yes |
| Telnyx | yes | REST `send_dtmf` when `APIKey` is set, otherwise in-band tones | REST API (needs `APIKey`) | `clear` | `mark` | yes |
| Plivo | yes | REST DTMF when `AuthID`/`AuthToken` are set, otherwise in-band tones | REST API (needs `AuthID`/`AuthToken`) | `clearAudio` | `checkpoint` / `playedStream` | yes |
| Exotel | yes | in-band tones | no (closes the socket) | `clear` | `mark` | no (wire is 16-bit PCM) |
| Vonage | yes | in-band tones | no (closes the socket) | `{"action":"clear"}` | `notify` | no (wire is 16-bit PCM) |
| Asterisk | yes | in-band tones | `HANGUP` command | `FLUSH_MEDIA` | `MARK_MEDIA` | yes, when the channel codec is µ-law or A-law |
| Genesys AudioHook | yes | in-band tones | `disconnect` message | `barge_in` event | no | yes |
| Jambonz | yes | in-band tones | `{"type":"disconnect"}` | `{"type":"killAudio"}` | no | no (wire is 16-bit PCM) |
| Protobuf | no | in-band tones | no (closes the socket) | interruption frame | no | no |

"In-band tones" means Strawgo generates the key tones (from `audio/dtmf`) and plays them as audio. You send digits the same way for every provider: queue a `frames.OutputDTMFFrame`, as shown in [Twilio: DTMF out](telephony-twilio.md#dtmf-out).

Playback acknowledgements are used when you set `TransportParams.PlaybackAck`, and always at the end of a call so the final words are heard before the hang-up.

To transfer a call to a person or another number, or to give the LLM ready-made `transfer_call`, `end_call` and `send_dtmf` tools, see [Call control](call-control.md). It has a per-provider table of which serializers can transfer.

## One server, many providers

`transports.Server` is an `http.Handler`, so one process can serve several providers on different paths. This complete program answers Twilio, Telnyx and Plivo calls with the same bot.

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

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

func main() {
	logger.Init()
	routes := map[string]func(*http.Request) serializers.FrameSerializer{
		"/twilio": func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		"/telnyx": func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTelnyxFrameSerializer(serializers.TelnyxConfig{APIKey: os.Getenv("TELNYX_API_KEY")})
		},
		"/plivo": func(*http.Request) serializers.FrameSerializer {
			return serializers.NewPlivoFrameSerializer(serializers.PlivoConfig{
				AuthID:    os.Getenv("PLIVO_AUTH_ID"),
				AuthToken: os.Getenv("PLIVO_AUTH_TOKEN"),
			})
		},
	}
	mux := http.NewServeMux()
	var servers []*transports.Server
	for path, newSerializer := range routes {
		srv, err := transports.NewServer(transports.ServerConfig{Path: path, NewSerializer: newSerializer, Handler: runCall})
		if err != nil {
			log.Fatal(err)
		}
		mux.Handle(path, srv)
		servers = append(servers, srv)
	}
	httpSrv := &http.Server{Addr: ":8080", Handler: mux, ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		for _, srv := range servers {
			srv.Shutdown()
		}
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutdownCtx)
	}()
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func runCall(ctx context.Context, s *transports.Session) error {
	info := s.CallInfo()
	log.Printf("%s call %s from %q to %q", info.Provider, info.CallID, info.From, info.To)
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a helpful phone assistant. Keep answers short."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: os.Getenv("CARTESIA_API_KEY"), VoiceID: os.Getenv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	return pipeline.NewPipelineTask(pipe).Run(ctx)
}
```

Each section below shows the `NewSerializer` function for one provider. Swap it into a server like the one above.

## Telnyx

**What Telnyx must send.** Start a bidirectional media stream to `wss://YOUR_HOST/telnyx` with Call Control. The serializer expects JSON messages with these shapes:

```json
{"event": "start", "stream_id": "YOUR_STREAM_ID", "start": {"call_control_id": "YOUR_CALL_CONTROL_ID", "from": "+15551234567", "to": "+15557654321", "client_state": "YOUR_STATE", "media_format": {"encoding": "PCMU", "sample_rate": 8000}}}
```

followed by `media` (`{"event":"media","media":{"payload":"BASE64_AUDIO"}}`), `dtmf` (`{"event":"dtmf","dtmf":{"digit":"5"}}`), `mark` echoes and finally `stop`.

**Constructor and config.**

```go
func telnyxSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewTelnyxFrameSerializer(serializers.TelnyxConfig{
		APIKey: os.Getenv("TELNYX_API_KEY"),
	})
}
```

| `TelnyxConfig` field | Default | Meaning |
| --- | --- | --- |
| `APIKey` | none | Enables REST hang-up and REST DTMF out. |
| `InboundEncoding` | from `start`, else `PCMU` | Encoding of caller audio. |
| `OutboundEncoding` | same as inbound | Encoding of bot audio. |
| `StreamID`, `CallControlID` | from `start` | Only needed if you know them in advance. |
| `BaseURL` | `https://api.telnyx.com` | API root. |
| `DisableAutoHangUp` | false | Skip the hang-up request. |
| `HTTPClient` | `http.DefaultClient` | Client for REST calls. |

**Call info.** `CallID` is the call control id, `StreamID` the stream id, `From` and `To` the numbers, and `CustomParameters["client_state"]` the client state.

**Call control.** Hang-up is `POST /v2/calls/{call_control_id}/actions/hangup` with a bearer token; a "call already ended" response (`422` with code `90018`) counts as success. DTMF out is `POST /v2/calls/{call_control_id}/actions/send_dtmf` with 250 ms per digit. Clear is `{"event":"clear"}`, and playback acknowledgements use `{"event":"mark","mark":{"name":"..."}}`.

## Plivo

**What Plivo must send.** Use a bidirectional `<Stream>` in your Plivo XML, with µ-law at 8000 Hz. For example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">wss://YOUR_HOST/plivo</Stream>
</Response>
```

The serializer expects a `start` message (`{"event":"start","start":{"callId":"...","streamId":"...","accountId":"..."},"extra_headers":"{\"customer_id\":\"12345\"}"}`), then `media`, `dtmf`, `playedStream` and `stop` events.

**Constructor and config.**

```go
func plivoSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewPlivoFrameSerializer(serializers.PlivoConfig{
		AuthID:    os.Getenv("PLIVO_AUTH_ID"),
		AuthToken: os.Getenv("PLIVO_AUTH_TOKEN"),
	})
}
```

| `PlivoConfig` field | Meaning |
| --- | --- |
| `AuthID`, `AuthToken` | Enable REST hang-up and REST DTMF out. |
| `StreamID`, `CallID` | Filled from `start`; set only if known in advance. |
| `BaseURL` | API root, default `https://api.plivo.com`. |
| `DisableAutoHangUp`, `HTTPClient` | As for Twilio. |

**Call info.** `CallID`, `StreamID` and `AccountID` come from `start`. `extra_headers` is a JSON string; when it decodes to an object of strings, it becomes `CustomParameters`.

**Call control.** Hang-up is `DELETE /v1/Account/{AuthID}/Call/{CallID}/`; a `404` counts as success. DTMF out is `POST /v1/Account/{AuthID}/Call/{CallID}/DTMF/`. Bot audio is sent as `playAudio`, clear as `clearAudio`, and playback acknowledgements as `checkpoint`, which Plivo answers with `playedStream`.

## Exotel

**What Exotel must send.** Point an Exotel voice-streaming (bidirectional) applet at `wss://YOUR_HOST/exotel`. The serializer expects 16-bit little-endian PCM in base64 and these messages:

```json
{"event": "start", "stream_sid": "YOUR_STREAM_SID", "start": {"stream_sid": "YOUR_STREAM_SID", "call_sid": "YOUR_CALL_SID", "account_sid": "YOUR_ACCOUNT_SID", "from": "+15551234567", "to": "+15557654321", "custom_parameters": {"customer_id": "12345"}, "media_format": {"sample_rate": "8000"}}}
```

then `media`, `dtmf`, `mark` and `stop`.

**Constructor and config.**

```go
func exotelSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewExotelFrameSerializer(serializers.ExotelConfig{})
}
```

| `ExotelConfig` field | Meaning |
| --- | --- |
| `SampleRate` | Fixes the wire rate. Zero means 8000, or the rate from `start` (`sample_rate` may be a number or a string). |
| `StreamSID`, `CallSID` | Filled from `start`. |

**Call info.** `CallID`, `StreamID`, `AccountID`, `From`, `To` and `CustomParameters` all come from `start`.

**Call control.** There is no REST hang-up; on `EndFrame` the session closes the WebSocket. Configure Exotel's flow to end the call when the stream ends. Clear is `{"event":"clear","stream_sid":"..."}`.

## Vonage

**What Vonage must send.** Return an NCCO with a `connect` action to a WebSocket endpoint, using 16-bit linear PCM:

```json
[
  {
    "action": "connect",
    "endpoint": [
      {
        "type": "websocket",
        "uri": "wss://YOUR_HOST/vonage",
        "content-type": "audio/l16;rate=16000",
        "headers": {"customer_id": "12345"}
      }
    ]
  }
]
```

Vonage first sends a text message with `"event": "websocket:connected"` and the `content-type`, then binary audio frames.

**Constructor and config.**

```go
func vonageSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewVonageFrameSerializer(serializers.VonageConfig{})
}
```

`VonageConfig.SampleRate` fixes the wire rate. Zero means 16000, or the `rate=` value from the `content-type` in the connected message.

**Call info.** Every string field of the connected message (other than `event` and `content-type`) becomes a `CustomParameters` entry, so your NCCO `headers` show up there. `CallID` is taken from `call_uuid`, `uuid` or `conversation_uuid`, whichever is present.

**Call control.** Bot audio goes out as binary frames. Clear is `{"action":"clear"}`. Playback acknowledgements use `{"action":"notify","payload":{"name":"..."}}`, which Vonage answers with a `websocket:notify` event. DTMF arrives as `websocket:dtmf`. There is no hang-up request and no `stop` event: when the caller hangs up, Vonage closes the socket and the session cancels the pipeline.

## Asterisk

**What Asterisk must send.** Use Asterisk's `chan_websocket` to connect a channel to `ws://YOUR_HOST:6969/ws`. The serializer accepts control messages in either plain-text form (`MEDIA_START connection_id:abc channel:PJSIP/100-0001 format:ulaw optimal_frame_size:160`) or JSON form (`{"event":"MEDIA_START","connection_id":"abc","channel":"PJSIP/100-0001","format":"ulaw"}`) and replies in the same form. Audio is binary in the channel's codec.

**Constructor and config.** The [Asterisk example](../../examples/asterisk/main.go) passes the channel id in the URL query:

```go
func asteriskSerializer(r *http.Request) serializers.FrameSerializer {
	return serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
		ChannelID: r.URL.Query().Get("channel"),
	})
}
```

| `AsteriskSerializerConfig` field | Meaning |
| --- | --- |
| `Codec` | Forces the codec (`ulaw`, `alaw`, `slin`, `slin16`, ...). Empty means `ulaw` until `MEDIA_START` reports the real format. |
| `SampleRate` | Forces the rate. Otherwise 8000 for G.711 and the number in `slinNN` formats. |
| `ChannelID` | Channel id if known in advance; `MEDIA_START` overrides it. |
| `DisableAutoHangUp` | Do not send `HANGUP` when the pipeline ends. |

**Call info.** `CallID` is the channel, `StreamID` the connection id, and `Encoding`/`SampleRate` the negotiated format. The serializer also exposes `ChannelID()`, `Codec()` and `SampleRate()`.

**Call control.** On `EndFrame` or `CancelFrame` the bot sends `HANGUP`. Clear is `FLUSH_MEDIA`. Playback acknowledgements use `MARK_MEDIA correlation_id:...`, answered by `MEDIA_MARK_PROCESSED`. `QUEUE_DRAINED`, which Asterisk sends once it has played everything it was sent, also ends the wait for the current mark. A `QUEUE_DRAINED` that arrives before the mark was sent is ignored. Asterisk's `MEDIA_XOFF` and `MEDIA_XON` pause and resume the bot's audio, so the output never overruns Asterisk's buffer. `DTMF_END digit:5` becomes a DTMF frame, and `HANGUP` from Asterisk ends the pipeline.

## Genesys AudioHook

**What Genesys must send.** Configure an AudioHook integration in Genesys Cloud that connects to `wss://YOUR_HOST/genesys`. Genesys sends an `open` message; the serializer answers with `opened` automatically, offering PCMU at the configured rate. It also answers `ping` with `pong` and `close` with `closed`.

> [!WARNING]
> AudioHook connections carry authentication headers (API key and request signature). Strawgo does not check them. Add a check in `ServerConfig.CheckOrigin` or an HTTP middleware; see [Deployment: security checklist](deployment.md#security-checklist).

**Constructor and config.**

```go
func genesysSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewGenesysAudioHookSerializer(serializers.GenesysConfig{
		Channel: serializers.GenesysChannelExternal,
		OnDTMF: func(msg map[string]any) {
			log.Printf("genesys dtmf: %v", msg["parameters"])
		},
	})
}
```

| `GenesysConfig` field | Default | Meaning |
| --- | --- | --- |
| `SampleRate` | 8000 | PCMU rate offered in `opened`. |
| `Channel` | `GenesysChannelExternal` | Which side to listen to; `GenesysChannelBoth` accepts stereo but the bot hears only the external (customer) channel. |
| `SupportedLanguages`, `SelectedLanguage` | none | Sent in `opened`. |
| `StartPaused` | false | Start paused; audio is ignored until Genesys resumes. |
| `OnOpen`, `OnClose`, `OnPaused`, `OnResumed`, `OnUpdate`, `OnError`, `OnDTMF` | none | Callbacks with the raw message. |

**Call info.** Genesys has no handshake step, so `s.CallInfo()` is empty. Read call data from the serializer after the `open` message has arrived:

```go
func genesysCallData(s *transports.Session) (string, map[string]any) {
	g, ok := s.Serializer().(*serializers.GenesysAudioHookSerializer)
	if !ok {
		return "", nil
	}
	return g.ConversationID(), g.InputVariables()
}
```

The serializer also has `CallInfo()` (with `From` from the participant's ANI and `To` from DNIS), `Participant()`, `SessionID()`, `SetOutputVariables`, and builders for protocol messages (`PauseRequest`, `ResumeRequest`, `DisconnectMessage`, `ErrorMessage`) that you can send with an `OutputTransportMessageUrgentFrame`.

**Call control.** On `EndFrame` the bot sends `disconnect` with reason `completed`, action `transfer`, and your output variables. An interruption sends a `barge_in` event. Audio is ignored while Genesys has the stream paused.

## Jambonz

**What Jambonz must send.** Use a `listen` verb that streams mono audio to `wss://YOUR_HOST/jambonz` with bidirectional audio enabled, for example:

```json
{
  "verb": "listen",
  "url": "wss://YOUR_HOST/jambonz",
  "mixType": "mono",
  "sampleRate": 8000,
  "bidirectionalAudio": {"enabled": true, "streaming": true, "sampleRate": 8000}
}
```

Jambonz first sends a JSON text preamble with call metadata (`callSid`, `traceId`, `from`, `to`, `sampleRate`, `mixType`), then binary 16-bit PCM.

**Constructor and config.**

```go
func jambonzSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewJambonzFrameSerializer(serializers.JambonzConfig{})
}
```

| `JambonzConfig` field | Meaning |
| --- | --- |
| `AudioInSampleRate`, `AudioOutSampleRate` | Fix the rates. Zero means use `sampleRate` from the preamble. |
| `DisableDisconnectEnd` | Do not send `{"type":"disconnect"}` when the pipeline ends. |

**Call info.** `CallID` is `callSid`, `StreamID` is `traceId`, and `From`/`To` come from the preamble. `CallMeta()`, `CallSID()`, `TraceID()` and `MixType()` on the serializer give you the rest.

**Call control.** Clear is `{"type":"killAudio"}`. DTMF arrives as `{"event":"dtmf","dtmf":"5"}`. If `mixType` is anything other than `mono`, caller audio is dropped with a warning, so set `"mixType": "mono"`.

## Generic protobuf

Use the protobuf serializer when your own client app (for example a Pipecat client SDK) connects over WebSocket instead of a phone provider.

**What the client must send.** Binary WebSocket messages in Pipecat's frame format: a wrapper message whose field 1 is a text frame, 2 an audio frame (16-bit PCM with sample rate and channel count), 3 a transcription, 4 a JSON message, and 5 an interruption.

**Constructor.**

```go
func protobufSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewProtobufFrameSerializer()
}
```

Incoming text and transcription messages become `TextFrame` and `TranscriptionFrame`. Outgoing, the session sends the bot's audio, an interruption message when the bot is interrupted, and any `OutputTransportMessageFrame` you push; it does not forward text or transcription frames to the client. There is no handshake and no call info; read query parameters or headers from `s.Request()` to identify the user. Audio from the client is resampled to the pipeline input rate. Bot audio is sent at `TransportParams.AudioOutSampleRate`, or the task's output rate (24000 by default) if that is unset. JSON messages (field 4) become `InputTransportMessageFrame`s, so an [RTVI](rtvi.md) client can talk to the bot over this serializer.

## Next steps

- [Twilio](telephony-twilio.md): the detailed walkthrough of every call-control feature.
- [Audio](audio.md): sample rates, native µ-law passthrough and DTMF tones.
- [Serializers reference](../reference/serializers.md): every interface and method.
- [Troubleshooting](troubleshooting.md)
