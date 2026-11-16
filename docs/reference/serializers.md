# Serializers reference

This page documents the `serializers` package: the `FrameSerializer` interface, the optional interfaces a serializer can add, and every built-in serializer with its config fields and wire behaviour.

A **serializer** translates between a carrier's WebSocket messages and frames. A **frame** is one typed message in the pipeline. A **carrier** is the telephony provider (Twilio, Telnyx and so on) that streams the call audio to your server. **G.711** is the 8 kHz telephone audio codec; it comes in two flavours, mu-law (PCMU) and A-law (PCMA). **DTMF** tones are keypad presses.

Import path: `github.com/square-key-labs/strawgo-ai/src/serializers`.

> [!NOTE]
> The serializers are unit-tested against message shapes in this repository's tests, and the Twilio, Telnyx and Plivo REST hang-ups and transfers against a mock server. None has been run against a live carrier account. See the [status table](../providers/README.md#framework-components).

## Quick example

You return a serializer from `ServerConfig.NewSerializer`, one per connection. Read credentials from the environment and pass request details through when the carrier sends them in the URL.

```go
func newSerializer(r *http.Request) serializers.FrameSerializer {
	switch r.URL.Query().Get("carrier") {
	case "telnyx":
		return serializers.NewTelnyxFrameSerializer(serializers.TelnyxConfig{APIKey: os.Getenv("TELNYX_API_KEY")})
	case "plivo":
		return serializers.NewPlivoFrameSerializer(serializers.PlivoConfig{
			AuthID:    os.Getenv("PLIVO_AUTH_ID"),
			AuthToken: os.Getenv("PLIVO_AUTH_TOKEN"),
		})
	default:
		return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
			AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
			AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
		})
	}
}
```

## Interfaces

```text
type FrameSerializer interface {
	SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error)
	Serialize(dst []Message, frame frames.Frame) ([]Message, error)
	Deserialize(msg Message) (frames.Frame, error)
}
```

| Method | Contract |
| --- | --- |
| `SerializeAudio` | Append the messages that carry 16-bit mono PCM `pcm` at `sampleRate`. The transport calls it for every outgoing chunk. |
| `Serialize` | Append the messages for a non-audio frame (interruption, transport message, end of call). Append nothing for frames the carrier has no message for. |
| `Deserialize` | Turn one incoming message into a frame. Return `nil, nil` to ignore it. |

`Message` is `{Type MessageType; Data []byte}` with `TextMessage` (1) and `BinaryMessage` (2); `Text(s)` and `Binary(b)` build one.

Optional interfaces the transport checks for:

| Interface | Methods | Transport behaviour |
| --- | --- | --- |
| `Handshaker` | `Handshake(msg Message) (done bool, err error)`, `CallInfo() CallInfo` | Messages are fed to `Handshake` before the session starts, until it returns `done`. `CallInfo` becomes `Session.CallInfo()`. |
| `WireFormat` | `WireSampleRates() (in, out int)` | The output transport sends audio at the `out` rate. |
| `NativeAudioSerializer` | `NativeAudioFormat() (codec string, sampleRate int)`, `SetNativeAudioInput(enabled bool)`, `SerializeNativeAudio(dst, data)` | Enables `AudioInNative` and `AudioOutNative` in `TransportParams`. |
| `PlaybackAckSerializer` | `SerializePlaybackAck(correlationID string) (Message, error)` | Used by `TransportParams.PlaybackAck` and at the end of a call. The carrier's reply becomes a `PlaybackCompleteFrame`. |
| `Hangupper` | `Hangup(ctx context.Context) error` | Called once after an `EndFrame` or `CancelFrame`, within `HangupTimeout`. |
| `Transferer` | `CanTransfer() bool`, `Transfer(ctx, TransferRequest) error` | Called when a `TransferCallFrame` reaches the output, after the speech queued before it, within `HangupTimeout`. On success the transport ends the pipeline and the later hang-up is skipped. See [Call control](../guides/call-control.md). |
| `DTMFSerializer` | `SerializeDTMF(dst []Message, buttons []frames.KeypadEntry) ([]Message, error)` | DTMF output as a wire message. No built-in serializer implements it. |
| `DTMFSender` | `CanSendDTMF() bool`, `SendDTMF(ctx, buttons) error` | DTMF output through the carrier's REST API. Without either DTMF interface, the transport plays in-band tones. |
| `LabeledMessage` | `MessageLabel() string` | Messages labelled `"rtvi-ai"` are RTVI messages; telephony serializers skip them. |

`CallInfo` fields: `Provider`, `StreamID`, `CallID`, `AccountID`, `From`, `To`, `Encoding` (all `string`), `SampleRate int`, `CustomParameters map[string]string`.

`MediaFlowFrame{Paused bool}` is a system frame the Asterisk serializer emits for flow control; the transport consumes it and pauses or resumes sending.

Shared behaviour of the telephony serializers below:

- Outgoing `OutputTransportMessageFrame` and `OutputTransportMessageUrgentFrame` values are sent as text (`string`, `json.RawMessage`), binary (`[]byte`) or JSON, except RTVI messages, which are skipped.
- A keypad event becomes an `InputDTMFFrame`; invalid digits are ignored.
- Outgoing audio is resampled to the wire rate and encoded to the wire codec.

Your handler reads what the handshake learned through the session. This reads a custom parameter the carrier passed in, for example from a Twilio `<Parameter name="customer_id" value="...">` tag:

```go
func customerID(s *transports.Session) string {
	info := s.CallInfo()
	log.Printf("%s call %s from %s", info.Provider, info.CallID, info.From)
	return info.CustomParameters["customer_id"]
}
```

## Summary

| Serializer | Constructor | Wire audio | Handshake | Clear on interruption | Playback ack | Hang-up | DTMF out | Transfer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Twilio | `NewTwilioFrameSerializer(TwilioConfig)` | mu-law 8 kHz, base64 JSON | `start` event | `clear` | `mark` | REST | in-band tones | REST (TwiML `<Dial>`) |
| Telnyx | `NewTelnyxFrameSerializer(TelnyxConfig)` | PCMU (or as configured) 8 kHz, base64 JSON | `start` event | `clear` | `mark` | REST | REST | REST |
| Plivo | `NewPlivoFrameSerializer(PlivoConfig)` | mu-law 8 kHz, base64 JSON | `start` event | `clearAudio` | `checkpoint` | REST | REST | REST, needs `TransferURL` |
| Exotel | `NewExotelFrameSerializer(ExotelConfig)` | 16-bit PCM, 8 kHz default, base64 JSON | `start` event | `clear` | `mark` | none | in-band tones | none (end the stream; the flow continues) |
| Vonage | `NewVonageFrameSerializer(VonageConfig)` | 16-bit PCM, 16 kHz default, binary | `websocket:connected` | `{"action":"clear"}` | `notify` | none | in-band tones | none |
| Asterisk | `NewAsteriskFrameSerializer(AsteriskSerializerConfig)` | ulaw 8 kHz default, binary | `MEDIA_START` | `FLUSH_MEDIA` | `MARK_MEDIA` | `HANGUP` command | in-band tones | none |
| Genesys | `NewGenesysAudioHookSerializer(GenesysConfig)` | PCMU 8 kHz, binary | `open` message | `barge_in` event | none | `disconnect` message | in-band tones | `transfer_to` output variable |
| Jambonz | `NewJambonzFrameSerializer(JambonzConfig)` | 16-bit PCM, rate from config or preamble, binary | first text message | `killAudio` | none | `disconnect` message | in-band tones | none |
| Protobuf | `NewProtobufFrameSerializer()` | 16-bit PCM in protobuf | none | `InterruptionFrame` message | none | none | none | none |
| RTVI client | `NewRTVIClientSerializer()` | 16-bit PCM base64 in RTVI `raw-audio` | none | none | none | none | none | none |

## Twilio

`TwilioConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `StreamSID` | `string` | from the `start` event | Media stream ID; every outgoing message carries it. |
| `CallSID` | `string` | from the `start` event | Call ID, used for hang-up. |
| `AccountSID` | `string` | from the `start` event | Account ID, used for hang-up. |
| `AuthToken` | `string` | none | Hang-up credential. |
| `APIKeySID`, `APIKeySecret` | `string` | none | Alternative hang-up credential, used when `AuthToken` is empty. |
| `Region`, `Edge` | `string` | none | When both are set, the REST host is `api.<edge>.<region>.twilio.com`. |
| `BaseURL` | `string` | `https://api.twilio.com` | Override the REST root. |
| `DisableAutoHangUp` | `bool` | false | Do not end the call through REST when the pipeline ends. |
| `HTTPClient` | `*http.Client` | `http.DefaultClient` | Client for REST calls. |

Wire behaviour:

| Direction | Message | Frame |
| --- | --- | --- |
| in | `start` | Handshake: fills IDs, `CustomParameters` from `<Parameter>` tags, and the media format. |
| in | `media` | `AudioFrame` (decoded to 16-bit PCM at 8 kHz, or mu-law with native input). |
| in | `dtmf` | `InputDTMFFrame`. |
| in | `mark` | `PlaybackCompleteFrame` with `correlation_id` = mark name. |
| in | `stop` | `EndFrame` (the transport treats it as the caller hanging up). |
| out | audio | `{"event":"media","streamSid":...,"media":{"payload":"<base64 mu-law>"}}` |
| out | `InterruptionFrame` | `{"event":"clear","streamSid":...}` |
| out | playback ack | `{"event":"mark","streamSid":...,"mark":{"name":...}}` |
| hang-up | `Hangup` | `POST /2010-04-01/Accounts/{AccountSID}/Calls/{CallSID}.json` with `Status=completed` and basic auth. Skipped without credentials. A call that already ended (error 20404) is not an error. |
| transfer | `Transfer` | Same URL, method and auth, with a `Twiml` parameter: `<Response>[<Say>announcement</Say>]<Dial [callerId="..."]><Number>` or `<Sip>` destination `</...></Dial></Response>`. Needs the same credentials as hang-up. |

This authenticates the REST hang-up with an API key instead of the account's Auth Token:

```go
func newTwilioSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
		AccountSID:   os.Getenv("TWILIO_ACCOUNT_SID"),
		APIKeySID:    os.Getenv("TWILIO_API_KEY_SID"),
		APIKeySecret: os.Getenv("TWILIO_API_KEY_SECRET"),
	})
}
```

## Telnyx

`TelnyxConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `StreamID` | `string` | from the `start` event | Stream ID. |
| `CallControlID` | `string` | from the `start` event | Used for hang-up and DTMF. |
| `APIKey` | `string` | none | Bearer token for REST calls. Without it there is no hang-up or REST DTMF. |
| `InboundEncoding` | `string` | `PCMU`, or the `start` event's encoding | Codec of caller audio. |
| `OutboundEncoding` | `string` | the inbound encoding | Codec of audio sent to Telnyx. |
| `BaseURL` | `string` | `https://api.telnyx.com` | REST root. |
| `DisableAutoHangUp` | `bool` | false | Skip the REST hang-up. |
| `HTTPClient` | `*http.Client` | `http.DefaultClient` | Client for REST calls. |

In: `start` (handshake; `From`, `To`, and `client_state` in `CustomParameters`), `media`, `dtmf`, `mark`, `stop`. Out: `{"event":"media","media":{"payload":...}}`, `{"event":"clear"}`, `{"event":"mark","mark":{"name":...}}`. Hang-up: `POST /v2/calls/{id}/actions/hangup`. Transfer: `POST /v2/calls/{id}/actions/transfer` with JSON `{"to":...,"from":...}` (`from` only when a caller ID is set; announcements are not sent). DTMF: `POST /v2/calls/{id}/actions/send_dtmf` with `duration_millis` 250.

This asks for A-law (PCMA) in both directions instead of the PCMU default:

```go
func newTelnyxSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewTelnyxFrameSerializer(serializers.TelnyxConfig{
		APIKey:           os.Getenv("TELNYX_API_KEY"),
		InboundEncoding:  "PCMA",
		OutboundEncoding: "PCMA",
	})
}
```

## Plivo

`PlivoConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `StreamID` | `string` | from the `start` event | Stream ID. |
| `CallID` | `string` | from the `start` event | Call UUID for hang-up and DTMF. |
| `AuthID`, `AuthToken` | `string` | none | REST credentials. Without both there is no hang-up or REST DTMF. |
| `BaseURL` | `string` | `https://api.plivo.com` | REST root. |
| `TransferURL` | `string` | none | Absolute URL of your own Plivo XML endpoint used for transfers. Without it there is no transfer. |
| `DisableAutoHangUp` | `bool` | false | Skip the REST hang-up. |
| `HTTPClient` | `*http.Client` | `http.DefaultClient` | Client for REST calls. |

In: `start`, `media`, `dtmf`, `playedStream` (becomes `PlaybackCompleteFrame`), `stop`. Out: `playAudio` with `contentType` `audio/x-mulaw` and `sampleRate` 8000, `clearAudio`, `checkpoint`. Hang-up: `DELETE /v1/Account/{AuthID}/Call/{CallID}/`. Transfer: `POST /v1/Account/{AuthID}/Call/{CallID}/` with JSON `{"legs":"aleg","aleg_url":...,"aleg_method":"GET"}`, where `aleg_url` is `TransferURL` plus `destination`, `caller_id` and `announcement` query parameters; your endpoint returns the `<Dial>` XML. DTMF: `POST /v1/Account/{AuthID}/Call/{CallID}/DTMF/`.

This keeps the call open when the pipeline ends, for example because your own code transfers the call afterwards:

```go
func newPlivoSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewPlivoFrameSerializer(serializers.PlivoConfig{
		AuthID:            os.Getenv("PLIVO_AUTH_ID"),
		AuthToken:         os.Getenv("PLIVO_AUTH_TOKEN"),
		DisableAutoHangUp: true,
	})
}
```

## Exotel

`ExotelConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `StreamSID` | `string` | from the `start` event | Stream ID. |
| `CallSID` | `string` | from the `start` event | Call ID. |
| `SampleRate` | `int` | 8000, or the `start` event's rate | Wire rate. Set it to pin the rate. |

In: `start`, `media`, `dtmf`, `mark`, `stop`. Out: `{"event":"media","stream_sid":...,"media":{"payload":<base64 PCM>}}`, `clear`, `mark`. No REST hang-up and no transfer: when the pipeline ends the stream closes and Exotel moves to the next applet in the call flow, so put the agent or number to reach in that applet and end the call with `callcontrol.EndCall`.

This pins the wire rate to 16 kHz instead of taking it from the `start` event:

```go
func newExotelSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewExotelFrameSerializer(serializers.ExotelConfig{SampleRate: 16000})
}
```

## Vonage

`VonageConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `SampleRate` | `int` | 16000, or the rate in the `websocket:connected` content type | Wire rate. Set it to pin the rate. |

Audio both ways is raw 16-bit PCM in binary messages. The `websocket:connected` handshake's other string fields become `CustomParameters`; `CallID` is the first of `call_uuid`, `uuid`, `conversation_uuid`. In: `websocket:dtmf`, `websocket:notify` (playback ack). Out: `{"action":"clear"}`, `{"action":"notify","payload":{"name":...}}`. No REST hang-up.

The zero config takes the rate from the `websocket:connected` message, with 16 kHz as the fallback:

```go
func newVonageSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewVonageFrameSerializer(serializers.VonageConfig{})
}
```

## Asterisk

For Asterisk's `chan_websocket` media channel.

`AsteriskSerializerConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ChannelID` | `string` | from `MEDIA_START` | Channel ID. |
| `Codec` | `string` | `ulaw`, or the `MEDIA_START` format | Wire codec. `slin` means 16-bit PCM at 8 kHz; `slin16`, `slin44` and so on set the rate. When set, the format in `MEDIA_START` is ignored. |
| `SampleRate` | `int` | from the codec (8000 for ulaw/alaw, 16000 for linear16) | Override the wire rate. |
| `DisableAutoHangUp` | `bool` | false | Do not send `HANGUP` when the pipeline ends. |

Audio both ways is binary. Control messages are plain-text commands, or JSON when `MEDIA_START` arrived as JSON. In: `MEDIA_START` (handshake), `HANGUP` (`EndFrame`), `DTMF_END`, `MEDIA_XOFF` and `MEDIA_XON` (pause and resume sending), `MEDIA_MARK_PROCESSED`, `QUEUE_DRAINED` (both `PlaybackCompleteFrame`). Out: `FLUSH_MEDIA` on interruption, `MARK_MEDIA correlation_id:<id>` for acks, `HANGUP` on `EndFrame` or `CancelFrame`. Extra methods: `ChannelID()`, `Codec()`, `SampleRate()`.

This forces 16-bit PCM at 16 kHz, whatever `MEDIA_START` announces:

```go
func newAsteriskSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{Codec: "slin16"})
}
```

## Genesys AudioHook

Implements the Genesys Cloud AudioHook protocol, version `"2"` (`GenesysProtocolVersion`).

`GenesysConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `SampleRate` | `int` | 8000 | Wire rate. |
| `Channel` | `GenesysChannel` | `GenesysChannelExternal` | Which side to listen to: `external` (the customer), `internal` (the agent) or `both` (stereo). |
| `SupportedLanguages` | `[]string` | none | Languages offered in the `opened` reply. |
| `SelectedLanguage` | `string` | none | Language selected in the `opened` reply. |
| `StartPaused` | `bool` | false | Open the session paused. |
| `OnOpen`, `OnClose`, `OnPaused`, `OnResumed`, `OnUpdate`, `OnError`, `OnDTMF` | `func(msg map[string]any)` | none | Called with the raw protocol message. |

In: `open` and `close` (answered automatically with `opened` and `closed` through `OutputTransportMessageUrgentFrame`), `ping` (answered with `pong`), `paused`, `resumed`, `update`, `error`, `dtmf`, binary audio. Out: audio only while open and not paused; `barge_in` event on interruption; a `disconnect` message (reason `completed`, action `transfer`, with output variables) on `EndFrame` or `CancelFrame`. Transfer: `Transfer` cannot dial; it adds the destination as the `transfer_to` output variable (`GenesysTransferVariable`) of that `disconnect` message, and your Architect flow routes on it. Transport messages pass through only if they are protocol messages (they have a `version` key). Helper methods: `SessionID`, `ConversationID`, `Participant`, `InputVariables`, `OutputVariables`, `SetOutputVariables`, `IsOpen`, `IsPaused`, `PauseRequest`, `ResumeRequest`, `BargeInEvent`, `DisconnectMessage`, `ErrorMessage`.

This listens to the customer side and logs the raw `open` message, which carries the conversation details:

```go
func newGenesysSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewGenesysAudioHookSerializer(serializers.GenesysConfig{
		Channel: serializers.GenesysChannelExternal,
		OnOpen: func(msg map[string]any) {
			log.Printf("audiohook open: %v", msg)
		},
	})
}
```

## Jambonz

For the jambonz `listen` verb.

`JambonzConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AudioInSampleRate` | `int` | `sampleRate` from the preamble | Caller audio rate. |
| `AudioOutSampleRate` | `int` | `sampleRate` from the preamble | Rate of audio sent back. |
| `DisableDisconnectEnd` | `bool` | false | Do not send `{"type":"disconnect"}` when the pipeline ends. |

The first text message is the preamble (handshake); `callSid`, `traceId` and `mixType` are read from it (`CallSID()`, `TraceID()`, `MixType()`, `CallMeta()`). Audio is binary 16-bit PCM both ways. Caller audio is dropped with a warning unless `mixType` is `mono`. In: `dtmf` events. Out: `{"type":"killAudio"}` on interruption, `{"type":"disconnect"}` once on `EndFrame` or `CancelFrame`.

The zero config takes both rates from the preamble and sends `disconnect` when the pipeline ends:

```go
func newJambonzSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewJambonzFrameSerializer(serializers.JambonzConfig{})
}
```

## Protobuf

`NewProtobufFrameSerializer()` exchanges binary protobuf messages with a custom client. Each message is a wrapper with one field set:

| Field | Number | Content | Direction |
| --- | --- | --- | --- |
| text | 1 | `id` (1), `name` (2), `text` (3) | out from `TextFrame`; in as `TextFrame` |
| audio | 2 | `id` (1), `name` (2), `audio` (3), `sample_rate` (4), `num_channels` (5) | both |
| transcription | 3 | `id`, `name`, `text` (3), `user_id` (4), `timestamp` (5, RFC 3339) | out from `TranscriptionFrame`; in as a final `TranscriptionFrame` |
| message | 4 | `data` (1, JSON) | out from transport messages; in as `InputTransportMessageFrame` |
| interruption | 5 | `id`, `name` | both |

This serves Pipecat-style protobuf clients on their own path; the serializer takes no config:

```go
func protobufServerConfig(handler transports.SessionHandler) transports.ServerConfig {
	return transports.ServerConfig{
		Addr: ":8080",
		Path: "/ws",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewProtobufFrameSerializer()
		},
		Handler: handler,
	}
}
```

## RTVI client

`NewRTVIClientSerializer()` is the client side of the RTVI protocol, for a Strawgo process that talks to an RTVI server. It sends audio as RTVI `raw-audio` messages and forwards only RTVI-labelled transport messages. Incoming RTVI events become frames: `bot-llm-started`, `bot-llm-text`, `bot-llm-stopped`, `bot-llm-marker`, `bot-tts-text`, `bot-started-speaking`, `bot-stopped-speaking`, `bot-interrupted`, `llm-function-call-in-progress`, `llm-function-call-stopped` and `user-transcription`. See the [RTVI guide](../guides/rtvi.md) for the server side.

This returns the RTVI client serializer for one connection; like the protobuf serializer it takes no config:

```go
func newRTVISerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewRTVIClientSerializer()
}
```

## Next steps

- [Transports reference](transports.md): the session that drives a serializer.
- [Telephony providers guide](../guides/telephony-providers.md): carrier setup step by step.
- [Twilio guide](../guides/telephony-twilio.md): TwiML, marks and hang-up.
- [Frames reference](frames.md): the frames serializers produce.
