# Twilio

This guide connects Strawgo to Twilio Media Streams. You build a complete phone bot, answer inbound calls with TwiML, place outbound calls, read per-call data, handle keypad digits, and end calls cleanly.

## Before you begin

- A Twilio account, a Twilio phone number, your Account SID and Auth Token.
- API keys for Deepgram (STT), OpenAI (LLM) and Cartesia (TTS). Any other provider works the same way; see [Providers](../providers/README.md).
- A public `wss://` URL that reaches your machine, for example from a tunnel. [Your first phone bot](../getting-started/first-phone-bot.md) shows how to set one up.
- Read [Transports overview](transports-overview.md) for the server and session model.

Terms used on this page:

- **Media Stream**: Twilio's feature that sends a call's audio to your WebSocket server and plays audio you send back.
- **TwiML**: the XML Twilio reads to decide what to do with a call.
- **DTMF**: the tones a phone keypad makes. Strawgo turns them into frames.
- **Interruption** (barge-in): the caller talks over the bot, so the bot stops and listens.

## The complete program

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
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

const systemPrompt = `You are a friendly assistant on a phone call.
Answer in one or two short sentences. Never use lists, markdown or emojis.
When the caller says goodbye, call the end_call function.`

func main() {
	logger.Init()
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: ":8080",
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: runCall,
		Params: transports.TransportParams{
			SessionTimeout:   10 * time.Minute,
			PlaybackAck:      true,
			InterruptionFade: 20 * time.Millisecond,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := srv.ListenAndServe(ctx); err != nil {
		log.Fatal(err)
	}
}

func runCall(ctx context.Context, s *transports.Session) error {
	info := s.CallInfo()
	log.Printf("call %s started, stream %s, customer %q", info.CallID, info.StreamID, info.CustomParameters["customer_id"])

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: mustEnv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: mustEnv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: systemPrompt})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: mustEnv("CARTESIA_API_KEY"), VoiceID: mustEnv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})

	conversation := llm.NewContext()
	conversation.SetTools(llm.Tools{Functions: []llm.FunctionSchema{
		{Name: "end_call", Description: "Hang up after the caller has said goodbye.", Properties: map[string]any{}},
	}})
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(),
		stt,
		aggregators.NewDTMFAggregator(),
		pair.User(),
		llmSvc,
		tts,
		s.Output(),
		pair.Assistant(),
	})
	task := pipeline.NewPipelineTask(pipe)

	llmSvc.RegisterFunction("end_call", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		if err := task.QueueFrame(frames.NewTTSSpeakFrame("Thanks for calling. Goodbye!")); err != nil {
			return llm.FunctionResult{}, err
		}
		if err := task.QueueFrame(frames.NewEndFrame()); err != nil {
			return llm.FunctionResult{}, err
		}
		runLLM := false
		return llm.FunctionResult{Value: "call ended", RunLLM: &runLLM}, nil
	})

	task.OnFrameReachedDownstream(func(f frames.Frame) {
		log.Printf("call %s: caller pressed %s", info.CallID, f.(*frames.InputDTMFFrame).Button)
	}, (*frames.InputDTMFFrame)(nil))

	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hi! How can I help you today?")); err != nil {
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

Run it:

```bash
export TWILIO_ACCOUNT_SID=YOUR_TWILIO_ACCOUNT_SID
export TWILIO_AUTH_TOKEN=YOUR_TWILIO_AUTH_TOKEN
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export CARTESIA_API_KEY=YOUR_CARTESIA_API_KEY
export CARTESIA_VOICE_ID=YOUR_CARTESIA_VOICE_ID
go run .
```

The repository has a similar runnable program in [examples/twilio](../../examples/twilio/main.go), and one with tools in [examples/function_calling](../../examples/function_calling/main.go).

### How it works

1. `NewServer` accepts WebSocket connections on `/media`. Each one gets a new `TwilioFrameSerializer`.
2. Twilio sends a `connected` message and then a `start` message. The serializer reads `start` during the handshake and fills in `CallInfo`: `CallID` is the Call SID, `StreamID` is the Stream SID, `AccountID` is the Account SID, and `CustomParameters` holds your TwiML `<Parameter>` values.
3. `runCall` builds one pipeline for this call. `DTMFAggregator` collects keypad digits and passes them to the LLM as text (see [DTMF in](#dtmf-in)).
4. The greeting is queued before `Run`, so the bot speaks first.
5. The LLM can call `end_call`. The handler queues a goodbye and then an `EndFrame`. The `EndFrame` makes the output finish playing, then hang up the call through Twilio's REST API (see [Auto hang-up](#auto-hang-up)).
6. When the call ends for any reason, `task.Run` returns, the handler returns, and the server closes the connection.

## Answer inbound calls with TwiML

Configure your Twilio number's voice webhook to return TwiML that connects the call to your server. Use `<Connect><Stream>` so the stream is bidirectional; the bot can only talk back on a bidirectional stream.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://YOUR_HOST/media">
      <Parameter name="customer_id" value="12345"/>
    </Stream>
  </Connect>
</Response>
```

You can paste this into a TwiML Bin in the Twilio Console, or serve it from your own webhook. Serving it yourself lets you put per-call values (such as the caller's number from Twilio's webhook request) into `<Parameter>` elements.

## Place an outbound call

To have the bot call someone, ask Twilio's REST API to create a call whose TwiML connects to your stream:

```bash
curl -X POST "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Calls.json" \
  -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN" \
  --data-urlencode "To=+15551234567" \
  --data-urlencode "From=+15557654321" \
  --data-urlencode 'Twiml=<Response><Connect><Stream url="wss://YOUR_HOST/media"><Parameter name="customer_id" value="12345"/></Stream></Connect></Response>'
```

Replace `To` with the number to call and `From` with your Twilio number. When the callee answers, Twilio opens the stream and your handler runs exactly as for an inbound call. To detect voicemail on outbound calls, see [Voicemail detection](voicemail-detection.md).

## Read call information

`s.CallInfo()` returns what the `start` message carried:

| Field | Twilio value |
| --- | --- |
| `Provider` | `"twilio"` |
| `CallID` | Call SID (`CA...`) |
| `StreamID` | Stream SID (`MZ...`) |
| `AccountID` | Account SID (`AC...`) |
| `Encoding`, `SampleRate` | `mulaw`, `8000` (or what `mediaFormat` reported) |
| `CustomParameters` | Every `<Parameter name="..." value="..."/>` from your TwiML |
| `From`, `To` | Not set; Twilio's `start` message does not include them. Pass them as `<Parameter>` values if you need them. |

You can also read the raw upgrade request with `s.Request()`, for example a query string you added to the stream URL.

A real `start` message looks like this:

```json
{
  "event": "start",
  "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "start": {
    "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "callSid": "CAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "accountSid": "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "customParameters": {"customer_id": "12345"},
    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
  }
}
```

## DTMF in

When the caller presses a key, Twilio sends:

```json
{"event": "dtmf", "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", "dtmf": {"track": "inbound_track", "digit": "7"}}
```

The serializer turns this into an `InputDTMFFrame` with `Button` set to the key. You have two ways to use it:

- **Let the LLM read it.** Add `aggregators.NewDTMFAggregator()` before the user aggregator. It interrupts the bot on the first digit (when `AllowInterruptions` is `true`), collects digits until `#` or a 2-second pause, and then sends a final transcription such as `DTMF: 1234#`. Change the behaviour with `WithDTMFTimeout`, `WithDTMFTerminationDigit` and `WithDTMFPrefix`.
- **Handle it in Go.** Subscribe with `task.OnFrameReachedDownstream`, as the program above does. Pass `(*frames.InputDTMFFrame)(nil)` as the example; only its type matters.

## DTMF out

To press keys on the far end (for example to navigate a phone menu), queue an `OutputDTMFFrame`:

```go
func pressKeys(task *pipeline.PipelineTask) error {
	f, err := frames.NewOutputDTMFFrameFromString("1234#")
	if err != nil {
		return err
	}
	return task.QueueFrame(f)
}
```

The Twilio serializer has no message for sending digits, so Strawgo generates the tones itself (300 ms tone plus 200 ms gap per key, from `audio/dtmf`) and plays them as audio, in order with the bot's speech. `OutputDTMFUrgentFrame` plays the tones as soon as possible instead of waiting behind queued speech. See [IVR navigation](ivr-navigation.md) for a bot that does this from the LLM.

## Auto hang-up

When the pipeline ends, the serializer hangs up the call with Twilio's REST API (`POST .../Calls/{CallSid}.json` with `Status=completed`). This matters because with `<Connect><Stream>`, closing the WebSocket alone does not end the call; Twilio would continue with the next TwiML verb.

The hang-up needs credentials:

| `TwilioConfig` field | Meaning |
| --- | --- |
| `AccountSID` + `AuthToken` | Basic auth with your Auth Token. |
| `APIKeySID` + `APIKeySecret` | Basic auth with an API key instead of the Auth Token. |
| `Region`, `Edge` | Use a regional API host (`api.EDGE.REGION.twilio.com`); both must be set. |
| `BaseURL` | Override the API root entirely (useful for tests). |
| `DisableAutoHangUp` | Do not call the API; you hang up yourself. |
| `HTTPClient` | The client used for the request. |

The Call SID comes from the `start` message, so you do not need to set `CallSID` yourself. Without credentials, Strawgo skips the hang-up silently. A `404` with Twilio error `20404` (the call already ended) is treated as success.

To end the call from your code, queue an `EndFrame` (as `end_call` does above) or push an `EndTaskFrame` upstream from a processor. Both let queued speech finish first. `CancelFrame` ends immediately and also hangs up.

## Playback acknowledgements (marks)

The output sends audio at real-time speed, but Twilio still buffers a little before the caller hears it. To know exactly when the caller heard the last word, turn on `PlaybackAck`:

1. When the bot finishes a response, the output sends a mark:

   ```json
   {"event": "mark", "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", "mark": {"name": "strawgo-ws-1-1"}}
   ```

2. Twilio echoes the mark when playback reaches it:

   ```json
   {"event": "mark", "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", "mark": {"name": "strawgo-ws-1-1"}}
   ```

3. Only then does the output emit `BotStoppedSpeakingFrame`. If no echo arrives within `PlaybackAckTimeout` (default 1 s), it emits the frame anyway.

At the end of a call, the output always sends a final mark (if it played any audio) and waits for it before hanging up, so the goodbye is not cut off.

## Interruptions and "clear"

When the caller interrupts, an `InterruptionFrame` reaches the output. The output:

1. drops every audio chunk it has not sent yet,
2. sends Twilio a clear message so Twilio drops its own buffer:

   ```json
   {"event": "clear", "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"}
   ```

3. emits `BotStoppedSpeakingFrame`.

What counts as an interruption is decided by your turn-taking setup, not by the transport. See [Turn-taking](turn-taking.md).

### Soften the cut with `InterruptionFade`

A hard cut can click. Set `TransportParams.InterruptionFade` (for example `20 * time.Millisecond`) and the output fades the last bit of audio to silence before clearing. The fade works on µ-law directly, so it also applies to [native passthrough](audio.md#native-passthrough). Zero (the default) keeps the hard cut.

## Twilio troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| The call connects but the bot is silent and the log shows no `connected (provider=twilio ...)` line | Twilio cannot reach your URL, or the URL is `ws://` or `https://` | Use `wss://YOUR_HOST/media`; check the tunnel; look at the call's debugger events in the Twilio Console. |
| `handshake: ...` in the log | Twilio did not send `start` within `HandshakeTimeout`, or something else connected to the path | Check the TwiML; make sure only Twilio connects to that path. |
| The bot hears you, but you never hear the bot | `<Start><Stream>` (one-way) was used | Use `<Connect><Stream>`. |
| The call stays open after the bot says goodbye | No credentials for auto hang-up | Set `AccountSID` and `AuthToken` (or the API key pair) in `TwilioConfig`. Look for `hangup:` warnings in the log. |
| `hangup: twilio hangup CA...: status 401` | Wrong Auth Token or API key | Check the credentials; an API key must belong to the same account. |
| The bot's last sentence is cut off when the call ends | Hang-up happened before playback | Keep the default final mark; if you disabled marks with a custom serializer, set `AudioOutEndSilence`. |
| `CustomParameters` is empty | Parameters were put on the URL instead of `<Parameter>` elements | Use `<Parameter>` elements, or read the query string from `s.Request()`. |

The general [Troubleshooting](troubleshooting.md) page covers problems that are not specific to Twilio.

## Next steps

- [Other telephony providers](telephony-providers.md)
- [Function calling](function-calling.md): give the bot tools such as `end_call`.
- [Recording](recording.md): save the call audio and a transcript.
- [Deployment](deployment.md): run in production and validate Twilio's request signature.
