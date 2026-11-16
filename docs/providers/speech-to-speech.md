# Speech-to-speech providers

This page shows how to configure the two speech-to-speech integrations, OpenAI Realtime and Gemini Live: their config fields and defaults, a complete phone bot, and what you can change while a call is running. For the concepts and trade-offs, read the [speech-to-speech guide](../guides/speech-to-speech.md).

> [!NOTE]
> Neither integration has been verified against the live vendor API; both are tested against mock websocket servers. AWS Nova Sonic has no integration in this repository.

A speech-to-speech service replaces the usual STT (speech-to-text), LLM and TTS (text-to-speech) trio with one model that takes the caller's audio and returns the bot's audio. It is a single pipeline **processor** that keeps one websocket open for the call. The vendor runs its own **VAD** (voice activity detection, the part that hears when someone starts and stops talking) unless you switch it off.

## Choosing a provider

| If you need | Use | Why |
| --- | --- | --- |
| Tool choice control (`auto`, `none`, a named function) | OpenAI Realtime | `ToolChoice` and `LLMSetToolChoiceFrame` are handled |
| Tools changed mid-call without reconnecting | OpenAI Realtime | it sends a `session.update`; Gemini Live reconnects |
| Transcripts of the caller from the vendor | OpenAI Realtime | `TranscriptionModel` turns on input transcription |
| Non-PCM input sent as-is | Gemini Live | `InputMIMEType` or the codec of the incoming audio is passed in the MIME type |
| Your own turn detection instead of the vendor's | either | `DisableServerVAD` (OpenAI) or `DisableVAD` (Gemini) |

## Complete example

This program answers Twilio calls with OpenAI Realtime. It registers one function the model can call.

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
	"github.com/square-key-labs/strawgo-ai/src/services/openai_realtime"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

var tools = llm.Tools{Functions: []llm.FunctionSchema{{
	Name:        "get_time",
	Description: "Current local time of the service.",
	Properties:  map[string]any{},
}}}

func main() {
	logger.Init()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required")
	}
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: ":8080",
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: func(ctx context.Context, s *transports.Session) error { return runCall(ctx, s, apiKey) },
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

func runCall(ctx context.Context, s *transports.Session, apiKey string) error {
	rt := openai_realtime.NewLLMService(openai_realtime.LLMConfig{
		APIKey:            apiKey,
		SystemInstruction: "You are a friendly phone assistant. Keep answers short.",
		Voice:             "alloy",
		Tools:             tools,
	})
	rt.RegisterFunction("get_time", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: map[string]any{"time": time.Now().Format(time.Kitchen)}}, nil
	})

	conversation := llm.NewContext(llm.User("Greet the caller and ask how you can help."))
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), pair.User(), rt, s.Output(), pair.Assistant(),
	}))
	if err := task.QueueFrame(frames.NewLLMRunFrame()); err != nil {
		return err
	}
	return task.Run(ctx)
}
```

Run it and point a Twilio number at `wss://YOUR_HOST/media` as described in [Your first phone bot](../getting-started/first-phone-bot.md):

```sh
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
go run .
```

How it works:

1. There is no STT or TTS in the pipeline. Audio from `s.Input()` goes straight into the realtime service, and its audio goes to `s.Output()`.
2. The context aggregators still record the conversation, so you can read `conversation.Messages()` after the call.
3. `NewLLMRunFrame` sends the context to the model, which speaks first using the seed user message.
4. `RegisterFunction` works the same way as on a normal LLM service. The schema in `Tools` tells the model the function exists.

## OpenAI Realtime

`openai_realtime.NewLLMService(cfg)` or the shortcut `NewOpenAIRealtimeLLMService(apiKey, model)`. Conventional key variable: `OPENAI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `gpt-realtime-2.1` | Sent as the `model` query parameter of the websocket URL and in the session. |
| `SystemInstruction` | `string` | none | Session instructions. |
| `Voice` | `string` | none | Sent as `audio.output.voice`. |
| `Speed` | `float64` | unset | Speaking speed, sent as `audio.output.speed`. |
| `Tools` | `llm.Tools` | none | Function schemas offered to the model. |
| `ToolChoice` | `llm.ToolChoice` | unset | |
| `Endpoint` | `string` | `wss://api.openai.com/v1/realtime` | |
| `InputAudioFormat`, `OutputAudioFormat` | `string` | `pcm16` | `pcm16` (or `audio/pcm`), `g711_ulaw` (or `audio/pcmu`), `g711_alaw` (or `audio/pcma`). Sent as format objects such as `{"type": "audio/pcm", "rate": 24000}`. |
| `OutputSampleRate` | `int` | 24000 for PCM, otherwise 8000 | |
| `OutputChannels` | `int` | 1 | |
| `TranscriptionModel` | `string` | `gpt-realtime-whisper` | Model used to transcribe the caller. |
| `TranscriptionLanguage` | `string` | unset | ISO-639-1 code of the caller's language. |
| `TranscriptionPrompt` | `string` | unset | Hint text for transcription. Not sent with `gpt-realtime-whisper`, which does not accept it. |
| `TurnDetection` | `*openai_realtime.TurnDetection` | unset (the server's default, `server_vad`) | `Type` is `TurnDetectionServerVAD` (with `Threshold`, `PrefixPaddingMs`, `SilenceDurationMs`) or `TurnDetectionSemanticVAD` (with `Eagerness`, `CreateResponse`, `InterruptResponse`). Zero and nil values are left out. |
| `DisableServerVAD` | `bool` | false | Sends `turn_detection: null` and overrides `TurnDetection`. The pipeline's `UserStoppedSpeakingFrame` then commits the audio buffer and asks for a response. |
| `NoiseReduction` | `string` | unset | `near_field` or `far_field`. |
| `ReasoningEffort` | `string` | unset | `minimal`, `low`, `medium`, `high` or `xhigh`. Only sent when the model name contains `gpt-realtime-2`; other models reject it. |
| `Truncation` | `string` | unset (the server's `auto`) | `auto` or `disabled`. |
| `TruncationRetentionRatio` | `float64` | 0 | When above 0, sends a `retention_ratio` truncation that keeps this fraction of the token limit after truncating, so truncation (and the prompt cache reset it causes) happens less often. Takes precedence over `Truncation`. |
| `FunctionTimeout` | `time.Duration` | 0 | Per-call timeout for registered functions. |
| `Backoff` | `realtime.Backoff` | 500 ms initial, 15 s max, 6 attempts | Reconnect policy. |
| `Dialer` | `*websocket.Dialer` | `websocket.DefaultDialer` | |

The service speaks the GA Realtime API: it connects with only the `Authorization: Bearer` header (no `OpenAI-Beta` header), sends a `session.update` with `"type": "realtime"` and the audio settings nested under `audio.input` and `audio.output`, and reads the GA server events `response.output_audio.delta`, `response.output_audio_transcript.delta`, `response.output_text.delta` and `conversation.item.added`. The beta event names (`response.audio.delta` and the others) are ignored.

What it handles at runtime:

| Frame | Effect |
| --- | --- |
| `LLMUpdateSettingsFrame` | Only `SystemInstruction` is applied; it is sent in a `session.update`. |
| `LLMSetToolsFrame` | New tools are sent in a `session.update`. |
| `LLMSetToolChoiceFrame` | New tool choice is sent in a `session.update`. |
| `LLMMessagesAppendFrame` | Messages are added as conversation items; `RunLLM` asks for a response. |
| `InterruptionFrame` | The current response is cancelled and pending interruptible function calls are cancelled. |

## Gemini Live

`gemini.NewLiveService(cfg)` or the shortcut `NewGeminiLiveService(apiKey, model)`. Conventional key variable: `GEMINI_API_KEY`. The key is sent in the `x-goog-api-key` header of the websocket handshake.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `gemini-3.8-live` | Sent as `models/{Model}` in the setup message. |
| `SystemInstruction` | `string` | none | |
| `Voice` | `string` | none | Prebuilt voice name, for example `Puck`. |
| `Endpoint` | `string` | the `v1beta` `BidiGenerateContent` websocket URL | |
| `InputMIMEType` | `string` | `audio/pcm;rate={rate}` (with `encoding=` for non-PCM input) | |
| `OutputSampleRate` | `int` | 24000 | |
| `OutputChannels` | `int` | 1 | |
| `DisableVAD` | `bool` | false | When true, the pipeline's user speaking frames are sent as `activityStart` and `activityEnd`. |
| `Tools` | `llm.Tools` | none | On Gemini 3.8 and later Live models (except thinking models) each function is declared `BLOCKING`, so the model waits for the result. |
| `ContextWindowCompression` | `bool` | false | Sends `contextWindowCompression` with a sliding window, so long sessions are compressed instead of ending. |
| `CompressionTriggerTokens` | `int` | 0 (the server's default, 80% of the context window) | Token count at which compression starts. |
| `FunctionTimeout` | `time.Duration` | 0 | |
| `Backoff` | `realtime.Backoff` | 500 ms initial, 15 s max, 6 attempts | |
| `Dialer` | `*websocket.Dialer` | `websocket.DefaultDialer` | |

```go
func newGeminiLive() *gemini.LiveService {
	return gemini.NewLiveService(gemini.LiveConfig{
		APIKey:            os.Getenv("GEMINI_API_KEY"),
		Model:             gemini.DefaultLiveModel,
		SystemInstruction: "You are a helpful voice assistant on a phone call.",
		Voice:             "Puck",
	})
}
```

The setup message always includes `historyConfig.initialHistoryInClientContent`, so the conversation in the first context is sent as client content. With a Gemini 3 model, every client content message that completes a turn is followed by a realtime text input of a single space, because Gemini 3 does not start a response without realtime input.

What it handles at runtime:

| Frame | Effect |
| --- | --- |
| `LLMUpdateSettingsFrame` | Only `SystemInstruction` is applied, through `UpdateSystemInstruction`. |
| `LLMSetToolsFrame` | Tools are replaced and the session reconnects. |
| `LLMMessagesAppendFrame` | Messages are sent as client content; `RunLLM` asks for a turn. |
| `InterruptionFrame` | Output stops and interruptible function calls are cancelled. |

## Change instructions during a call

Model, voice and audio formats are fixed when the service connects. You can change the instructions:

```go
func beStricter(task *pipeline.PipelineTask) error {
	return task.QueueFrame(frames.NewLLMUpdateSettingsFrame(llm.Settings{
		SystemInstruction: "Only answer questions about opening hours.",
	}))
}
```

## Next steps

- [Speech-to-speech guide](../guides/speech-to-speech.md): when to choose speech-to-speech over STT, LLM and TTS.
- [Function calling](../guides/function-calling.md): tools work the same way here.
- [Telephony with Twilio](../guides/telephony-twilio.md): the transport used above.
- [Provider matrix](README.md): every integration at a glance.
