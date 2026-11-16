# STT providers

This page shows how to configure every STT (speech-to-text) integration: the config fields and their defaults, a constructor example, the quirks the code handles, and how to change the model or language while a call is running.

> [!NOTE]
> No STT integration has been verified against the live vendor API. See the status column in the [provider matrix](README.md#stt).

An STT service is a pipeline **processor** that receives the caller's audio as `AudioFrame`s and pushes `TranscriptionFrame`s (final text) and `InterimTranscriptionFrame`s (partial text). A **turn** is one stretch of the caller speaking; **VAD** (voice activity detection) decides when a turn starts and stops. Some STT vendors detect the end of a turn on their side, which this page calls server-side turn detection.

## Choosing a provider

| If you need | Use | Why |
| --- | --- | --- |
| Partial transcripts while the caller is still talking | a websocket provider (Deepgram, AssemblyAI, Azure, Cartesia, ElevenLabs Realtime, Gladia, OpenAI Realtime, Sarvam, Sarvam Realtime, Soniox, AWS Transcribe) | they stream audio continuously and report interim results |
| Turn ends decided by the STT vendor | `deepgram.NewFluxSTTService`, `soniox` with `DisableVADForceTurnEndpoint`, `gladia` with `EnableVAD`, `openai.NewRealtimeSTTService` with `TurnDetection` | these push their own user-turn strategies into the pipeline |
| One request per utterance, no open socket | a segmented provider (Whisper, OpenAI, Groq, ElevenLabs Scribe, fal, Google) | the base buffers the turn and sends one WAV file |
| Telephony audio without resampling | `deepgram` with `Encoding: "mulaw"` | the service accepts 8 kHz mu-law directly |
| AWS IAM or Google service-account credentials | `aws/transcribe`, `google` | they use the vendor SDK credential chains |

Two kinds of STT exist in the code:

- **Streaming** services keep one websocket open for the whole call (`stt.NewStreaming`). Most reconnect automatically when the socket drops.
- **Segmented** services buffer audio between "user started speaking" and "user stopped speaking", then send one HTTP request with a WAV file (`stt.NewSegmented`). They need a VAD or another source of speaking frames in the pipeline.

## Complete example

This program answers Twilio phone calls with Deepgram STT, an OpenAI LLM and Cartesia TTS. Swap the `stt :=` line for any constructor on this page.

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

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
	stt := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   os.Getenv("DEEPGRAM_API_KEY"),
		Model:    "nova-3",
		Language: "en",
	})
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a friendly phone assistant. Answer in one or two short sentences.",
	})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: os.Getenv("CARTESIA_VOICE_ID"),
	})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	}))
	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hello! How can I help?")); err != nil {
		return err
	}
	return task.Run(ctx)
}
```

Run it with the keys in your environment, then point a Twilio number at `wss://YOUR_HOST/media` as described in [Your first phone bot](../getting-started/first-phone-bot.md).

```sh
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export CARTESIA_API_KEY=YOUR_CARTESIA_API_KEY
export CARTESIA_VOICE_ID=YOUR_CARTESIA_VOICE_ID
go run .
```

The STT service sits right after `s.Input()`, so it sees the caller's audio first. Its transcripts flow into the user context aggregator, which decides when the turn is complete and asks the LLM for a reply.

## Options every STT service shares

Every service on this page except the standalone OpenAI Realtime STT embeds `*stt.Base`, which provides:

- `SampleRate()`: the rate the service sends to the vendor. It is the config's `SampleRate` if set, otherwise the pipeline's `AudioInSampleRate` (16000 by default), otherwise 16000.
- `Settings()` and `UpdateSettings(delta)`: the current `frames.STTSettings{Model, Language, Extra}`.
- Handling of `STTMuteFrame` (stop sending audio while muted) and `STTUpdateSettingsFrame`.
- TTFB metrics (time from the end of the caller's speech to the final transcript) when metrics are enabled on the task.

The library never reads environment variables for keys. The variable names below are the convention the examples use.

## AssemblyAI

Universal streaming over a websocket. Conventional key variable: `ASSEMBLYAI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent in the `Authorization` header. |
| `Model` | `string` | `u3-rt-pro` | Sent as `speech_model`. |
| `Language` | `string` | none | Sent as `language_code` when set. |
| `SampleRate` | `int` | pipeline input rate | |
| `MinTurnSilence`, `MaxTurnSilence` | `int` | 0 (not sent) | Sent as `min_turn_silence` and `max_turn_silence`. |
| `DisableFormatTurns` | `bool` | false | `format_turns` is true unless set. |
| `DisableVADEndpoint` | `bool` | false | When false, the service asks AssemblyAI to finalize the turn when the pipeline's VAD says the user stopped. |
| `KeytermsPrompt` | `[]string` | none | Sent as `keyterms_prompt`. |
| `BaseURL` | `string` | `wss://streaming.assemblyai.com/v3/ws` | |
| `OnEndOfTurn` | `func(string)` | nil | Called with the transcript when AssemblyAI reports end of turn. |

```go
func newAssemblyAI() *assemblyai.STTService {
	return assemblyai.NewSTTService(assemblyai.STTConfig{APIKey: os.Getenv("ASSEMBLYAI_API_KEY")})
}
```

Runtime changes reconnect the websocket with the new model and language.

## AWS Transcribe

Streaming transcription through the AWS SDK. `NewSTTService(ctx, cfg)` loads AWS config and returns an error; `NewSTTServiceWithClient` takes your own `*transcribestreaming.Client`. Credentials come from `cfg.AWS` or the AWS SDK default chain (environment, shared config, instance role).

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AWS` | `awsauth.Config` | zero | Region (default `us-east-1`), static keys, profile, endpoint. |
| `Language` | `string` | `en-US` when `LanguageOptions` is empty | Language code. |
| `SampleRate` | `int` | pipeline input rate | |
| `EnablePartialResultsStabilization`, `PartialResultsStability` | `bool`, `string` | off | Partial result stabilization. |
| `ShowSpeakerLabel` | `bool` | false | Speaker labels. |
| `LanguageOptions`, `IdentifyMultipleLanguages`, `PreferredLanguage` | | none | Automatic language identification. |
| `VocabularyName`, `VocabularyFilterName`, `VocabularyFilterMethod`, `LanguageModelName` | `string` | none | Custom vocabularies and models. |
| `BufferDuration` | `time.Duration` | 2 s | Audio buffered while the stream (re)starts. |
| `CloseTimeout` | `time.Duration` | 5 s | Wait for the stream to finish at shutdown. |

```go
func newTranscribe(ctx context.Context) (*transcribe.STTService, error) {
	return transcribe.NewSTTService(ctx, transcribe.STTConfig{
		AWS:      awsauth.Config{Region: "us-east-1"},
		Language: "en-US",
	})
}
```

Runtime changes restart the transcription stream.

## Azure Speech

Azure Speech-to-Text over a websocket. Conventional key variable: `AZURE_SPEECH_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `SubscriptionKey` | `string` | none | Azure Speech resource key. |
| `Region` | `string` | `eastus` | Used to build the endpoint. |
| `Language` | `string` | `en-US` | |
| `SampleRate` | `int` | pipeline input rate | |
| `Endpoint` | `string` | `wss://{Region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1` | |

```go
func newAzureSTT() *azure.STTService {
	return azure.NewSTTService(azure.STTConfig{
		SubscriptionKey: os.Getenv("AZURE_SPEECH_KEY"),
		Region:          "westeurope",
	})
}
```

Runtime changes reconnect the websocket.

## Cartesia

Cartesia streaming STT over a websocket. Conventional key variable: `CARTESIA_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `BaseURL` | `string` | host `api.cartesia.ai`, scheme `wss` | Host or URL; only scheme and host are used. |
| `Model` | `string` | `ink-whisper` | |
| `Language` | `string` | `en` | |
| `Encoding` | `string` | `pcm_s16le` | |
| `SampleRate` | `int` | pipeline input rate | |
| `Keyterms` | `[]string` | none | Stored in `Extra["keyterm"]`. |
| `MinVolume`, `MaxSilenceDurationSecs` | `*float64` | nil | Stored in `Extra`. |
| `ExtraHeaders` | `http.Header` | none | Added to the websocket handshake. |

```go
func newCartesiaSTT() *cartesia.STTService {
	return cartesia.NewSTTService(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})
}
```

The service asks Cartesia to finalize when the user stops speaking. Runtime changes reconnect the websocket.

## Deepgram

Deepgram live transcription over a websocket. Conventional key variable: `DEEPGRAM_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent as `Authorization: Token <key>`. |
| `Model` | `string` | none | Sent as `model` when set; otherwise Deepgram's default applies. |
| `Language` | `string` | none | Sent as `language` when set. |
| `Encoding` | `string` | `linear16` | `mulaw` and `alaw` switch to 1 byte per sample and 8000 Hz unless `SampleRate` is set. |
| `SampleRate` | `int` | pipeline input rate | |
| `KeepaliveInterval` | `time.Duration` | 5 s | A `KeepAlive` message is sent after this much idle time. |
| `SmartFormat` | `bool` | false | Sends `smart_format=true`. |
| `BaseURL` | `string` | `wss://api.deepgram.com/v1/listen` | |

```go
func newDeepgram() *deepgram.STTService {
	return deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:      os.Getenv("DEEPGRAM_API_KEY"),
		Model:       "nova-3",
		Language:    "en",
		SmartFormat: true,
	})
}
```

Interim results are always requested. Runtime changes reconnect the websocket with the new query string.

## Deepgram Flux

Deepgram's turn-taking model (`/v2/listen`). Flux decides when the caller's turn ends, so the service pushes its own user-turn strategies into the pipeline when it connects. Conventional key variable: `DEEPGRAM_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `flux-general-en` | `flux-general-multi` (`FluxMultilingualModel`) for other languages. |
| `SampleRate` | `int` | pipeline input rate | |
| `BaseURL` | `string` | `wss://api.deepgram.com/v2/listen` | |
| `EagerEOTThreshold`, `EOTThreshold` | `*float64` | nil | End-of-turn confidence thresholds. `EagerEOTThreshold` defaults to 0.5 when `EnableEagerEndOfTurn` is set. |
| `EOTTimeoutMs` | `*int` | nil | |
| `Keyterms`, `LanguageHints` | `[]string` | none | |
| `MinConfidence` | `*float64` | nil | |
| `Numerals`, `ProfanityFilter`, `RedactUsage`, `MIPOptOut` | `*bool` | nil | |
| `Redact` | `string` | none | |
| `Tags` | `[]string` | none | |
| `DisableInterruptions` | `bool` | false | When true, the caller starting to speak does not interrupt the bot. |
| `EnableEagerEndOfTurn` | `bool` | false | Answer Flux's eager end of turn speculatively: the LLM starts on the eager transcript and its reply is held until the turn is confirmed. Pushes `EagerTranscriptionFrame` and, on `TurnResumed`, `UserTurnResponseCancelFrame`, and recommends `FluxEagerUserTurnStrategies`. See [Turn-taking: speculative replies](../guides/turn-taking.md#speculative-replies-eager-end-of-turn). |
| `EagerMatchPolicy` | `user_stop.EagerMatchPolicy` | `user_stop.NormalizedMatch{}` | With `EnableEagerEndOfTurn`, decides whether the final transcript still matches the eager one. `user_stop.ExactMatch{}` requires identical text. |
| `WatchdogMinTimeout` | `time.Duration` | 500 ms | |
| `OnStartOfTurn`, `OnTurnResumed`, `OnEndOfTurn`, `OnEagerEndOfTurn`, `OnUpdate` | callbacks | nil | Called on Flux events. |

```go
func newFlux() *deepgram.FluxSTTService {
	return deepgram.NewFluxSTTService(deepgram.FluxConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
}
```

`deepgram.FluxUserTurnStrategies(enableInterruptions)` returns the same strategies if you want to set them on the task yourself, and `deepgram.FluxEagerUserTurnStrategies(enableInterruptions, policy)` the ones recommended with `EnableEagerEndOfTurn`. Runtime changes to keyterms, thresholds, language hints, numerals, profanity filter or redact-usage are sent as a configure message without reconnecting. Changing the model or `redact` reconnects, and the reconnect waits until the caller stops speaking.

## ElevenLabs Scribe

Per-utterance transcription over HTTP (segmented). Conventional key variable: `ELEVENLABS_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `scribe_v2` | |
| `Language` | `string` | `eng` | |
| `TagAudioEvents`, `NoVerbatim`, `Diarize` | `*bool` | nil | |
| `Keyterms` | `[]string` | none | |
| `SampleRate` | `int` | pipeline input rate | |
| `BaseURL` | `string` | `https://api.elevenlabs.io` | |
| `HTTPClient` | `*http.Client` | 30 s timeout | |

```go
func newScribe() *elevenlabs.STTService {
	return elevenlabs.NewSTTService(elevenlabs.STTConfig{APIKey: os.Getenv("ELEVENLABS_API_KEY")})
}
```

## ElevenLabs Scribe Realtime

Streaming transcription over a websocket.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `scribe_v2_realtime` | |
| `Language` | `string` | none | |
| `CommitStrategy` | `string` | `manual` | `manual` (the pipeline's end of speech commits) or `vad` (`elevenlabs.CommitVAD`). |
| `SampleRate` | `int` | pipeline input rate | |
| `Host` | `string` | `api.elevenlabs.io` | |
| `Keyterms`, `SecondaryLanguages` | `[]string` | none | |
| `NoVerbatim`, `FilterBackgroundAudio` | `*bool` | nil | |
| `VADSilenceThresholdSecs`, `VADThreshold` | `*float64` | nil | |
| `MinSpeechDurationMs`, `MinSilenceDurationMs` | `*int` | nil | |
| `IncludeTimestamps`, `EnableLogging`, `IncludeLanguageDetection` | `bool` | false | |

```go
func newScribeRealtime() *elevenlabs.RealtimeSTTService {
	return elevenlabs.NewRealtimeSTTService(elevenlabs.RealtimeSTTConfig{APIKey: os.Getenv("ELEVENLABS_API_KEY")})
}
```

Runtime changes reconnect the websocket.

## fal (Wizper)

Per-utterance transcription on fal (segmented). Conventional key variable: `FAL_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Language` | `string` | `en` | |
| `Task` | `string` | `transcribe` | |
| `ChunkLevel` | `string` | `segment` | |
| `Version` | `string` | `3` | |
| `SampleRate` | `int` | pipeline input rate | |
| `URL` | `string` | `https://fal.run/fal-ai/wizper` | |
| `HTTPClient` | `*http.Client` | 30 s timeout | |

```go
func newFal() *fal.STTService {
	return fal.NewSTTService(fal.STTConfig{APIKey: os.Getenv("FAL_KEY")})
}
```

## Gladia

Gladia live transcription: an HTTP request opens a session, then audio streams over a websocket. Conventional key variable: `GLADIA_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `URL` | `string` | `https://api.gladia.io/v2/live` | Session init URL. |
| `Region` | `string` | none | Sent as a `region` query parameter when set. |
| `Model` | `string` | `solaria-1` | |
| `Language` | `string` | none | |
| `Encoding` | `string` | `wav/pcm` | |
| `BitDepth`, `Channels` | `int` | 16, 1 | |
| `SampleRate` | `int` | pipeline input rate | |
| `CustomMetadata` | `map[string]any` | none | |
| `Endpointing` | `*float64` | nil | |
| `MaximumDurationWithoutEndpointing` | `*int` | 5 | |
| `LanguageConfig`, `PreProcessing`, `RealtimeProcessing`, `MessagesConfig` | pointers to Gladia option structs | nil | Language list and code switching, audio enhancer, custom vocabulary, translation, message selection. |
| `EnableVAD` | `bool` | false | Use Gladia's speech events as the turn source. |
| `DisableInterruptions` | `bool` | false | With `EnableVAD`, caller speech does not interrupt the bot. |
| `ReplayBuffer` | `time.Duration` | 10 s | Recent audio replayed after a reconnect. |
| `HTTPClient` | `*http.Client` | 10 s timeout | For the session init request. |

```go
func newGladia() *gladia.STTService {
	return gladia.NewSTTService(gladia.STTConfig{APIKey: os.Getenv("GLADIA_API_KEY"), EnableVAD: true})
}
```

This service does not implement a settings hook, so an `STTUpdateSettingsFrame` only changes the stored settings, which a new session uses after a reconnect.

## Google Speech-to-Text

Per-utterance recognition with the Speech-to-Text v2 REST API (segmented). `NewSTTService` returns an error because it may load credentials.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | When empty, Application Default Credentials (or `CredentialsJSON`) are used. |
| `CredentialsJSON` | `[]byte` | none | Service-account JSON. |
| `ProjectID` | `string` | from the credentials | Required when you use `APIKey`. |
| `Location` | `string` | `global` | Non-global locations use `{Location}-speech.googleapis.com`. |
| `Model` | `string` | `latest_long` | |
| `Languages` | `[]string` | `en-US` | Stored in `Extra["language_codes"]`; when empty, `Settings.Language` is used. |
| `SampleRate` | `int` | pipeline input rate | |
| `Adaptation`, `DenoiserConfig` | `map[string]any` | none | |
| `Features` | `map[string]bool` | automatic punctuation on | Recognition features. |
| `BaseURL`, `HTTPClient` | | derived | |

```go
func newGoogleSTT() (*google.STTService, error) {
	return google.NewSTTService(google.STTConfig{
		APIKey:    os.Getenv("GOOGLE_API_KEY"),
		ProjectID: os.Getenv("GOOGLE_CLOUD_PROJECT"),
		Languages: []string{"en-US"},
	})
}
```

The maximum segment length is 55 seconds.

## Groq Whisper

Per-utterance transcription on Groq (segmented). Conventional key variable: `GROQ_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `whisper-large-v3-turbo` | |
| `Language` | `string` | `en` | |
| `Prompt` | `string` | none | |
| `Temperature` | `*float64` | nil | |
| `SampleRate` | `int` | pipeline input rate | |
| `BaseURL` | `string` | `https://api.groq.com/openai/v1/audio/transcriptions` | Full URL. |
| `HTTPClient` | `*http.Client` | 30 s timeout | |

```go
func newGroqSTT() *groq.STTService {
	return groq.NewSTTService(groq.STTConfig{APIKey: os.Getenv("GROQ_API_KEY")})
}
```

## OpenAI transcription

Per-utterance transcription with `/v1/audio/transcriptions` (segmented). Conventional key variable: `OPENAI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `gpt-transcribe` | |
| `Language` | `string` | `en` | |
| `Prompt` | `string` | none | |
| `Temperature` | `*float64` | nil | |
| `Keywords` | `[]string` | none | |
| `SampleRate` | `int` | pipeline input rate | |
| `BaseURL` | `string` | `https://api.openai.com/v1/audio/transcriptions` | Full URL. |
| `HTTPClient` | `*http.Client` | 30 s timeout | |

```go
func newOpenAISTT() *openai.STTService {
	return openai.NewSTTService(openai.STTConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
}
```

## OpenAI Realtime transcription

Streaming transcription over the Realtime API websocket, in `src/services/openai`. Audio is resampled to 24 kHz before sending.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `gpt-realtime-whisper` | |
| `Language` | `string` | `en` | |
| `Prompt` | `string` | none | Not sent for the default model. |
| `NoiseReduction`, `Delay` | `string` | none | |
| `TurnDetection` | `map[string]any` | nil | When set, it is sent as the session's `turn_detection` and the service pushes its own turn strategies. When nil, turn detection is off and the pipeline's VAD decides. |
| `DisableInterruptions` | `bool` | false | With server turn detection, caller speech does not interrupt the bot. |
| `SampleRate` | `int` | pipeline input rate | |
| `BaseURL` | `string` | `wss://api.openai.com/v1/realtime` | |

```go
func newOpenAIRealtimeSTT() *openai.RealtimeSTTService {
	return openai.NewRealtimeSTTService(openai.RealtimeSTTConfig{
		APIKey:        os.Getenv("OPENAI_API_KEY"),
		TurnDetection: map[string]any{"type": "server_vad"},
	})
}
```

Runtime changes are sent as a `session.update` message without reconnecting.

## OpenAI Realtime STT (standalone)

`openai_realtime.NewSTTService` is a separate, older implementation that is not built on `stt.Base`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `gpt-realtime-whisper` | |
| `Language` | `string` | empty | Language hint for transcription. |
| `Prompt` | `string` | empty | Transcription prompt. Not sent for `gpt-realtime-whisper`, which rejects it. |
| `NoiseReduction` | `string` | empty | Input noise reduction type. |
| `Endpoint` | `string` | `wss://api.openai.com/v1/realtime?intent=transcription` | |
| `TurnDetection` | `string` | empty (server VAD) | `local` or `false` switch server VAD off. |
| `Backoff` | `realtime.Backoff` | 500 ms initial, 15 s max, 6 attempts | Reconnect policy. |
| `Dialer` | `*websocket.Dialer` | `websocket.DefaultDialer` | |

```go
func newRealtimeSTTStandalone() *openai_realtime.STTService {
	return openai_realtime.NewSTTService(openai_realtime.STTConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
}
```

It does not handle `STTUpdateSettingsFrame`. Call `SetModel` on it directly instead; `SetLanguage` has no effect.

## Sarvam

Sarvam streaming STT over a websocket, aimed at Indian languages. Conventional key variable: `SARVAM_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `saaras:v3` | |
| `Language` | `string` | none | Short codes such as `hi` or `en` are mapped to `hi-IN`, `en-IN` and so on. |
| `Mode` | `string` | `transcribe` | |
| `Prompt` | `string` | none | |
| `Encoding` | `string` | `wav` | |
| `SampleRate` | `int` | pipeline input rate | |
| `VADSignals` | `*bool` | nil | When true, Sarvam's speech events are pushed as user started and stopped speaking frames. |
| `HighVADSensitivity` | `*bool` | nil | |
| `KeepaliveInterval` | `time.Duration` | 0 (off) | |
| `BaseURL` | `string` | `wss://api.sarvam.ai/speech-to-text/ws` | The `speech-to-text-translate` URL is used instead when the model is `saaras:v2.5`. |

```go
func newSarvamSTT() *sarvam.STTService {
	return sarvam.NewSTTService(sarvam.STTConfig{APIKey: os.Getenv("SARVAM_API_KEY"), Language: "hi"})
}
```

Runtime changes reconnect the websocket. `SetLanguage` maps short codes the same way as the config.

## Sarvam Realtime

Sarvam's low-latency realtime STT over a websocket (`wss://api.sarvam.ai/speech-to-text-realtime/ws`). It streams interim and final transcripts and detects the start and end of speech on Sarvam's side. Conventional key variable: `SARVAM_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent as the `API-SUBSCRIPTION-KEY` header. |
| `Model` | `string` | `saaras:v3-realtime` | Only `saaras:v3-realtime` and `saaras:v4` are accepted. The second-generation model is `saaras:v4` with no `-realtime` suffix; Sarvam rejects `saaras:v4-realtime`. |
| `Language` | `string` | `en-IN` | Short codes such as `hi` or `or` are mapped to `hi-IN`, `or-IN` and so on. `auto` asks Sarvam to detect the language of each utterance. Other values are sent as given. |
| `StreamType` | `string` | `balanced` | `fast`, `balanced` or `simulated`. |
| `Mode` | `string` | `transcribe` | |
| `Prompt` | `string` | none | |
| `Threshold` | `*float64` | nil | Sarvam VAD sensitivity. |
| `SilenceDurationMS` | `*int` | nil | Silence before Sarvam ends speech. |
| `MinSpeechDurationMS` | `*int` | nil | |
| `PrefixPaddingMS` | `*int` | nil | |
| `ReturnTimestamps` | `bool` | false | |
| `Keyterms` | `[]string` | nil | Recognition hints for `saaras:v4`, one phrase per item. |
| `SampleRate` | `int` | pipeline input rate | Must be 8000 or 16000. |
| `KeepaliveInterval` | `time.Duration` | 0 (off) | When set, an idle connection is kept open with Sarvam's `ping` event. |
| `BaseURL` | `string` | `wss://api.sarvam.ai/speech-to-text-realtime/ws` | |

`NewRealtimeSTTService` returns an error for an unsupported model or sample rate.

```go
func newSarvamRealtimeSTT() (*sarvam.RealtimeSTTService, error) {
	return sarvam.NewRealtimeSTTService(sarvam.RealtimeSTTConfig{APIKey: os.Getenv("SARVAM_API_KEY"), Language: "hi"})
}
```

How it behaves:

- Audio is sent as base64 PCM in 50 ms `audio_input` events.
- Sarvam's `vad.speech_start` and `vad.speech_end` are pushed upstream as user started and stopped speaking frames.
- `transcript.partial` becomes an `InterimTranscriptionFrame` and `transcript.final` a finalized `TranscriptionFrame`.
- An `error` event is pushed as a non-fatal error. A connection Sarvam closes with code 4000 (rejected settings, such as an unknown model or language) is reported as fatal and is not retried.
- If the connection drops mid-utterance, the utterance is ended and the latest partial transcript is pushed as the final transcript.
- Sarvam does the endpointing. The pipeline-driven `manual` endpointing mode is not supported.

## Soniox

Soniox real-time transcription over a websocket. Conventional key variable: `SONIOX_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `URL` | `string` | `wss://stt-rt.soniox.com/transcribe-websocket` | |
| `Model` | `string` | `stt-rt-v6` | |
| `SampleRate` | `int` | pipeline input rate | |
| `AudioFormat` | `string` | `pcm_s16le` | |
| `NumChannels` | `int` | 1 | |
| `LanguageHints` | `[]string` | none | |
| `LanguageHintsStrict` | `*bool` | nil | |
| `Context` | `any` | nil | A `soniox.Context` or any JSON value. |
| `EnableSpeakerDiarization`, `EnableLanguageIdentification` | `bool` | false | |
| `MaxEndpointDelayMs`, `EndpointLatencyLevel` | `*int` | nil | |
| `EndpointSensitivity` | `*float64` | nil | |
| `ClientReferenceID` | `string` | none | |
| `DisableVADForceTurnEndpoint` | `bool` | false | When false, the pipeline's VAD forces Soniox to end the turn. When true, Soniox's own endpoint detection decides and the service pushes its own turn strategies. |
| `DisableInterruptions` | `bool` | false | With Soniox endpointing, caller speech does not interrupt the bot. |

```go
func newSoniox() *soniox.STTService {
	return soniox.NewSTTService(soniox.STTConfig{
		APIKey:        os.Getenv("SONIOX_API_KEY"),
		LanguageHints: []string{"en", "es"},
	})
}
```

Runtime changes reconnect the websocket.

## Whisper

Per-utterance transcription against an OpenAI-style `/audio/transcriptions` endpoint (segmented). Conventional key variable: `OPENAI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `whisper-1` | |
| `Language` | `string` | none | Sent when set. |
| `SampleRate` | `int` | pipeline input rate | |
| `BaseURL` | `string` | `https://api.openai.com/v1/audio/transcriptions` | Full URL; point it at any Whisper-compatible server. |
| `HTTPClient` | `*http.Client` | 30 s timeout | |

```go
func newWhisper() *whisper.STTService {
	return whisper.NewSTTService(whisper.STTConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
}
```

## Change model or language during a call

Queue an `STTUpdateSettingsFrame` on the task. The base merges the delta into its settings and then:

- **Segmented** services (Whisper, OpenAI, Groq, ElevenLabs Scribe, fal, Google) use the new settings for the next utterance.
- **Most streaming** services reconnect the websocket with the new settings: AssemblyAI, Azure, Cartesia, Deepgram, ElevenLabs Realtime, Sarvam, Soniox. AWS Transcribe restarts its stream.
- **OpenAI Realtime transcription** sends a `session.update`. **Sarvam Realtime** sends a `config.update` for a language change and reconnects for a model change. **Deepgram Flux** sends a configure message or reconnects, as described above.
- **Gladia** keeps the new settings until its next reconnect.

```go
func switchToSpanish(task *pipeline.PipelineTask) error {
	return task.QueueFrame(frames.NewSTTUpdateSettingsFrame(frames.STTSettings{
		Model:    "nova-3",
		Language: "es",
	}))
}
```

The frame is uninterruptible. Without a `Service` field, the first STT service that receives it applies it and does not pass it on. You can also call `SetLanguage`, `SetModel` or `UpdateSettings` on the service directly. A worked example is in the [multilingual bot recipe](../cookbook/multilingual-bot.md).

## Next steps

- [Turn-taking](../guides/turn-taking.md): VAD, server-side turn detection and interruptions.
- [TTS providers](tts.md): the other half of the voice.
- [STT and TTS bases reference](../reference/stt-tts-bases.md): `stt.Base` options in full.
- [Writing a custom provider](custom.md): add an STT vendor.
