# TTS providers

This page shows how to configure every TTS (text-to-speech) integration: the config fields and their defaults, a constructor example, the quirks the code handles, and how to change the voice, model or language while a call is running.

> [!NOTE]
> No TTS integration has been verified against the live vendor API. See the status column in the [provider matrix](README.md#tts).

A TTS service is a pipeline **processor** that receives the LLM's text (`LLMTextFrame`) or a fixed phrase (`TTSSpeakFrame`) and pushes audio (`TTSAudioFrame`) towards the transport output. When the caller interrupts the bot (a **barge-in**), the service cancels the speech that is still in flight.

## Choosing a provider

| If you need | Use | Why |
| --- | --- | --- |
| Word-level timestamps, so the context records the reply word by word up to a barge-in | Cartesia, ElevenLabs (`UseStreaming`), ElevenLabs HTTP, Inworld, Resemble, Rime, Speechify | these set `WordTimestamps` and report words to the base |
| Speech to start before a full sentence is ready | Cartesia, ElevenLabs (`UseStreaming`), Inworld, Neuphonic, Rime, Sarvam with `StreamTokens` | they send tokens instead of sentences |
| One persistent connection for the whole call | the websocket providers: Cartesia, Deepgram, ElevenLabs (`UseStreaming`), Fish, Inworld, Neuphonic, Resemble, Rime, Sarvam | they open a socket at start and reuse it |
| A plain HTTP request per sentence | the HTTP providers: Azure, Google, OpenAI, Groq, MiniMax, Async, Polly, Cartesia HTTP, Deepgram HTTP, Neuphonic HTTP, Rime HTTP, Sarvam HTTP, Piper, XTTS | nothing to keep alive between turns |
| A model you host yourself | `piper`, `xtts` | no API key; you give the server URL |
| mu-law or a-law output | Deepgram (`Encoding`), Azure (`OutputFormat`), Google (`Encoding`), Async (`Encoding`), ElevenLabs (`OutputFormat`), Cartesia (`Encoding`) | the service tags its audio with that codec, so telephony audio needs no PCM step at the vendor |

## Complete example

This program answers Twilio calls with Deepgram STT, an OpenAI LLM and ElevenLabs TTS over a websocket. Swap the `tts :=` line for any constructor on this page.

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
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
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
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a friendly phone assistant. Answer in one or two short sentences.",
	})
	tts := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       os.Getenv("ELEVENLABS_API_KEY"),
		VoiceID:      os.Getenv("ELEVENLABS_VOICE_ID"),
		Model:        "eleven_flash_v2_5",
		UseStreaming: true,
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

Run it with the keys in your environment and point a Twilio number at `wss://YOUR_HOST/media`, as described in [Your first phone bot](../getting-started/first-phone-bot.md):

```sh
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export ELEVENLABS_API_KEY=YOUR_ELEVENLABS_API_KEY
export ELEVENLABS_VOICE_ID=YOUR_ELEVENLABS_VOICE_ID
go run .
```

The TTS service sits between the LLM and `s.Output()`. The `TTSSpeakFrame` queued before `Run` makes the bot greet the caller without asking the LLM.

## Options every TTS service shares

Every service on this page embeds `*tts.Base`, which provides:

- Sentence aggregation: LLM tokens are collected into sentences before they are sent, unless the provider streams tokens.
- `SampleRate()`: the config's `SampleRate` if set, otherwise the pipeline's `AudioOutSampleRate` (24000 by default). Azure and Google fix their own rate from their output format.
- `Settings()`, `UpdateSettings(delta)`, `SetVoice`, `SetModel`, `SetLanguage`: the current `frames.TTSSettings{Model, Voice, Language, Speed, Extra}`.
- Handling of interruptions, `TTSUpdateSettingsFrame`, TTFB and character-usage metrics.

The library never reads environment variables for keys. The variable names below are the convention the examples use.

## Async

HTTP synthesis on Async. Conventional key variable: `ASYNC_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Voice` | `string` | none | Voice ID. |
| `Model` | `string` | `async_flash_v1.5` | |
| `Language` | `string` | none | |
| `URL` | `string` | `https://api.async.com` | |
| `Version` | `string` | `v1` | |
| `Encoding` | `string` | `pcm_s16le` | `pcm_mulaw` and `pcm_alaw` set the codec to mulaw or alaw. |
| `Container` | `string` | `raw` | |
| `SampleRate` | `int` | pipeline output rate | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newAsync() *asyncai.HTTPTTSService {
	return asyncai.NewHTTPTTSService(asyncai.TTSConfig{
		APIKey: os.Getenv("ASYNC_API_KEY"),
		Voice:  os.Getenv("ASYNC_VOICE_ID"),
	})
}
```

## AWS Polly

Polly through the AWS SDK. `NewTTSService(ctx, cfg)` loads AWS config and returns an error; `NewTTSServiceWithClient` takes your own `*polly.Client`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AWS` | `awsauth.Config` | zero | Region (default `us-east-1`), static keys, profile, endpoint. Empty fields use the AWS SDK default chain. |
| `Voice` | `string` | `Joanna` | |
| `Language` | `string` | `en-US` | |
| `Engine` | `string` | none | Stored as the model; overridable with `Extra["engine"]`. |
| `Pitch`, `Rate`, `Volume` | `string` | none | SSML prosody attributes. |
| `LexiconNames` | `[]string` | none | |
| `SampleRate` | `int` | pipeline output rate | |

Polly is asked for 8000 Hz or 16000 Hz PCM, and the result is resampled when the pipeline rate differs. Text is wrapped in SSML.

```go
func newPolly(ctx context.Context) (*polly.TTSService, error) {
	return polly.NewTTSService(ctx, polly.TTSConfig{
		AWS:    awsauth.Config{Region: "us-east-1"},
		Voice:  "Joanna",
		Engine: "neural",
	})
}
```

## Azure Speech

HTTP synthesis on Azure Speech. Conventional key variable: `AZURE_SPEECH_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `SubscriptionKey` | `string` | none | |
| `Region` | `string` | `eastus` | |
| `Voice` | `string` | `en-US-JennyNeural` | |
| `Language` | `string` | `en-US` | |
| `OutputFormat` | `string` | `raw-16khz-16bit-mono-pcm` | Also sets the sample rate and codec. |
| `Endpoint` | `string` | `https://{Region}.tts.speech.microsoft.com/cognitiveservices/v1` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newAzureTTS() *azure.TTSService {
	return azure.NewTTSService(azure.TTSConfig{
		SubscriptionKey: os.Getenv("AZURE_SPEECH_KEY"),
		Region:          "westeurope",
		Voice:           "en-GB-SoniaNeural",
		Language:        "en-GB",
	})
}
```

## Cartesia

Cartesia over a websocket. Conventional key variables: `CARTESIA_API_KEY`, and `CARTESIA_VOICE_ID` for the voice.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | Sent as `X-API-Key`. |
| `VoiceID` | `string` | none | |
| `Model` | `string` | `sonic-3` | |
| `CartesiaVersion` | `string` | `2025-04-16` | `Cartesia-Version` header. |
| `Language` | `string` | `en` | |
| `SampleRate` | `int` | pipeline output rate | |
| `Encoding` | `string` | `pcm_s16le` | `pcm_mulaw`, `pcm_alaw` and `pcm_f32le` are recognised. |
| `Container` | `string` | `raw` | |
| `GenerationConfig` | `*cartesia.GenerationConfig` | nil | `Volume`, `Speed`, `Emotion`. `Settings.Speed` overrides `Speed`. |
| `StreamTokens` | `bool` | false | Send LLM tokens as they arrive instead of whole sentences. |
| `AggregateSentences` | `bool` | false | Present in the struct but not read by the code; sentence aggregation is the default anyway. |
| `PronunciationDictID` | `string` | none | |
| `BaseURL` | `string` | `wss://api.cartesia.ai/tts/websocket` | |

```go
func newCartesia() *cartesia.TTSService {
	return cartesia.NewTTSService(cartesia.TTSConfig{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: os.Getenv("CARTESIA_VOICE_ID"),
	})
}
```

Word timestamps are always requested. Settings are read for every request, so a runtime change applies from the next sentence without reconnecting.

### Cartesia over HTTP

`cartesia.NewHTTPTTSService(cartesia.HTTPTTSConfig{...})` has the same fields minus `Container`, `StreamTokens` and `AggregateSentences`, plus `HTTPClient`. `BaseURL` defaults to `https://api.cartesia.ai`.

```go
func newCartesiaHTTP() *cartesia.HTTPTTSService {
	return cartesia.NewHTTPTTSService(cartesia.HTTPTTSConfig{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: os.Getenv("CARTESIA_VOICE_ID"),
	})
}
```

## Deepgram Aura

Deepgram TTS over a websocket. Conventional key variable: `DEEPGRAM_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `aura-asteria-en` | Deepgram's model name is also the voice. |
| `Encoding` | `string` | `linear16` | `mulaw` and `alaw` use 1 byte per sample and 8000 Hz unless `SampleRate` is set. |
| `SampleRate` | `int` | pipeline output rate | |
| `BaseURL` | `string` | `wss://api.deepgram.com/v1/speak` | |

```go
func newDeepgramTTS() *deepgram.TTSService {
	return deepgram.NewTTSService(deepgram.TTSConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "aura-2-thalia-en"})
}
```

Runtime changes reconnect the websocket.

### Deepgram Aura over HTTP

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Voice` | `string` | `aura-2-helena-en` | |
| `Encoding` | `string` | `linear16` | as above |
| `SampleRate` | `int` | pipeline output rate | |
| `Speed` | `float64` | 0 (not sent) | |
| `MIPOptOut` | `*bool` | nil | |
| `URL` | `string` | `https://api.deepgram.com/v1/speak` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newDeepgramHTTP() *deepgram.HTTPTTSService {
	return deepgram.NewHTTPTTSService(deepgram.HTTPTTSConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
}
```

## ElevenLabs

`elevenlabs.NewTTSService` runs over a websocket when `UseStreaming` is true, and over HTTP otherwise. Conventional key variables: `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `VoiceID` | `string` | none | |
| `Model` | `string` | none | Set it; there is no default in this constructor. |
| `OutputFormat` | `string` | `pcm_{SampleRate}` | `ulaw_*`, `alaw_*`, `mp3_*`, `opus_*` and `pcm_*` set the codec. |
| `UseStreaming` | `bool` | false | Websocket mode with word timestamps. |
| `VoiceSettings` | `*elevenlabs.VoiceSettings` | stability 0.5, similarity boost 0.75 | |
| `Language` | `string` | none | Sent only with `eleven_flash_v2_5` or `eleven_turbo_v2_5`. |
| `StreamTokens` | `bool` | false | Token streaming; only with `UseStreaming`. |
| `AggregateSentences` | `bool` | false | Present in the struct but not read by the code. |
| `WebsocketURL` | `string` | `wss://api.elevenlabs.io` | |
| `HTTPURL` | `string` | `https://api.elevenlabs.io` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newElevenLabs() *elevenlabs.TTSService {
	return elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       os.Getenv("ELEVENLABS_API_KEY"),
		VoiceID:      os.Getenv("ELEVENLABS_VOICE_ID"),
		Model:        "eleven_flash_v2_5",
		UseStreaming: true,
	})
}
```

Runtime changes reconnect the websocket in streaming mode; in HTTP mode the next request uses them.

### ElevenLabs HTTP with timestamps

`elevenlabs.NewHTTPTTSService` streams each context over HTTP and still reports word timestamps.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey`, `VoiceID`, `Language`, `SampleRate`, `VoiceSettings` | | | as above |
| `Model` | `string` | `eleven_flash_v2_5` | |
| `OptimizeStreamingLatency` | `*int` | nil | |
| `EnableLogging` | `*bool` | nil | |
| `ApplyTextNormalization` | `string` | none | |
| `BaseURL` | `string` | `https://api.elevenlabs.io` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newElevenLabsHTTP() *elevenlabs.HTTPTTSService {
	return elevenlabs.NewHTTPTTSService(elevenlabs.HTTPTTSConfig{
		APIKey:  os.Getenv("ELEVENLABS_API_KEY"),
		VoiceID: os.Getenv("ELEVENLABS_VOICE_ID"),
	})
}
```

## Fish Audio

Fish Audio live TTS over a websocket with MessagePack messages. Conventional key variable: `FISH_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `ReferenceID` | `string` | none | The voice (stored as `Voice`). |
| `Model` | `string` | `s2.1-pro` | |
| `Language` | `string` | none | |
| `Latency` | `string` | `balanced` | |
| `Speed` | `float64` | 1 | |
| `Volume` | `int` | 0 | |
| `Params` | `map[string]any` | none | Extra request fields. |
| `SampleRate` | `int` | pipeline output rate | |
| `URL` | `string` | `wss://api.fish.audio/v1/tts/live` | |

```go
func newFish() *fish.TTSService {
	return fish.NewTTSService(fish.TTSConfig{APIKey: os.Getenv("FISH_API_KEY"), ReferenceID: os.Getenv("FISH_VOICE_ID")})
}
```

Runtime changes reconnect the websocket.

## Google Cloud Text-to-Speech

HTTP synthesis with an API key. Conventional key variable: `GOOGLE_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `LanguageCode` | `string` | `en-US` | |
| `VoiceName` | `string` | `en-US-Neural2-C` | |
| `Gender` | `google.VoiceGender` | `FEMALE` | |
| `Encoding` | `google.AudioEncoding` | `LINEAR16` | `MULAW` and `ALAW` use 1 byte per sample. |
| `SampleRate` | `int` | 16000 | Fixed by this field, not by the pipeline. |
| `Endpoint` | `string` | `https://texttospeech.googleapis.com/v1/text:synthesize` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newGoogleTTS() *google.TTSService {
	return google.NewTTSService(google.TTSConfig{APIKey: os.Getenv("GOOGLE_API_KEY"), VoiceName: "en-US-Neural2-F"})
}
```

## Groq (Orpheus)

HTTP synthesis on Groq. Conventional key variable: `GROQ_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `BaseURL` | `string` | `https://api.groq.com/openai/v1` | |
| `Model` | `string` | `canopylabs/orpheus-v1-english` | |
| `Voice` | `string` | `autumn` | |
| `Speed` | `float64` | 1 | |
| `SampleRate` | `int` | pipeline output rate | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newGroqTTS() *groq.TTSService {
	return groq.NewTTSService(groq.TTSConfig{APIKey: os.Getenv("GROQ_API_KEY")})
}
```

## Inworld

Inworld over a websocket (`NewTTSService`) or HTTP streaming (`NewHTTPTTSService`). Both report word timestamps. Conventional key variable: `INWORLD_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `VoiceID` | `string` | `Ashley` | |
| `Model` | `string` | `inworld-tts-2` | |
| `Language` | `string` | none | |
| `Speed` | `float64` | 0 (not set) | |
| `SampleRate` | `int` | pipeline output rate | |
| `Encoding` | `string` | `PCM` | |
| `Params` | `map[string]any` | none | Extra request fields. |
| `AutoMode` | `*bool` | true unless `StreamTokens` | Websocket only. |
| `ApplyTextNormalization` | `string` | none | Websocket only. |
| `MaxBufferDelayMs` | `int` | 3000 | Websocket only. |
| `BufferCharThreshold` | `int` | 0 | Websocket only. |
| `StreamTokens` | `bool` | false | Websocket only. |
| `URL` | `string` | websocket: `wss://api.inworld.ai/tts/v1/voice:streamBidirectional`; HTTP: `https://api.inworld.ai/tts/v1/voice:stream` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | HTTP only. |

```go
func newInworld() *inworld.TTSService {
	return inworld.NewTTSService(inworld.TTSConfig{APIKey: os.Getenv("INWORLD_API_KEY")})
}

func newInworldHTTP() *inworld.HTTPTTSService {
	return inworld.NewHTTPTTSService(inworld.TTSConfig{APIKey: os.Getenv("INWORLD_API_KEY")})
}
```

Runtime changes reconnect the websocket service; the HTTP service uses them on the next request.

## MiniMax

HTTP synthesis on MiniMax. Conventional key variable: `MINIMAX_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `GroupID` | `string` | none | Appended as a `GroupId` query parameter when set. |
| `Model` | `string` | `speech-2.8-turbo` | |
| `VoiceID` | `string` | `Calm_Woman` | |
| `Language` | `string` | none | |
| `Speed` | `float64` | 0 (not set) | |
| `Params` | `map[string]any` | none | |
| `SampleRate` | `int` | pipeline output rate | |
| `URL` | `string` | `https://api.minimax.io/v1/t2a_v2` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newMiniMax() *minimax.TTSService {
	return minimax.NewTTSService(minimax.TTSConfig{APIKey: os.Getenv("MINIMAX_API_KEY"), GroupID: os.Getenv("MINIMAX_GROUP_ID")})
}
```

## Neuphonic

Neuphonic over a websocket (`NewTTSService`) or HTTP (`NewHTTPTTSService`). Conventional key variable: `NEUPHONIC_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `VoiceID` | `string` | none | |
| `Language` | `string` | `en` | |
| `Speed` | `float64` | 1 | |
| `Temperature` | `float64` | 0 (not sent) | |
| `IPAKey` | `string` | none | |
| `SampleRate` | `int` | pipeline output rate | |
| `StreamTokens` | `bool` | false | Websocket only. |
| `URL` | `string` | websocket: `wss://api.neuphonic.com`; HTTP: `https://api.neuphonic.com` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | HTTP only. |

```go
func newNeuphonic() *neuphonic.TTSService {
	return neuphonic.NewTTSService(neuphonic.TTSConfig{APIKey: os.Getenv("NEUPHONIC_API_KEY")})
}
```

Runtime changes reconnect the websocket service.

## OpenAI

HTTP synthesis with `/v1/audio/speech`. Conventional key variable: `OPENAI_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `BaseURL` | `string` | `https://api.openai.com/v1` | |
| `Model` | `string` | `gpt-4o-mini-tts` | |
| `Voice` | `string` | `alloy` | Against the default base URL, the voice must be one of `openai.ValidTTSVoices` or the request fails. |
| `Instructions` | `string` | none | Speaking-style instructions. |
| `Speed` | `float64` | 0 (not set) | |
| `SampleRate` | `int` | pipeline output rate | OpenAI returns 24 kHz; the service resamples to this rate. |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newOpenAITTS() *openai.TTSService {
	return openai.NewTTSService(openai.TTSConfig{
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		Voice:        "coral",
		Instructions: "Speak warmly and at a calm pace.",
	})
}
```

## Piper (self-hosted)

HTTP synthesis against your own Piper server. No API key.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `BaseURL` | `string` | none | Required. |
| `Voice` | `string` | none | |
| `Speed` | `float64` | 0 (not set) | |
| `Params` | `map[string]any` | none | |
| `SampleRate` | `int` | pipeline output rate | Audio is resampled to this rate. |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newPiper() *piper.TTSService {
	return piper.NewTTSService(piper.TTSConfig{BaseURL: "http://localhost:5000"})
}
```

## Resemble

Resemble over a websocket, with word timestamps. Conventional key variable: `RESEMBLE_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `VoiceID` | `string` | none | |
| `SampleRate` | `int` | pipeline output rate | |
| `ApplyCustomPronunciations` | `bool` | false | |
| `URL` | `string` | `wss://websocket.cluster.resemble.ai/stream` | |

```go
func newResemble() *resemble.TTSService {
	return resemble.NewTTSService(resemble.TTSConfig{APIKey: os.Getenv("RESEMBLE_API_KEY"), VoiceID: os.Getenv("RESEMBLE_VOICE_ID")})
}
```

## Rime

Rime over a websocket (`NewTTSService`, with word timestamps) or HTTP (`NewHTTPTTSService`). Conventional key variable: `RIME_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `VoiceID` | `string` | none | |
| `Model` | `string` | `coda` | |
| `Language` | `string` | none | |
| `Speed` | `float64` | 0 (not set) | |
| `SampleRate` | `int` | pipeline output rate | |
| `Params` | `map[string]any` | none | |
| `StreamTokens` | `bool` | false | Websocket only. |
| `URL` | `string` | websocket: `wss://users-ws.rime.ai/ws3`; HTTP: `https://users.rime.ai/v1/rime-tts` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | HTTP only. |

```go
func newRime() *rime.TTSService {
	return rime.NewTTSService(rime.TTSConfig{APIKey: os.Getenv("RIME_API_KEY"), VoiceID: os.Getenv("RIME_VOICE_ID")})
}
```

Runtime changes reconnect the websocket service.

## Sarvam

Sarvam over a websocket (`NewTTSService`) or HTTP (`NewHTTPTTSService`), aimed at Indian languages. Conventional key variable: `SARVAM_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `Model` | `string` | `bulbul:v3` | Must be one of `bulbul:v2`, `bulbul:v3-beta`, `bulbul:v3`, `bulbul:v4-flash`. |
| `VoiceID` | `string` | the model's default speaker (`shubh` for v3) | |
| `Language` | `string` | `en-IN` | |
| `Pace` | `float64` | 0 (not set) | Stored as `Speed`. |
| `SampleRate` | `int` | pipeline output rate | |
| `Params` | `map[string]any` | none | |
| `StreamTokens` | `bool` | false | Websocket only. |
| `URL` | `string` | websocket: `wss://api.sarvam.ai/text-to-speech/ws`; HTTP: `https://api.sarvam.ai` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | HTTP only. |

```go
func newSarvamTTS() *sarvam.TTSService {
	return sarvam.NewTTSService(sarvam.TTSConfig{APIKey: os.Getenv("SARVAM_API_KEY"), Language: "hi-IN"})
}
```

A runtime model change reconnects the websocket; other changes are sent as a config message. An unknown model is rejected with an error frame.

## Speechify

Speechify over HTTP streaming with word timestamps. Conventional key variable: `SPEECHIFY_API_KEY`.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `APIKey` | `string` | none | |
| `VoiceID` | `string` | `geffen_32` | |
| `Model` | `string` | `simba-3.5` | |
| `Language` | `string` | none | |
| `Params` | `map[string]any` | none | |
| `SampleRate` | `int` | pipeline output rate | |
| `URL` | `string` | `https://api.speechify.ai` | |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newSpeechify() *speechify.TTSService {
	return speechify.NewTTSService(speechify.TTSConfig{APIKey: os.Getenv("SPEECHIFY_API_KEY")})
}
```

## XTTS (self-hosted)

HTTP synthesis against your own XTTS server. No API key.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `BaseURL` | `string` | none | Required. |
| `Voice` | `string` | none | |
| `Language` | `string` | `en` | |
| `SampleRate` | `int` | pipeline output rate | XTTS returns 24 kHz; the service resamples. |
| `HTTPClient` | `*http.Client` | `&http.Client{}` | |

```go
func newXTTS() *xtts.TTSService {
	return xtts.NewTTSService(xtts.TTSConfig{BaseURL: "http://localhost:8020", Voice: "speaker"})
}
```

## Change voice, model or language during a call

Queue a `TTSUpdateSettingsFrame` on the task. The base merges the delta into its settings. Then:

- **HTTP** services and Cartesia, Resemble, Speechify, ElevenLabs HTTP and Inworld HTTP use the new settings from the next sentence.
- Deepgram, ElevenLabs (websocket mode), Fish, Inworld, Neuphonic and Rime reconnect their websocket.
- Sarvam reconnects for a model change and sends a config message otherwise.

```go
func switchVoice(task *pipeline.PipelineTask, voiceID string) error {
	return task.QueueFrame(frames.NewTTSUpdateSettingsFrame(frames.TTSSettings{
		Voice:    voiceID,
		Language: "es",
	}))
}
```

The frame is uninterruptible. Without a `Service` field, the first TTS service that receives it applies it and does not pass it on. You can also call `SetVoice`, `SetModel`, `SetLanguage` or `UpdateSettings` on the service directly. Which fields a vendor honours depends on the integration; for example ElevenLabs sends `Language` only for its multilingual flash and turbo models. A worked example is in the [multilingual bot recipe](../cookbook/multilingual-bot.md).

## Next steps

- [Audio](../guides/audio.md): sample rates, codecs and resampling.
- [STT providers](stt.md): the listening side.
- [STT and TTS bases reference](../reference/stt-tts-bases.md): `tts.Base` options in full.
- [Writing a custom provider](custom.md): add a TTS vendor.
