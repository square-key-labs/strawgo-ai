# Audio

This guide covers everything about the audio itself: sample rates from the caller to the TTS service and back, sending µ-law straight through without conversion, resampling, background sound, noise gating, where voice activity detection goes, generating keypad tones, and classifying audio.

## Before you begin

- Read [Transports overview](transports-overview.md) for sessions and `TransportParams`.
- Terms used on this page:
  - **PCM** (also "linear16") is raw 16-bit little-endian audio samples. Strawgo's pipeline works in mono PCM.
  - **µ-law** and **A-law** (G.711) are 8-bit telephone encodings at 8000 samples per second.
  - The **sample rate** is samples per second (Hz). **Resampling** converts audio from one rate to another.
  - **VAD** (voice activity detection) decides whether a chunk of audio contains speech.
  - **DTMF** is the pair of tones a phone keypad makes.

## Sample rates end to end

| Stage | Rate | Set by |
| --- | --- | --- |
| Wire, caller to bot | The provider's rate, for example 8000 Hz for Twilio | The serializer (see the table in [Transports overview](transports-overview.md#sample-rates-end-to-end)) |
| Pipeline input (STT, VAD) | 16000 Hz by default | `PipelineTaskConfig.AudioInSampleRate`, or `TransportParams.AudioInSampleRate` to override per transport |
| TTS output | 24000 Hz by default | `PipelineTaskConfig.AudioOutSampleRate`, or the TTS service's own `SampleRate` |
| Wire, bot to caller | The provider's rate | The serializer; for serializers without a fixed rate, `TransportParams.AudioOutSampleRate` |

The session's input resamples caller audio to the pipeline input rate. The session's output decodes, mixes channels down to mono, and resamples TTS audio to the wire rate. You do not need to add converters for the normal case.

To avoid an unnecessary resampling step on 8 kHz phone calls, ask the TTS service for 8 kHz directly:

```go
func phoneRateConfig() *pipeline.PipelineTaskConfig {
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.AudioOutSampleRate = 8000
	return cfg
}
```

Pass it to `pipeline.NewPipelineTaskWithConfig`. TTS services that do not set their own `SampleRate` use the task's output rate.

## Native passthrough

Phone providers send and expect µ-law. By default Strawgo decodes µ-law to PCM on the way in and encodes PCM to µ-law on the way out. If your STT and TTS services speak µ-law natively, you can skip both conversions.

```go
func nativeParams() transports.TransportParams {
	return transports.TransportParams{
		AudioInNative:  true,
		AudioOutNative: true,
	}
}

func nativeServices() (*deepgram.STTService, *cartesia.TTSService) {
	stt := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:     os.Getenv("DEEPGRAM_API_KEY"),
		Model:      "nova-3",
		Language:   "en",
		Encoding:   "mulaw",
		SampleRate: 8000,
	})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{
		APIKey:     os.Getenv("CARTESIA_API_KEY"),
		VoiceID:    os.Getenv("CARTESIA_VOICE_ID"),
		Model:      "sonic-2",
		Language:   "en",
		Encoding:   "pcm_mulaw",
		SampleRate: 8000,
	})
	return stt, tts
}
```

How it behaves:

- **`AudioInNative`** asks the serializer to deliver caller audio in its wire encoding. Each `AudioFrame` then carries µ-law bytes and the metadata `codec = "mulaw"`. Input does not resample it.
- **`AudioOutNative`** sends a TTS frame without conversion only when all of these hold: the serializer supports native audio (Twilio, Telnyx, Plivo, Asterisk with a G.711 codec, Genesys); the frame's codec matches the wire codec; the frame's rate equals the wire rate; the frame is mono; and no output mixer is configured. Any other frame is converted as usual, so turning it on is safe.

> [!WARNING]
> With `AudioInNative`, processors that read `AudioFrame` data as PCM see µ-law bytes instead. That includes `vad.NewVADInputProcessor`, `classify.NewProcessor` and the [recording](recording.md) processor. If you need VAD with native input, use the transport's VAD hook (`TransportParams.NewAudioInAnalyzer`, below), which decodes a private PCM copy for analysis. The input filter does the same.

## Resampling and conversion

The session handles rate conversion for you. When you need it yourself, there are three levels:

| Need | Use |
| --- | --- |
| Resample a complete buffer once | `resample.ResampleBytes(pcm, inRate, outRate)` or `resample.Resample` for `[]int16` |
| Resample a stream chunk by chunk, keeping filter state between chunks | `resample.New(inRate, outRate)` then `ProcessBytes` per chunk and `Flush` at the end |
| Convert codec and rate inside a pipeline | `audio.NewAudioConverterProcessor(audio.AudioConverterConfig{...})` |
| Decode or encode G.711 | `audio.DecodePCM16(dst, src, codec)`, `audio.EncodePCM16(dst, pcm, codec)`, or the `g711` package |

```go
func toPhoneAudio(pcm16k []byte) ([]byte, error) {
	pcm8k, err := resample.ResampleBytes(pcm16k, 16000, 8000)
	if err != nil {
		return nil, err
	}
	return g711.PCM16ToUlaw(nil, pcm8k), nil
}
```

`audio.NormalizeCodec` accepts the common spellings (`ulaw`, `PCMU`, `audio/x-mulaw`, `l16`, `slin` and so on) and returns `mulaw`, `alaw` or `linear16`. `audio.AudioConverterConfig` takes `InputSampleRate`, `InputCodec`, `OutputSampleRate` and `OutputCodec`; it converts both caller `AudioFrame`s and `TTSAudioFrame`s that pass through it.

## A complete example: noise gate, background sound and VAD

This Twilio bot cleans up caller audio with a noise gate, plays quiet background sound under the bot's voice, and runs Silero VAD in the pipeline. It needs a running `onnx-worker` (see [Deployment](deployment.md#run-the-onnx-worker-sidecar)) and a WAV file at `office.wav`.

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/audio/filter"
	"github.com/square-key-labs/strawgo-ai/src/audio/mixer"
	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
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
		Params: transports.TransportParams{
			NewInputFilter: func() filter.Filter {
				return filter.NewNoiseGate(filter.DefaultNoiseGateParams())
			},
			NewOutputMixer: func() mixer.Mixer {
				return mixer.NewSoundfileMixer(map[string]string{"office": "office.wav"}, "office", mixer.WithVolume(0.2))
			},
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
	analyzer, err := vad.NewSileroVADAnalyzer(16000, vad.DefaultVADParams(), mustEnv("ONNX_WORKER_SOCK"))
	if err != nil {
		return err
	}
	defer analyzer.Cleanup()

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: mustEnv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: mustEnv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a receptionist. Keep answers short."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: mustEnv("CARTESIA_API_KEY"), VoiceID: mustEnv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), vad.NewVADInputProcessor(analyzer), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	return pipeline.NewPipelineTask(pipe).Run(ctx)
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s is required", key)
	}
	return v
}
```

The sections below explain each part.

## Mixers: background sound

A mixer adds sound to the bot's outgoing audio. `mixer.NewSoundfileMixer(files, defaultSound, opts...)` loops a WAV file under everything the bot says.

- `files` maps a name to a WAV path. Files are loaded when the call starts and resampled to the wire rate. Stereo files are mixed down to mono.
- Options: `mixer.WithVolume(v)` (default 0.4), `mixer.WithLoop(bool)` (default true), `mixer.WithMixing(bool)` to start disabled.
- Once a mixer is set, the output sends audio continuously: when the bot is quiet the caller hears the background sound alone.
- Use `TransportParams.NewOutputMixer` so each call gets its own mixer (and its own playback position).

Change it during a call by pushing frames down the pipeline (for example from a function call handler):

```go
func switchBackground(task *pipeline.PipelineTask) error {
	if err := task.QueueFrame(frames.NewMixerUpdateSettingsFrame(map[string]any{
		"sound":  "office",
		"volume": 0.1,
		"loop":   true,
	})); err != nil {
		return err
	}
	return task.QueueFrame(frames.NewMixerEnableFrame(true))
}
```

The settings keys are `sound` (a name from `files`), `volume` (number) and `loop` (bool). `MixerEnableFrame(false)` silences the background without removing the mixer. A mixer disables [native passthrough](#native-passthrough) for output, because the audio must be decoded to mix it. To write your own mixer, implement `mixer.Mixer` (`Start`, `Stop`, `ProcessFrame`, `Mix`). The [background ambience recipe](../cookbook/background-ambience.md) has a fuller example.

## Filters: noise gate

An input filter cleans caller audio before STT and VAD see it. `filter.NewNoiseGate(params)` mutes audio below a threshold, which keeps background noise from triggering speech detection.

| `NoiseGateParams` field | Default | Meaning |
| --- | --- | --- |
| `ThresholdDB` | -45 | Level (dBFS) above which the gate opens. |
| `AttackMs` | 2 | How fast the gate opens. |
| `HoldMs` | 150 | How long it stays open after the level drops. |
| `ReleaseMs` | 80 | How fast it closes. |
| `FloorDB` | -80 | How quiet "closed" is. |

Change it during a call with `frames.NewFilterUpdateSettingsFrame(map[string]any{"threshold_db": -40.0})` (keys: `threshold_db`, `attack_ms`, `hold_ms`, `release_ms`, `floor_db`) and turn it off or on with `frames.NewFilterEnableFrame(false)`. These frames must travel downstream through `s.Input()`, so queue them on the task.

Write your own filter as a function, or combine several:

```go
func newInputFilter() filter.Filter {
	halveVolume := filter.NewFuncFilter(func(dst, audio []byte) []byte {
		return pcm.Scale(dst, audio, 0.5)
	})
	return filter.NewChain(filter.NewNoiseGate(filter.DefaultNoiseGateParams()), halveVolume)
}
```

The filter runs on PCM. With native input, the session decodes the audio, filters it, and encodes it back.

The [WebRTC transport](webrtc.md) takes the same `InputFilter` and `NewInputFilter` fields in `webrtc.Params`. It filters the decoded caller audio before the analyzer and before the audio enters the pipeline, and honours the same filter frames.

## Where VAD goes

VAD decides when the caller is speaking. Turn-taking (deciding when the caller has finished) builds on it; see [Turn-taking](turn-taking.md). There are two places to run it.

**In the pipeline (recommended).** Put `vad.NewVADInputProcessor(analyzer)` right after `s.Input()`, as in the complete example. It emits the VAD speaking frames, supports a smart-turn model with `vad.NewVADInputProcessorWithTurn`, and works on every transport. Create the analyzer inside your handler so each call has its own, and call `Cleanup()` when the call ends.

**Inside the transport input.** `transports.NewVADAnalyzerHook(analyzer)` runs an analyzer on the input's audio and emits `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`. Use it when you enable `AudioInNative`, since it analyzes a decoded copy:

```go
func vadHookParams(socket string) transports.TransportParams {
	return transports.TransportParams{
		AudioInNative: true,
		NewAudioInAnalyzer: func() transports.AudioInputAnalyzer {
			analyzer, err := vad.NewSileroVADAnalyzer(8000, vad.DefaultVADParams(), socket)
			if err != nil {
				log.Printf("vad unavailable: %v", err)
				return nil
			}
			return transports.NewVADAnalyzerHook(analyzer)
		},
	}
}
```

The factory cannot return an error or close the analyzer when the call ends, so prefer the pipeline processor when you can.

Silero VAD accepts 8000 Hz or 16000 Hz audio only and runs in the separate `onnx-worker` process; the analyzer connects to it over a Unix socket. If the socket is missing, `NewSileroVADAnalyzer` fails with `failed to connect to onnx-worker at ...`.

## DTMF tone generation

When a serializer cannot send keypad digits as a message, the output plays tones generated by `audio/dtmf` (see [Twilio: DTMF out](telephony-twilio.md#dtmf-out)). You can also use the package directly. This program writes the tones for `123#` to a WAV file:

```go
package main

import (
	"log"

	"github.com/square-key-labs/strawgo-ai/src/audio/dtmf"
	"github.com/square-key-labs/strawgo-ai/src/audio/wav"
)

func main() {
	tones, err := dtmf.Sequence("123#", 8000)
	if err != nil {
		log.Fatal(err)
	}
	if err := wav.WriteFile("tones.wav", tones, 8000, 1); err != nil {
		log.Fatal(err)
	}
}
```

Each key is a 300 ms tone followed by a 200 ms gap, at 20 % of full scale. `dtmf.ToneWithOptions(key, rate, dtmf.Options{Tone: ..., Gap: ..., Amplitude: ...})` changes those. Keys `0`-`9`, `*`, `#` and `A`-`D` are supported by the generator; `frames.KeypadEntry` (used by DTMF frames) accepts only `0`-`9`, `*` and `#`.

## Audio classification

`classify.NewProcessor(classifier, cfg)` cuts the caller's audio into windows and runs a classifier on each one in the background. Use it to detect music, silence, or anything else a model can label. Results arrive in a callback and, unless `NoFrames` is set, as `classify.ResultFrame`s pushed downstream.

```go
func newSilenceClassifier() *classify.Processor {
	return classify.NewProcessor(classify.NewEnergyClassifier(), classify.Config{
		Window: time.Second,
		OnResult: func(result classify.Result, offset, duration time.Duration) {
			log.Printf("%v-%v: %s (%.2f)", offset, offset+duration, result.Label, result.Confidence)
		},
	})
}
```

Put it after `s.Input()`. `classify.NewEnergyClassifier()` labels each window `silence` or `sound` by level; it is a working example rather than a useful model. Plug in your own by implementing `classify.Classifier`, or wrap a function with `classify.ClassifierFunc`. The `Window.Audio` slice is reused after `Classify` returns, so copy it if you keep it. If classification is slower than real time, windows are skipped rather than queued; `Skipped()` counts them.

## Other audio helpers

The `pcm` package works on 16-bit PCM byte slices: `Mix`, `Scale`, `ToMono`, `Interleave`, `RMS`, `DBFS`, `Volume`, `IsSilence` and `FadeOut`. The `wav` package reads (`ReadFile`, `Decode`) and writes (`WriteFile`, `Encode`, and a streaming `NewWriter`) 16-bit PCM WAV files. See the [audio reference](../reference/audio.md).

## Next steps

- [Turn-taking](turn-taking.md): VAD, smart turn and interruptions.
- [Recording](recording.md): save call audio to WAV.
- [Background ambience recipe](../cookbook/background-ambience.md)
- [Audio reference](../reference/audio.md)
