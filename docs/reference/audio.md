# Audio reference

This page documents Strawgo's audio packages: codec helpers and the converter processor in `audio`, and the `resample`, `g711`, `pcm`, `wav`, `dtmf`, `mixer`, `filter` and `classify` subpackages.

**PCM** is raw, uncompressed audio; Strawgo's internal format is 16-bit signed little-endian mono PCM, called `linear16`. The **sample rate** is how many samples per second the audio has (8000 Hz for phone calls, 16000 or 24000 for speech services). **G.711** is the 8-bit telephone codec, in two flavours: mu-law (also called ulaw or PCMU) and A-law (PCMA). **Resampling** converts audio from one sample rate to another. **DTMF** tones are the beeps of a phone keypad.

Import paths: `github.com/square-key-labs/strawgo-ai/src/audio` and `.../src/audio/<package>`.

For when and why to use these, read the [audio guide](../guides/audio.md). Voice activity detection and smart turn are in [turns](turns.md).

## Quick example

This function turns one second of 8 kHz mu-law phone audio into 16 kHz PCM, measures its level, and appends a DTMF `1` tone.

```go
func prepare(mulaw []byte) ([]byte, float64, error) {
	pcm8k := g711.UlawToPCM16(nil, mulaw)
	pcm16k, err := resample.ResampleBytes(pcm8k, 8000, 16000)
	if err != nil {
		return nil, 0, err
	}
	level := pcm.DBFS(pcm16k)
	withTone, err := dtmf.AppendTone(pcm16k, '1', 16000)
	if err != nil {
		return nil, 0, err
	}
	return withTone, level, nil
}
```

## audio: codecs and helpers

| Constant | Value |
| --- | --- |
| `CodecLinear16` | `"linear16"` |
| `CodecMulaw` | `"mulaw"` |
| `CodecAlaw` | `"alaw"` |

| Function | Signature | Description |
| --- | --- | --- |
| `NormalizeCodec` | `func NormalizeCodec(codec string) string` | Map aliases to the constants: `ulaw`, `pcmu`, `g711_ulaw`, `audio/x-mulaw`, `mu-law`, `audio/pcmu` give `mulaw`; `pcma`, `g711_alaw`, `audio/x-alaw`, `a-law`, `audio/pcma` give `alaw`; `""`, `pcm`, `l16`, `slin`, `pcm16`, `s16le`, `audio/l16` give `linear16`. Case-insensitive. Unknown values are returned unchanged. |
| `DecodePCM16` | `func DecodePCM16(dst, src []byte, codec string) ([]byte, error)` | Decode `src` in `codec` to 16-bit PCM, appended to `dst`. |
| `EncodePCM16` | `func EncodePCM16(dst, pcm []byte, codec string) ([]byte, error)` | Encode 16-bit PCM to `codec`, appended to `dst`. |
| `BytesPerSample` | `func BytesPerSample(codec string) int` | 1 for G.711, 2 for `linear16`. |
| `SilenceByte` | `func SilenceByte(codec string) byte` | The byte value of silence: `0xFF` for mu-law, `0xD5` for A-law, `0` for PCM. |
| `Resample` | `func Resample(input []int16, inputRate, outputRate int) []int16` | One-shot resample of samples. |
| `BytesToPCM`, `PCMToBytes` | `func BytesToPCM(data []byte) ([]int16, error)`, `func PCMToBytes(samples []int16) []byte` | Convert between bytes and samples. |
| `ClipAudio` | `func ClipAudio(pcm []int16, maxValue int16) []int16` | Clamp samples to `±maxValue` into a new slice. |
| `NormalizeAudio` | `func NormalizeAudio(pcm []int16, targetRMS float64) []int16` | Scale to a target RMS. |

`audio.Resampler` is a zero-value-ready streaming resampler for byte buffers, used by transports and serializers:

| Method | Description |
| --- | --- |
| `Resample(dst, pcm []byte, inRate, outRate int) []byte` | Resample a chunk, keeping filter state between calls. Equal or invalid rates copy the input. |
| `Flush(dst []byte) []byte` | Emit the samples still held in the filter. |
| `Reset()` | Drop state, for example after an interruption. |
| `MaxOutputBytes(inBytes, inRate, outRate int) int` | Upper bound on output size, for preallocating `dst`. |

This decodes a carrier payload to 16-bit PCM, whatever name the carrier uses for its codec (`"PCMU"`, `"ulaw"`, `"audio/x-mulaw"` and so on):

```go
func toPCM(payload []byte, carrierCodec string) ([]byte, error) {
	return audio.DecodePCM16(nil, payload, audio.NormalizeCodec(carrierCodec))
}
```

### Converter

```text
func NewAudioConverterProcessor(config AudioConverterConfig) *AudioConverterProcessor
```

A processor that converts `AudioFrame` and `TTSAudioFrame` data to one codec and rate.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `InputSampleRate` | `int` | the frame's rate | Used when a frame has no sample rate. |
| `InputCodec` | `string` | `linear16` | Used when a frame has no `"codec"` metadata. |
| `OutputSampleRate` | `int` | the input rate | Target rate. |
| `OutputCodec` | `string` | `linear16` | Target codec. |

Converted frames are new frames with the same channel count and metadata; `"codec"` is set to the output codec and the old value is kept as `"original_codec"`. `TTSAudioFrame.ContextID` is preserved. Frames already in the target format pass unchanged. A conversion error is pushed as a non-fatal `ErrorFrame`. An `InterruptionFrame` resets the TTS resampler.

## resample

A polyphase FIR resampler for 16-bit PCM.

| Item | Signature | Description |
| --- | --- | --- |
| `New` | `func New(inRate, outRate int) (*Resampler, error)` | Errors when either rate is not positive. Equal rates make a pass-through resampler. |
| `Process` | `func (r *Resampler) Process(dst, src []int16) []int16` | Resample a chunk, appending to `dst`. State carries over, so chunks join without clicks. |
| `ProcessBytes` | `func (r *Resampler) ProcessBytes(dst, src []byte) []byte` | Same on little-endian bytes. |
| `Flush` | `func (r *Resampler) Flush(dst []int16) []int16` | Emit the tail held in the filter. |
| `Reset` | `func (r *Resampler) Reset()` | Clear state. |
| `MaxOutput` | `func (r *Resampler) MaxOutput(inSamples int) int` | Upper bound on output samples. |
| `InRate`, `OutRate` | `func (r *Resampler) InRate() int` | The configured rates. |
| `Resample` | `func Resample(src []int16, inRate, outRate int) ([]int16, error)` | One-shot, including the flush. |
| `ResampleBytes` | `func ResampleBytes(src []byte, inRate, outRate int) ([]byte, error)` | One-shot on bytes. |

For a stream of chunks, keep one `Resampler` per stream so the filter state carries over between chunks:

```go
func newUpsampler() (func(chunk []byte) []byte, error) {
	r, err := resample.New(8000, 16000)
	if err != nil {
		return nil, err
	}
	return func(chunk []byte) []byte {
		return r.ProcessBytes(nil, chunk)
	}, nil
}
```

## g711

| Function | Description |
| --- | --- |
| `UlawToPCM16(dst, src []byte) []byte`, `AlawToPCM16(dst, src []byte) []byte` | Decode bytes to 16-bit PCM bytes, appended to `dst`. |
| `PCM16ToUlaw(dst, src []byte) []byte`, `PCM16ToAlaw(dst, src []byte) []byte` | Encode 16-bit PCM bytes. |
| `DecodeUlaw(dst []int16, src []byte) []int16`, `DecodeAlaw(dst []int16, src []byte) []int16` | Decode to samples. |
| `EncodeUlaw(dst []byte, src []int16) []byte`, `EncodeAlaw(dst []byte, src []int16) []byte` | Encode samples. |
| `UlawDecodeSample(u byte) int16`, `UlawEncodeSample(s int16) byte`, `AlawDecodeSample(a byte) int16`, `AlawEncodeSample(s int16) byte` | One sample. |

This transcodes A-law to mu-law through 16-bit PCM:

```go
func alawToUlaw(alaw []byte) []byte {
	return g711.PCM16ToUlaw(nil, g711.AlawToPCM16(nil, alaw))
}
```

## pcm

Helpers for 16-bit little-endian PCM byte slices.

| Function | Description |
| --- | --- |
| `Grow(s []byte, n int) []byte` | Extend `s` by `n` bytes, reallocating if needed. |
| `AppendSilence(dst []byte, n int) []byte` | Append `n` zero bytes. |
| `ToInt16(dst []int16, src []byte) []int16`, `FromInt16(dst []byte, src []int16) []byte` | Convert between bytes and samples. |
| `Mix(dst, a, b []byte) []byte` | Append the sum of `a` and `b` (clamped), as long as the longer one. |
| `MixInPlace(dst, src []byte, gain float64)` | Add `src × gain` into `dst`. |
| `Scale(dst, src []byte, factor float64) []byte` | Append `src × factor`. |
| `Interleave(dst, left, right []byte) []byte` | Make stereo from two mono buffers. |
| `ToMono(dst, src []byte, channels int) []byte` | Average channels to mono. |
| `Peak(b []byte) int` | Largest absolute sample. |
| `RMS(b []byte) float64`, `DBFS(b []byte) float64`, `ToDBFS(rms float64) float64` | Level: RMS as a fraction of full scale (0 to 1), and the same in dBFS, never below `MinDBFS` (-96). |
| `Volume(b []byte) float64` | Level from -60 dBFS to 0 mapped to 0 to 1. |
| `IsSilence(b []byte) bool`, `IsSilenceThreshold(b []byte, threshold int) bool` | Peak at or below `SpeakingThreshold` (20), or your threshold. |
| `FadeOut(b []byte)` | Linear fade to zero over the buffer. |
| `NormalizeValue(v, lo, hi float64) float64`, `ExpSmoothing(value, prev, factor float64) float64` | Small math helpers. |

This labels a chunk of 16-bit PCM as silence or gives its level:

```go
func describeChunk(chunk []byte) string {
	if pcm.IsSilence(chunk) {
		return "silence"
	}
	return fmt.Sprintf("%.1f dBFS", pcm.DBFS(chunk))
}
```

## wav

| Item | Signature | Description |
| --- | --- | --- |
| `Encode` | `func Encode(pcm []byte, sampleRate, channels int) []byte` | A complete WAV file in memory. |
| `Header` | `func Header(dataLen, sampleRate, channels int) [44]byte` | A 44-byte 16-bit PCM header. |
| `Write`, `WriteFile` | `func Write(w io.Writer, pcm []byte, sampleRate, channels int) error` | Write a WAV to a writer or a path. |
| `Decode`, `ReadFile` | `func Decode(r io.Reader) (*Audio, error)` | Read a 16-bit PCM WAV into `Audio{Data, SampleRate, Channels}`. Other formats return an error wrapping `ErrFormat`. |
| `NewWriter` | `func NewWriter(ws io.WriteSeeker, sampleRate, channels int) (*Writer, error)` | Streaming writer: `Write(pcm)` appends samples; `Close()` patches the header sizes. |

This saves 16 kHz mono PCM, for example a recorded turn, as a WAV file:

```go
func saveTurn(path string, pcm16k []byte) error {
	return wav.WriteFile(path, pcm16k, 16000, 1)
}
```

## dtmf

Generates keypad tones as 16-bit PCM.

| Item | Signature | Description |
| --- | --- | --- |
| `Tone` | `func Tone(key byte, sampleRate int) ([]byte, error)` | One tone with default options. |
| `ToneWithOptions` | `func ToneWithOptions(key byte, sampleRate int, opts Options) ([]byte, error)` | One tone with your options. |
| `AppendTone` | `func AppendTone(dst []byte, key byte, sampleRate int) ([]byte, error)` | Append a tone and its gap to `dst`. |
| `Generate` | `func Generate(dst []byte, key byte, sampleRate int, opts Options) ([]byte, error)` | Append with options. |
| `Sequence` | `func Sequence(keys string, sampleRate int) ([]byte, error)` | Tones for every key in `keys`. |
| `Frequencies` | `func Frequencies(key byte) (low, high float64, err error)` | The two frequencies of a key. |
| `Valid` | `func Valid(key byte) bool` | Whether `key` is `0`-`9`, `*`, `#` or `A`-`D` (lower case accepted). |

`Options`:

| Field | Type | Default (`DefaultOptions()`) | Meaning |
| --- | --- | --- | --- |
| `Tone` | `time.Duration` | 300 ms | Tone length. |
| `Gap` | `time.Duration` | 200 ms | Silence after the tone. |
| `Amplitude` | `float64` | 0.2 | Peak level, 0 to 1. |

To send DTMF on a call, push an `OutputDTMFFrame` instead; the transport chooses the method. See [frames](frames.md#dtmf).

This generates the tones for an extension followed by `#`, at the phone rate:

```go
func extensionTones(ext string) ([]byte, error) {
	return dtmf.Sequence(ext+"#", 8000)
}
```

## mixer

```text
type Mixer interface {
	Start(sampleRate int) error
	Stop() error
	ProcessFrame(frame frames.Frame) error
	Mix(dst, audio []byte) []byte
}
func NewSoundfileMixer(soundFiles map[string]string, defaultSound string, opts ...SoundfileOption) *SoundfileMixer
```

A **mixer** blends background audio into the bot's output. Set one in `TransportParams.NewOutputMixer` or `webrtc.Params.AudioOutMixer`. The transport calls `Start` with the output rate and `Mix` for every outgoing chunk, including silence.

`SoundfileMixer` plays WAV files. `soundFiles` maps names to file paths; `defaultSound` is the name to start with. Files are loaded, downmixed and resampled in `Start`; a file that fails to load is logged and skipped (`Start` fails only when none load).

| Option | Default | Effect |
| --- | --- | --- |
| `WithVolume(v float64)` | 0.4 | Gain applied to the background. |
| `WithLoop(loop bool)` | true | Loop the sound; false plays it once. |
| `WithMixing(on bool)` | true | Whether mixing starts enabled. |

Runtime control: `MixerEnableFrame{Enable}` toggles mixing; `MixerUpdateSettingsFrame{Settings}` accepts `"sound"` (string, a key of `soundFiles`; restarts from the beginning), `"volume"` (number) and `"loop"` (bool).

Build the mixer in `NewOutputMixer` so that every call gets its own instance and playback position:

```go
func ambienceParams() transports.TransportParams {
	return transports.TransportParams{
		NewOutputMixer: func() mixer.Mixer {
			return mixer.NewSoundfileMixer(map[string]string{"office": "office.wav"}, "office", mixer.WithVolume(0.2))
		},
	}
}
```

## filter

```text
type Filter interface {
	Start(sampleRate int) error
	Stop() error
	ProcessFrame(frame frames.Frame) error
	Filter(dst, audio []byte) []byte
}
```

A **filter** cleans up caller audio before it reaches VAD and STT. Set one in `TransportParams.NewInputFilter`. The transport starts it with the input rate and runs each caller chunk through `Filter`.

| Type | Constructor | Description |
| --- | --- | --- |
| `NoiseGate` | `NewNoiseGate(params NoiseGateParams)` | Attenuates audio below a threshold. `SetEnabled`, `SetParams`, `Params` change it in code. |
| `Chain` | `NewChain(filters ...Filter)` | Runs filters in order; starts, stops and passes frames to each. |
| `FuncFilter` | `NewFuncFilter(fn Func)` | Wraps `func(dst, audio []byte) []byte`; ignores frames. |

`NoiseGateParams` (`DefaultNoiseGateParams()` values):

| Field | Default | Meaning |
| --- | --- | --- |
| `ThresholdDB` | -45 | Level, in dBFS, above which the gate opens. |
| `AttackMs` | 2 | Time to open fully. |
| `HoldMs` | 150 | Time the gate stays open after the level drops. |
| `ReleaseMs` | 80 | Time to close. |
| `FloorDB` | -80 | Attenuation applied while closed. |

Runtime control: `FilterEnableFrame{Enable}`; `FilterUpdateSettingsFrame{Settings}` with numeric keys `threshold_db`, `attack_ms`, `hold_ms`, `release_ms`, `floor_db`. Unknown keys return an error.

This gives every call its own noise gate, a little more sensitive than the default:

```go
func gatedParams() transports.TransportParams {
	return transports.TransportParams{
		NewInputFilter: func() filter.Filter {
			p := filter.DefaultNoiseGateParams()
			p.ThresholdDB = -50
			return filter.NewNoiseGate(p)
		},
	}
}
```

## classify

```text
func NewProcessor(classifier Classifier, cfg Config) *Processor
type Classifier interface {
	Classify(ctx context.Context, window Window) (Result, error)
}
```

A processor that runs a classifier over sliding windows of caller audio without delaying the audio. Use it to label stretches of audio, for example silence versus sound.

| `Config` field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Name` | `string` | `"AudioClassifier"` | Processor name, also `ResultFrame.Classifier`. |
| `Window` | `time.Duration` | 1 s (`DefaultWindow`) | Audio per classification. |
| `Hop` | `time.Duration` | `Window` | Step between windows; values above `Window` become `Window`. |
| `OnResult` | `func(result Result, offset, duration time.Duration)` | none | Called for each result. |
| `NoFrames` | `bool` | false | Do not push `ResultFrame`s. |

Classification runs on a worker goroutine. If the classifier is still busy, new windows are skipped and counted by `Skipped()`. Errors are logged; a panic is turned into a `*PanicError`. Each result is pushed downstream as a `ResultFrame` (a data frame with `Result`, `Classifier`, `Offset`, `Duration`). `Window.Audio` is reused after `Classify` returns: copy it to keep it.

| Item | Description |
| --- | --- |
| `Result` | `Label string`, `Confidence float64`. |
| `ClassifierFunc` | Adapts a function to `Classifier`. |
| `EnergyClassifier` (`NewEnergyClassifier()`) | Labels a window `LabelSilence` or `LabelSound` by level; `SilenceDBFS` defaults to -45 (`DefaultSilenceDBFS`). |

This logs, every half second, whether the caller's line carries sound, using the built-in energy classifier. Put the processor after the session input:

```go
func lineMonitor() *classify.Processor {
	return classify.NewProcessor(classify.NewEnergyClassifier(), classify.Config{
		Window:   500 * time.Millisecond,
		NoFrames: true,
		OnResult: func(r classify.Result, offset, _ time.Duration) {
			log.Printf("%v: %s", offset, r.Label)
		},
	})
}
```

## Next steps

- [Audio guide](../guides/audio.md): sample rates, resampling, mixers and filters in practice.
- [Background ambience recipe](../cookbook/background-ambience.md): a mixer end to end.
- [Transports reference](transports.md#transportparams): where mixers, filters and analyzers plug in.
- [Turns reference](turns.md#vad): VAD and smart turn.
