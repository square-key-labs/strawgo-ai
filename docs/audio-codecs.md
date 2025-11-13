# Audio Codec Support in StrawGo

StrawGo provides comprehensive support for telephony audio codecs commonly used in VoIP applications, inspired by [Pipecat's codec handling](https://github.com/pipecat-ai/pipecat).

## Supported Codecs

### μ-law (G.711 μ-law) - North America & Japan
**Aliases:** `mulaw`, `ulaw`, `PCMU`

- **Sample Rate:** 8000 Hz
- **Use Case:** Telephony in North America, Japan
- **Services:** Twilio, Plivo, Telnyx (PCMU), Deepgram

```go
// Example: Converting Twilio audio (mulaw) to PCM for Deepgram
converter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  8000,
    InputCodec:       "mulaw",  // or "ulaw" or "PCMU"
    OutputSampleRate: 16000,
    OutputCodec:      "linear16",
})
```

### A-law (G.711 A-law) - Europe & Rest of World
**Aliases:** `alaw`, `PCMA`

- **Sample Rate:** 8000 Hz
- **Use Case:** Telephony in Europe and most of the world
- **Services:** Telnyx (PCMA), Deepgram

```go
// Example: Converting A-law audio to PCM
converter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  8000,
    InputCodec:       "alaw",  // or "PCMA"
    OutputSampleRate: 16000,
    OutputCodec:      "linear16",
})
```

### Linear PCM (Pulse Code Modulation)
**Aliases:** `linear16`, `pcm`, `PCM`

- **Sample Rates:** 8000, 16000, 24000 Hz (configurable)
- **Use Case:** Internal processing, AI services
- **Services:** Deepgram, ElevenLabs, OpenAI

```go
// Example: PCM at different sample rates
converter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  24000,  // ElevenLabs output
    InputCodec:       "linear16",  // or "pcm"
    OutputSampleRate: 8000,   // Twilio input
    OutputCodec:      "mulaw",
})
```

## Codec Name Variations

StrawGo automatically normalizes codec names for maximum compatibility:

| Codec | Accepted Names | Normalized To |
|-------|----------------|---------------|
| μ-law | `mulaw`, `ulaw`, `PCMU` | `mulaw` |
| A-law | `alaw`, `PCMA` | `alaw` |
| PCM | `linear16`, `pcm`, `PCM` | `linear16` |

This means you can use any variation and it will work correctly:

```go
// All of these are equivalent:
config1 := audio.AudioConverterConfig{InputCodec: "mulaw"}
config2 := audio.AudioConverterConfig{InputCodec: "ulaw"}
config3 := audio.AudioConverterConfig{InputCodec: "PCMU"}
```

## Service Codec Support

### Deepgram STT
Supports: `mulaw`/`ulaw`, `alaw`, `linear16`

```go
deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
    APIKey:   deepgramKey,
    Language: "en",
    Model:    "nova-2",
    Encoding: "mulaw",  // or "ulaw", "alaw", "linear16"
})
```

### ElevenLabs TTS
Supports: `ulaw_8000`, `pcm_16000`, `pcm_24000`

```go
elevenLabsTTS := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
    APIKey:       elevenLabsKey,
    VoiceID:      elevenLabsVoice,
    Model:        "eleven_turbo_v2",
    OutputFormat: "pcm_24000",  // or "ulaw_8000", "pcm_16000"
})
```

### Twilio Serializer
Fixed: μ-law at 8000 Hz

```go
twilioSerializer := serializers.NewTwilioFrameSerializer(streamSid, callSid)
// Automatically handles mulaw encoding/decoding
```

## Common Audio Pipeline Patterns

### Pattern 1: Twilio → Deepgram → OpenAI → ElevenLabs → Twilio

```go
// Input converter: Twilio mulaw 8kHz -> PCM 16kHz for Deepgram
inputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  8000,
    InputCodec:       "mulaw",
    OutputSampleRate: 16000,
    OutputCodec:      "linear16",
})

// Output converter: ElevenLabs PCM 24kHz -> Twilio mulaw 8kHz
outputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  24000,
    InputCodec:       "linear16",
    OutputSampleRate: 8000,
    OutputCodec:      "mulaw",
})

pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    transport.Input(),    // Twilio mulaw 8kHz
    inputConverter,       // -> PCM 16kHz
    deepgramSTT,          // PCM 16kHz -> text
    openaiLLM,            // text -> text
    elevenLabsTTS,        // text -> PCM 24kHz
    outputConverter,      // -> mulaw 8kHz
    transport.Output(),   // Twilio mulaw 8kHz
})
```

### Pattern 2: Telnyx PCMA → Deepgram → Telnyx PCMA

```go
// Input converter: Telnyx A-law 8kHz -> PCM 16kHz
inputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  8000,
    InputCodec:       "PCMA",  // A-law
    OutputSampleRate: 16000,
    OutputCodec:      "linear16",
})

// Output converter: PCM 24kHz -> A-law 8kHz
outputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  24000,
    InputCodec:       "linear16",
    OutputSampleRate: 8000,
    OutputCodec:      "PCMA",  // A-law
})
```

## Direct Codec Functions

For advanced use cases, you can use codec functions directly:

```go
import "github.com/square-key-labs/strawgo-ai/src/audio"

// μ-law encoding/decoding
mulawData := audio.PCMToMulaw(pcmSamples)
pcmSamples := audio.MulawToPCM(mulawData)

// A-law encoding/decoding
alawData := audio.PCMToAlaw(pcmSamples)
pcmSamples := audio.AlawToPCM(alawData)

// PCM byte conversion
pcmBytes := audio.PCMToBytes(pcmSamples)
pcmSamples, err := audio.BytesToPCM(pcmBytes)

// Resampling
resampled := audio.Resample(pcmSamples, 8000, 16000)
```

## Sample Rates

| Rate | Use Case | Common Services |
|------|----------|-----------------|
| 8000 Hz | Telephony standard | Twilio, Plivo, Telnyx |
| 16000 Hz | Wideband speech, STT | Deepgram, most STT services |
| 24000 Hz | High quality speech, TTS | ElevenLabs, OpenAI TTS |
| 48000 Hz | Studio quality | Professional audio |

## Regional Codec Usage

### North America & Japan
- **Standard:** μ-law (PCMU)
- **Services:** Twilio, Plivo, most US-based carriers

### Europe & Rest of World
- **Standard:** A-law (PCMA)
- **Services:** European carriers, Telnyx

### Global Services
Use codec negotiation:
```go
// Telnyx example with configurable codec
telnyxSerializer := serializers.NewTelnyxFrameSerializer(
    streamID,
    "PCMA",  // outbound encoding (Europe)
    "PCMU",  // inbound encoding (can be different)
)
```

## Codec Quality Comparison

| Codec | Bit Rate | Quality | Latency | CPU Usage |
|-------|----------|---------|---------|-----------|
| μ-law | 64 kbps | Telephony | Very Low | Very Low |
| A-law | 64 kbps | Telephony | Very Low | Very Low |
| PCM 16kHz | 256 kbps | Good | Low | Low |
| PCM 24kHz | 384 kbps | High | Low | Low |

**Note:** μ-law and A-law are lossy codecs optimized for voice transmission over limited bandwidth. They provide good intelligibility but are not suitable for music or high-fidelity audio.

## Troubleshooting

### Issue: Audio sounds distorted after conversion

**Solution:** Check sample rate matching:
```go
// ❌ Wrong: Mismatched sample rates
converter := audio.AudioConverterConfig{
    InputSampleRate: 16000,  // Wrong!
    InputCodec:      "mulaw",
    // mulaw is typically 8000 Hz
}

// ✅ Correct: Proper sample rate
converter := audio.AudioConverterConfig{
    InputSampleRate: 8000,   // Correct for mulaw
    InputCodec:      "mulaw",
}
```

### Issue: Codec not supported error

**Solution:** Check codec name spelling and use aliases:
```go
// ❌ Wrong
InputCodec: "ulaw8000"  // Not a valid codec name

// ✅ Correct - use normalized names
InputCodec: "ulaw"      // or "mulaw" or "PCMU"
```

### Issue: Service expects different codec format

**Solution:** Check service documentation for codec naming:
```go
// Deepgram uses standard names
Encoding: "mulaw"  // ✓

// ElevenLabs uses format_samplerate
OutputFormat: "ulaw_8000"  // ✓

// Telnyx uses ITU names
InboundEncoding: "PCMU"   // ✓ (for μ-law)
OutboundEncoding: "PCMA"  // ✓ (for A-law)
```

## Testing Codec Support

Run the codec test to verify all conversions work:

```bash
go run examples/codec_test.go
```

Or create your own test:
```go
package main

import (
    "github.com/square-key-labs/strawgo-ai/src/audio"
)

func main() {
    // Test μ-law
    pcm := []int16{0, 8192, 16384, 24576, 32767}
    mulaw := audio.PCMToMulaw(pcm)
    decoded := audio.MulawToPCM(mulaw)

    // Test A-law
    alaw := audio.PCMToAlaw(pcm)
    decoded2 := audio.AlawToPCM(alaw)
}
```

## Further Reading

- [G.711 Standard (Wikipedia)](https://en.wikipedia.org/wiki/G.711)
- [Pipecat Audio Utils](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/audio/utils.py)
- [Twilio Media Streams](https://www.twilio.com/docs/voice/twiml/stream)
- [Telnyx WebSocket API](https://developers.telnyx.com/docs/api/v2/call-control/Streaming)
- [Deepgram Encoding Options](https://developers.deepgram.com/docs/encoding)
