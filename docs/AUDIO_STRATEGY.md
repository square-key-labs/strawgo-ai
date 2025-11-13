# Audio Processing Strategy Guide

## Two Approaches for Handling Audio

StrawGo supports **two different approaches** for handling audio in voice call pipelines. Choose the one that best fits your use case.

---

## 🚀 Option 1: Mulaw Passthrough (Recommended for Telephony)

### What It Is
Audio stays in mulaw format throughout the entire pipeline - no conversions!

### Architecture
```
Twilio (8kHz mulaw)
    ↓
Deepgram STT (mulaw encoding)
    ↓
LLM (text only)
    ↓
ElevenLabs TTS (ulaw_8000 output)
    ↓
Twilio (8kHz mulaw)
```

### Example Code
```go
// Configure Deepgram to accept mulaw directly
deepgramSTT := services.NewDeepgramSTTService(services.DeepgramSTTConfig{
    Encoding: "mulaw",  // ← No conversion needed!
})

// Configure ElevenLabs to output mulaw directly
elevenLabsTTS := services.NewElevenLabsTTSService(services.ElevenLabsTTSConfig{
    OutputFormat: "ulaw_8000",  // ← No conversion needed!
})

// Simple pipeline - no audio converters!
pipeline := pipeline.NewPipeline([]processors.FrameProcessor{
    twilioTransport.Input(),
    deepgramSTT,
    llmService,
    elevenLabsTTS,
    twilioTransport.Output(),
})
```

### ✅ Advantages
- **Best performance** - zero CPU cycles wasted on conversion
- **Lowest latency** - no conversion delays
- **Simpler pipeline** - fewer components
- **Less memory usage** - no temporary buffers for conversion

### ❌ Disadvantages
- **Limited to telephony** - only works with mulaw-compatible services
- **No audio processing** - can't add filters, mixing, VAD, etc.
- **Fixed sample rate** - stuck at 8kHz

### Use When
- ✅ Building pure telephony applications (Twilio, Asterisk, SIP)
- ✅ Performance and latency are critical
- ✅ You don't need audio processing features
- ✅ Services support mulaw natively (Deepgram ✓, ElevenLabs ✓)

### Example File
See: `examples/voice_call_mulaw.go`

---

## 🔧 Option 2: PCM Pipeline with Converters

### What It Is
Convert audio to PCM (16-bit linear) for processing, then convert back.

### Architecture
```
Twilio (8kHz mulaw)
    ↓
AudioConverter (mulaw → 16kHz PCM)
    ↓
Deepgram STT (linear16 encoding)
    ↓
LLM (text only)
    ↓
ElevenLabs TTS (pcm_24000 output)
    ↓
AudioConverter (PCM → 8kHz mulaw)
    ↓
Twilio (8kHz mulaw)
```

### Example Code
```go
// Input converter: mulaw → PCM
inputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  8000,
    InputCodec:       "mulaw",
    OutputSampleRate: 16000,
    OutputCodec:      "linear16",
})

// Configure services for PCM
deepgramSTT := services.NewDeepgramSTTService(services.DeepgramSTTConfig{
    Encoding: "linear16",  // PCM mode
})

elevenLabsTTS := services.NewElevenLabsTTSService(services.ElevenLabsTTSConfig{
    OutputFormat: "pcm_24000",  // PCM mode
})

// Output converter: PCM → mulaw
outputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
    InputSampleRate:  24000,
    InputCodec:       "linear16",
    OutputSampleRate: 8000,
    OutputCodec:      "mulaw",
})

// Pipeline with converters
pipeline := pipeline.NewPipeline([]processors.FrameProcessor{
    twilioTransport.Input(),
    inputConverter,        // ← Convert to PCM
    deepgramSTT,
    llmService,
    elevenLabsTTS,
    outputConverter,       // ← Convert back to mulaw
    twilioTransport.Output(),
})
```

### ✅ Advantages
- **Flexible** - works with all services and formats
- **Audio processing** - can add VAD, filters, mixing, etc.
- **Sample rate control** - resample as needed
- **Future-proof** - easy to add new services

### ❌ Disadvantages
- **Higher CPU usage** - conversion overhead
- **Slightly higher latency** - conversion takes time
- **More complex** - more components to manage
- **More memory** - temporary buffers needed

### Use When
- ✅ Need audio processing features (VAD, filters, mixing)
- ✅ Using non-telephony transports (WebRTC, local files)
- ✅ Need to resample audio
- ✅ Want maximum flexibility for future features

### Example File
See: `examples/voice_call_complete.go`

---

## 📊 Performance Comparison

| Metric | Mulaw Passthrough | PCM Pipeline |
|--------|------------------|--------------|
| CPU Usage | ~5% | ~15% |
| Latency | ~50ms | ~70ms |
| Memory | Low | Medium |
| Flexibility | Low | High |
| Best For | Telephony | General Purpose |

*Note: Benchmarks approximate, based on single call*

---

## 🎯 Decision Matrix

### Choose Mulaw Passthrough If:
- ✅ You're building ONLY for telephony (Twilio/Asterisk)
- ✅ Performance is your top priority
- ✅ You DON'T need: VAD, audio filters, mixing, or resampling
- ✅ All your services support mulaw (check compatibility!)

### Choose PCM Pipeline If:
- ✅ You need audio processing features
- ✅ You might use non-telephony transports later
- ✅ You want maximum flexibility
- ✅ Slightly higher latency is acceptable

---

## 🔍 Service Compatibility

### Deepgram STT
- ✅ Supports mulaw: `Encoding: "mulaw"`
- ✅ Supports PCM: `Encoding: "linear16"`
- 📝 Other: alaw, flac, opus, etc.

### ElevenLabs TTS
- ✅ Supports mulaw: `OutputFormat: "ulaw_8000"`
- ✅ Supports PCM: `OutputFormat: "pcm_16000"` (or 8k, 22k, 24k, 44k)
- 📝 Other: mp3, opus

### OpenAI LLM
- N/A - Works with text only (no audio format)

### Google Gemini LLM
- N/A - Works with text only (no audio format)

---

## 🛠️ Implementation Notes

### Transport Behavior

**Twilio/Asterisk Transports:**
- Input: Automatically decode mulaw and tag with `codec: "mulaw"` metadata
- Output: Check frame metadata for `codec`:
  - If `codec == "mulaw"`: Send directly (no conversion)
  - If `codec == "linear16"` or missing: Convert PCM → mulaw

### Service Configuration

**Deepgram:**
```go
// Mulaw mode
DeepgramSTTConfig{
    Encoding: "mulaw",
}

// PCM mode (default)
DeepgramSTTConfig{
    Encoding: "linear16",  // or omit for default
}
```

**ElevenLabs:**
```go
// Mulaw mode
ElevenLabsTTSConfig{
    OutputFormat: "ulaw_8000",
}

// PCM mode (default)
ElevenLabsTTSConfig{
    OutputFormat: "pcm_24000",  // or omit for default
}
```

### Audio Converter

**Available codecs:**
- `"mulaw"` or `"ulaw"` - μ-law compression
- `"linear16"` - 16-bit PCM (default)

**Sample rates:**
- `8000` - Telephony quality
- `16000` - Wideband quality (recommended for STT)
- `24000` - High quality (ElevenLabs default)
- `44100` - CD quality

---

## 💡 Best Practices

### For Production Telephony Apps

1. **Start with Mulaw Passthrough**
   - Simpler and faster
   - Add converters only if needed

2. **Monitor Performance**
   - Track latency and CPU usage
   - Switch to PCM only if you need features

3. **Use Connection Metadata**
   - Transport automatically propagates `connection_id` / `stream_sid`
   - Codec is tagged in frame metadata

### For Development

1. **Use PCM Pipeline Initially**
   - More flexible for experimentation
   - Easier to add debugging processors

2. **Optimize Later**
   - Profile real usage patterns
   - Switch to mulaw if performance matters

---

## 🧪 Testing

### Test Mulaw Passthrough:
```bash
go run examples/voice_call_mulaw.go
```

### Test PCM Pipeline:
```bash
go run examples/voice_call_complete.go
```

Both examples require:
- `DEEPGRAM_API_KEY`
- `ELEVENLABS_API_KEY`
- `OPENAI_API_KEY`
- `ELEVENLABS_VOICE_ID` (optional)

---

## 📚 References

- [Deepgram Encoding Options](https://developers.deepgram.com/docs/encoding)
- [ElevenLabs Audio Formats](https://elevenlabs.io/docs/api-reference/text-to-speech)
- [Pipecat Audio Handling](https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/audio)

---

## 🤝 Summary

| Approach | Best For | Performance | Flexibility |
|----------|----------|-------------|-------------|
| **Mulaw Passthrough** | Telephony-only apps | ⚡⚡⚡ Excellent | ⭐ Limited |
| **PCM Pipeline** | General purpose | ⚡⚡ Good | ⭐⭐⭐ High |

**Our Recommendation:**
- **Telephony apps**: Start with mulaw passthrough
- **Everything else**: Use PCM pipeline
- **Not sure?**: Use PCM pipeline (more flexible, easier to change later)
