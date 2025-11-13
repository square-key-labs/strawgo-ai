# ElevenLabs TTS Production-Ready Upgrade

## Overview

The ElevenLabs TTS service has been completely overhauled to align with production best practices from Pipecat and ElevenLabs official documentation. This update fixes critical issues that prevented reliable audio streaming.

## What Changed

### 1. WebSocket Endpoint Migration

**Before:**
```go
wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model}
```

**After:**
```go
wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/multi-stream-input?model_id={model}&output_format={format}
```

**Why:** The `/multi-stream-input` endpoint supports:
- Context management for handling interruptions
- Better real-time performance
- Proper audio format specification
- Multiple concurrent streams

### 2. Context ID Management

**Added:**
- Unique UUID generated for each session
- Context ID included in all WebSocket messages
- Proper context validation in responses

**Benefits:**
- Enables user interruption handling
- Prevents stale audio from old contexts
- Supports future multi-stream scenarios

### 3. Keepalive Mechanism

**Added:** Goroutine that sends keepalive messages every 10 seconds

```go
{
  "text": "",
  "context_id": "abc-123-def-456"
}
```

**Why:** Without keepalive, ElevenLabs closes idle connections after ~30 seconds, causing audio to fail during long pauses.

### 4. Audio Format Negotiation

**Added:**
- Auto-detection of incoming codec from Asterisk
- Automatic output format configuration
- Format included in WebSocket URL

**Codec Mapping:**
| Asterisk Codec | ElevenLabs Format |
|---------------|-------------------|
| `mulaw`       | `ulaw_8000`       |
| `alaw`        | `alaw_8000`       |
| `linear16`    | `pcm_16000`       |

### 5. Base64 Audio Decoding

**Fixed:** Audio comes from ElevenLabs as base64-encoded data in JSON messages:

```json
{
  "audio": "uSxK...(base64)...",
  "alignment": {...},
  "contextId": "abc-123"
}
```

**Implementation:**
- Extract `"audio"` field from JSON
- Decode base64 to raw bytes
- Create TTSAudioFrame with correct codec

### 6. Message Validation

**Added:**
- Check `isFinal` to skip end markers
- Validate `contextId` to ignore stale messages
- Proper error handling for malformed responses

### 7. Improved Flush Mechanism

**Before:**
```go
{"text": "", "flush": true}
```

**After:**
```go
{
  "text": "",
  "context_id": "abc-123-def-456",
  "flush": true
}
```

## Breaking Changes

### None for Users

The changes are **backward compatible**. Existing code using `elevenlabs.NewTTSService()` will continue to work with improved reliability.

### Internal Changes

If you were directly accessing internal fields:
- New field: `contextID string` - ElevenLabs session context
- Behavior change: Lazy initialization now default (no need to call `Initialize()`)

## Migration Guide

### No Action Required

If you're using the service normally:

```go
tts := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
    APIKey:       "your-api-key",
    VoiceID:      "voice-id",
    Model:        "eleven_turbo_v2",
    UseStreaming: true,
})
```

Everything will work automatically with improved reliability!

### Optional: Explicit Format Control

If you want to override auto-detection:

```go
tts := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
    APIKey:       "your-api-key",
    VoiceID:      "voice-id",
    Model:        "eleven_turbo_v2",
    OutputFormat: "ulaw_8000",  // Explicit format
    UseStreaming: true,
})
```

## New Features

### 1. Automatic Format Matching

The service now detects the incoming codec from Asterisk/Twilio and automatically configures ElevenLabs to match:

```
Asterisk MEDIA_START format:ulaw
  ↓
StartFrame metadata: codec="mulaw"
  ↓
ElevenLabs auto-configured: output_format="ulaw_8000"
  ↓
Perfect audio quality! 🎉
```

### 2. Connection Stability

- **Keepalive every 10 seconds** - No more timeouts
- **Automatic reconnection** (future enhancement)
- **Graceful shutdown** with proper context cleanup

### 3. Better Logging

New debug logs help troubleshoot issues:

```
[ElevenLabsTTS] Streaming mode connected (context: abc-123-def-456)
[ElevenLabsTTS] Detected incoming codec: mulaw
[ElevenLabsTTS] Auto-configured output format: ulaw_8000
[ElevenLabsTTS] Sent keepalive
[ElevenLabsTTS] Received audio chunk: 12480 bytes (decoded from base64)
[ElevenLabsTTS] Pushing TTSAudioFrame downstream (codec: mulaw, rate: 8000)
```

## Performance Improvements

### Before
- ❌ Connections timeout after 30 seconds
- ❌ Audio format mismatches cause garbled audio
- ❌ Can't handle user interruptions
- ❌ Memory leaks on server side

### After
- ✅ Stable connections with keepalive
- ✅ Perfect audio quality with format matching
- ✅ Ready for interruption handling (context management)
- ✅ Proper cleanup prevents memory leaks

## Testing Checklist

After upgrading, verify:

- [ ] Audio plays back correctly on Asterisk calls
- [ ] No timeout errors in logs
- [ ] Keepalive messages appear every 10 seconds
- [ ] Format auto-detection logs show correct codec
- [ ] Long pauses (>30 seconds) don't break audio
- [ ] Multiple calls work without issues

## Troubleshooting

### Issue: No audio output

**Check logs for:**
```
[ElevenLabsTTS] Auto-configured output format: ulaw_8000
```

If missing, the format detection failed. Set `OutputFormat` explicitly.

### Issue: Connection timeouts

**Check logs for:**
```
[ElevenLabsTTS] Sent keepalive
```

If missing, the keepalive goroutine isn't starting. Check `UseStreaming: true`.

### Issue: Garbled audio

**Check format matching:**
- Asterisk sends: `mulaw`
- ElevenLabs configured: `ulaw_8000`

If mismatch, check StartFrame metadata.

## References

- **Pipecat Implementation**: Based on production-tested Pipecat framework
- **ElevenLabs Docs**: https://elevenlabs.io/docs/api-reference/websockets
- **Multi-Stream Endpoint**: Uses modern `/multi-stream-input` API
- **Audio Formats**: Supports `ulaw_8000`, `alaw_8000`, `pcm_16000`, `pcm_22050`, `pcm_24000`, `pcm_44100`

## Credits

This implementation was aligned with best practices from:
- [Pipecat](https://github.com/pipecat-ai/pipecat) - Production voice AI framework
- ElevenLabs official WebSocket API documentation
- Real-world production testing with Asterisk telephony

## Support

For issues or questions:
1. Check logs for error messages
2. Verify audio format compatibility
3. Ensure API key is valid
4. Check network connectivity to ElevenLabs

---

**Last Updated:** November 14, 2025
**Version:** 1.0.0-production-ready
