# Known Issues and Improvements

## Critical Issues ⚠️

### 1. Audio Format Conversion Missing
**Problem**: Audio format mismatch between services
- Asterisk/Twilio use 8kHz mulaw
- Deepgram expects 16kHz linear16
- ElevenLabs outputs 24kHz MP3

**Impact**: Audio will not work correctly without format conversion

**Solution Needed**:
- Add audio resampling (8kHz → 16kHz for STT)
- Add codec conversion (mulaw → linear16)
- Add format conversion for TTS output (MP3 → mulaw)

### 2. Processor Start/Stop Not Called in Transports
**Problem**: Transport input/output processors are created but never started
**Location**: `transports/asterisk_websocket.go`, `transports/twilio_websocket.go`

**Impact**: Processors won't process frames properly

**Solution**: The processors are added to the pipeline, so they get started by the pipeline. This is actually correct, but should be documented.

### 3. Service Initialization Not Automatic ✅ FIXED
**Problem**: Services require manual initialization before pipeline starts
**Location**: All example files call `service.Initialize()` separately

**Impact**: Easy to forget to initialize services

**Solution**: ✅ **FIXED** - Services now auto-initialize when they receive a StartFrame. Each service's HandleFrame() method checks for StartFrame and calls Initialize() if needed. This happens automatically when the pipeline starts.

## Medium Priority Issues 🔶

### 4. No Audio Buffering
**Problem**: No buffering for audio chunks, may cause issues with variable-sized packets
**Location**: All transport implementations

**Solution Needed**: Add buffering layer for audio data

### 5. No VAD (Voice Activity Detection)
**Problem**: No way to detect when user starts/stops speaking
**Impact**: Cannot generate UserStartedSpeaking/UserStoppedSpeaking frames

**Solution**: Integrate Silero VAD or similar

### 6. Connection ID Propagation
**Problem**: Connection metadata must be manually copied between frames
**Location**: Transport output processors

**Solution**: Consider automatic metadata propagation through pipeline

### 7. No Context Aggregators
**Problem**: Transcriptions must be manually added to LLM context
**Location**: LLM services

**Solution**: Add aggregator processors that automatically collect messages

### 8. Error Recovery
**Problem**: No automatic retry or recovery from transient errors
**Location**: All services

**Solution**: Add retry logic with exponential backoff

## Low Priority Issues 📝

### 9. No Metrics/Observability
**Problem**: No way to monitor pipeline performance
**Solution**: Add metrics collection (frame counts, latency, etc.)

### 10. No Function Calling
**Problem**: LLM services don't support function calling yet
**Location**: `services/openai_llm.go`, `services/gemini_llm.go`

**Solution**: Add function registry and execution

### 11. Hardcoded Buffer Sizes
**Problem**: Channel buffer sizes are hardcoded
**Location**: `processors/processor.go` (line 50-51)

**Solution**: Make buffer sizes configurable

### 12. No Graceful Degradation
**Problem**: If one service fails, entire pipeline stops
**Solution**: Add circuit breaker pattern

## Documentation Issues 📚

### 13. Missing API Documentation
**Problem**: No godoc comments on many exported types
**Solution**: Add comprehensive godoc comments

### 14. No Architecture Diagrams
**Problem**: Only text-based diagrams in README
**Solution**: Add visual architecture diagrams

## Security Concerns 🔒

### 15. CORS Wide Open
**Problem**: WebSocket upgrader allows all origins
**Location**: Both transport files (CheckOrigin returns true)

**Solution**: Implement proper origin checking

### 16. No API Key Validation
**Problem**: No validation that API keys are set before use
**Location**: All service files

**Solution**: Add validation in Initialize()

### 17. No Rate Limiting
**Problem**: No protection against excessive API calls
**Solution**: Add rate limiting for AI services

## Testing 🧪

### 18. No Unit Tests
**Problem**: No test files exist
**Solution**: Add comprehensive unit tests

### 19. No Integration Tests
**Problem**: No automated tests for full pipeline
**Solution**: Add integration tests with mocked services

## Performance 🚀

### 20. No Connection Pooling
**Problem**: Services create new connections for each request (HTTP mode)
**Solution**: Use connection pooling for HTTP clients

### 21. Potential Memory Leak
**Problem**: Audio data in frames is not explicitly freed
**Solution**: Consider using sync.Pool for audio buffers

## What Works Well ✅

1. Core frame system architecture
2. Priority queue handling
3. Pipeline composition
4. Bidirectional flow
5. WebSocket integrations
6. Streaming support
7. Multiple LLM options
8. Clean separation of concerns
9. Go idiomatic patterns
10. Graceful shutdown handling
