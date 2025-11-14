# Changelog

All notable changes to StrawGo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.4] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed audio mixing bug when interruption conditions not met
  - UserAggregator now consumes TranscriptionFrames instead of passing them downstream
  - Prevents LLM from receiving transcriptions when interruption conditions are not met
  - Fixes audio mixing issue where new TTS audio played over existing audio
  - Location: `src/processors/aggregators/user.go:153-155`

### Changed
- **BREAKING**: LLM services now only accept LLMContextFrame (removed backward compatibility)
  - Removed TranscriptionFrame handler from Gemini LLM service
  - Removed TranscriptionFrame handler from OpenAI LLM service
  - All LLM services must now be used with aggregators (UserAggregator)
  - Enforces proper pipeline architecture: STT → UserAggregator → LLMContextFrame → LLM
  - Locations:
    - `src/services/gemini/llm.go:86-113`
    - `src/services/openai/llm.go:86-113`

### Removed
- Dead code cleanup after backward compatibility removal
  - Removed unused `generateResponse()` function from OpenAI LLM service (98 lines)
  - Function became unreachable after removing TranscriptionFrame handling
  - All functionality preserved via `generateResponseFromContext()` method
  - Location: `src/services/openai/llm.go`

### Technical Details
- **Root Cause**: TranscriptionFrames were being passed downstream even when input was discarded
  - UserAggregator logged "discarding input" but still pushed frames downstream
  - LLM service's backward compatibility mode processed these frames anyway
  - Result: New LLM responses generated and mixed with currently playing audio

- **Fix Strategy**: Enforce single responsibility principle
  - UserAggregator: Consumes TranscriptionFrames, emits LLMContextFrames
  - LLM Services: Only process LLMContextFrames (no direct transcription handling)
  - Pipeline integrity: Each processor has clear input/output contracts

- **Migration Impact**: Applications not using aggregators will break
  - Required: Use UserAggregator between STT and LLM services
  - See: `examples/voice_call_complete.go` for proper pipeline setup

## [0.0.3] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed text duplication in LLMUserAggregator when handling interim and final transcriptions
  - Interim transcriptions are no longer appended to aggregation (following pipecat pattern)
  - Only final transcriptions are now added to the aggregation buffer
  - Prevents duplicate text being sent to LLM (e.g., "how are you doing how are you doing")
  - Removed unused `lastInterimText` field and complex replacement logic
  - Location: `src/processors/aggregators/user.go:115-151`

## [0.0.2] - 2025-11-14

### Fixed
- **CRITICAL**: Fixed TTSStartedFrame being emitted multiple times (once per text chunk)
  - Implemented pipecat's boolean flag pattern with `isSpeaking` state tracking
  - Added `sync.Mutex` for thread-safe concurrent access
  - Location: `src/services/elevenlabs/tts.go`

- **CRITICAL**: Fixed frame direction for TTS state tracking frames
  - Changed TTSStartedFrame/TTSStoppedFrame to push UPSTREAM instead of downstream
  - UserAggregator can now correctly track `botSpeaking` state
  - Without this fix, interruptions would not work at all
  - Location: `src/services/elevenlabs/tts.go`

- Added InterruptionFrame handler to TTS service
  - Resets `isSpeaking` flag when interrupted
  - Emits TTSStoppedFrame upstream on interruption

- Added proper state reset on LLMFullResponseEndFrame
  - Non-streaming mode resets immediately
  - Streaming mode resets when `isFinal` received from ElevenLabs

### Changed
- Updated aggregators README to document critical bug fixes
- Updated IMPLEMENTATION_COMPLETE.md with detailed bug analysis

## [0.0.1] - 2025-11-14

### Added
- **LLM Context Aggregators System**
  - Base LLMContextAggregator with shared functionality
  - LLMUserAggregator for user input accumulation and interruption logic
  - LLMAssistantAggregator for LLM response tracking and context updates
  - Background aggregation task with timeout handling
  - Location: `src/processors/aggregators/`

- **Intelligent Interruption System**
  - InterruptionTaskFrame (upstream) → InterruptionFrame (downstream) flow
  - Strategy-based interruption decisions via InterruptionStrategy interface
  - MinWordsInterruptionStrategy (configurable word threshold)
  - Bot speaking state tracking via TTS frames
  - Automatic input discard when interruption conditions not met

- **Enhanced LLM Context**
  - Tools and ToolChoice support for function calling
  - ToolCall and FunctionCall structures
  - Multi-turn conversation state management
  - Location: `src/services/service.go`

- **New Frame Types**
  - LLMContextFrame - Carries conversation context to LLM
  - LLMMessagesAppendFrame - Append messages to context
  - LLMMessagesUpdateFrame - Replace entire message history
  - FunctionCallsStartedFrame - Function call initiated
  - FunctionCallInProgressFrame - Function call executing
  - FunctionCallResultFrame - Function call completed
  - FunctionCallCancelFrame - Function call cancelled
  - Location: `src/frames/control.go`

- **LLM Service Enhancements**
  - LLMContextFrame handler in OpenAI service
  - Function calling support (tools, tool_choice, function calls)
  - Backward compatibility with existing TextFrame handling
  - Location: `src/services/openai/llm.go`

- **TTS State Tracking**
  - TTSStartedFrame emission at synthesis start
  - TTSStoppedFrame emission at synthesis end
  - State tracking for interruption support
  - Location: `src/services/elevenlabs/tts.go`

- **Documentation**
  - Comprehensive aggregators guide (`docs/AGGREGATORS.md` - 523 lines)
  - Implementation plan (`AGGREGATOR_IMPLEMENTATION_PLAN.md` - 750+ lines)
  - Standalone interruption demo (`examples/aggregators_with_interruptions.go` - 254 lines)
  - Complete API reference and troubleshooting guide

### Changed
- Updated `examples/voice_call_complete.go` to use aggregators
  - Configured interruption support with MinWordsInterruptionStrategy
  - Integrated LLMUserAggregator and LLMAssistantAggregator
  - Enabled intelligent interruptions with 3-word threshold

### Technical Details
- Total lines of code added: ~964 lines
- New components: 3 aggregator files
- Enhanced components: 5 existing files
- Documentation: 2 comprehensive docs + 1 implementation plan
- Examples: 2 working examples
- Compilation: ✅ All code verified
- Test coverage: Complete flow verified

### Acknowledgments
Implementation based on the excellent **Pipecat** framework architecture.
All core concepts and patterns adapted from pipecat's aggregator system.
Reference: `.local_context/pipecat/processors/aggregators/`

---

## Version History

### Version Numbering (Semantic Versioning)
- **0.x.x** - Pre-release versions (breaking changes may occur)
- **MAJOR** version: Incompatible API changes (1.0.0+)
- **MINOR** version: New functionality (backward compatible)
- **PATCH** version: Bug fixes (backward compatible)

### Current Version: 0.0.4
- Status: ✅ Alpha - Feature Complete
- Release Date: 2025-11-14
- All known bugs: Fixed
- Test Status: Verified

### Roadmap to 1.0.0
- [ ] Production testing in live environments
- [ ] Performance benchmarking
- [ ] API stability period (no breaking changes)
- [ ] Full test suite coverage

[Unreleased]: https://github.com/square-key-labs/strawgo-ai/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/square-key-labs/strawgo-ai/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/square-key-labs/strawgo-ai/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/square-key-labs/strawgo-ai/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/square-key-labs/strawgo-ai/releases/tag/v0.0.1
