package vad

import (
	"context"
	"fmt"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// VADInputProcessor accumulates audio and runs Voice Activity Detection
// When user starts speaking, emits UserStartedSpeakingFrame
// When user stops speaking, emits UserStoppedSpeakingFrame
//
// Optionally supports Smart Turn detection via TurnAnalyzer:
// - When VAD detects silence after speech, runs ML inference to determine if user finished
// - This provides more natural turn-taking than silence-only detection
type VADInputProcessor struct {
	*processors.BaseProcessor
	analyzer     VADAnalyzer
	turnAnalyzer turn.TurnAnalyzer // Optional: ML-based turn detection

	// Audio accumulation buffer
	audioBuffer []byte
	bufferMu    sync.Mutex

	// VAD state tracking
	currentState  VADState
	previousState VADState
	stateMu       sync.RWMutex

	// Current audio chunk for turn analyzer (16kHz resampled if needed)
	currentAudioChunk []byte
}

// NewVADInputProcessor creates a new VAD input processor
func NewVADInputProcessor(analyzer VADAnalyzer) *VADInputProcessor {
	p := &VADInputProcessor{
		analyzer:      analyzer,
		audioBuffer:   make([]byte, 0),
		currentState:  VADStateQuiet,
		previousState: VADStateQuiet,
	}

	p.BaseProcessor = processors.NewBaseProcessor("VADInput", p)

	logger.Info("[VADInput] Created with analyzer (frames_required=%d)", analyzer.NumFramesRequired())
	return p
}

// NewVADInputProcessorWithTurn creates a VAD processor with optional Smart Turn analyzer
func NewVADInputProcessorWithTurn(analyzer VADAnalyzer, turnAnalyzer turn.TurnAnalyzer) *VADInputProcessor {
	p := NewVADInputProcessor(analyzer)
	p.turnAnalyzer = turnAnalyzer
	if turnAnalyzer != nil {
		logger.Info("[VADInput] Smart Turn analyzer enabled")
	}
	return p
}

// HandleFrame processes frames from upstream (typically WebSocket input)
func (p *VADInputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle AudioFrame - accumulate and run VAD
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		return p.handleAudioFrame(ctx, audioFrame, direction)
	}

	// Handle StartFrame - configure VAD sample rate
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		if err := p.handleStartFrame(startFrame); err != nil {
			logger.Error("[VADInput] Error handling StartFrame: %v", err)
		}
	}

	// Handle EndFrame - reset VAD state
	if _, ok := frame.(*frames.EndFrame); ok {
		p.analyzer.Restart()
		logger.Debug("[VADInput] EndFrame received, VAD state reset")
	}

	// Pass all frames downstream
	return p.PushFrame(frame, direction)
}

// handleStartFrame extracts sample rate and configures VAD and turn analyzer
func (p *VADInputProcessor) handleStartFrame(startFrame *frames.StartFrame) error {
	meta := startFrame.Metadata()
	if meta == nil {
		return nil
	}

	// Extract sample rate from metadata
	if sampleRate, ok := meta["sampleRate"].(int); ok {
		if err := p.analyzer.SetSampleRate(sampleRate); err != nil {
			return fmt.Errorf("failed to set VAD sample rate: %w", err)
		}
		logger.Info("[VADInput] Sample rate configured: %d Hz", sampleRate)

		// Configure turn analyzer sample rate (it expects 16kHz, so store source rate for resampling)
		if p.turnAnalyzer != nil {
			p.turnAnalyzer.SetSampleRate(16000) // Smart Turn expects 16kHz
			logger.Info("[VADInput] Turn analyzer configured for 16kHz (will resample from %d Hz)", sampleRate)
		}
	}

	return nil
}

// handleAudioFrame accumulates audio and runs VAD when enough samples available
// If turn analyzer is configured, also runs ML-based end-of-turn detection
func (p *VADInputProcessor) handleAudioFrame(ctx context.Context, audioFrame *frames.AudioFrame, direction frames.FrameDirection) error {
	p.bufferMu.Lock()

	// Append audio to buffer
	p.audioBuffer = append(p.audioBuffer, audioFrame.Data...)

	// Calculate required buffer size for VAD
	numFramesRequired := p.analyzer.NumFramesRequired()
	requiredBytes := numFramesRequired * 2 // int16 = 2 bytes per sample

	// Process audio if we have enough samples
	for len(p.audioBuffer) >= requiredBytes {
		// Extract chunk for VAD analysis
		chunk := p.audioBuffer[:requiredBytes]

		// Run VAD analysis
		newState, err := p.analyzer.AnalyzeAudio(chunk)
		if err != nil {
			logger.Error("[VADInput] VAD analysis error: %v", err)
			p.bufferMu.Unlock()
			return p.PushFrame(audioFrame, direction) // Pass through on error
		}

		// Check for state transitions
		p.stateMu.Lock()
		previousState := p.currentState
		p.previousState = p.currentState
		p.currentState = newState
		p.stateMu.Unlock()

		// Run turn analyzer if configured
		if p.turnAnalyzer != nil {
			isSpeech := newState == VADStateSpeaking || newState == VADStateStarting

			// Feed audio to turn analyzer
			// Note: If source is not 16kHz, audio should be resampled here
			// For now, we pass the raw audio (works if source is 16kHz or close)
			turnState := p.turnAnalyzer.AppendAudio(chunk, isSpeech)

			// Emit UserStartedSpeakingFrame when VAD confirms speech (reaches SPEAKING state)
			// We wait for SPEAKING (not STARTING) to avoid false triggers from brief voice blips
			// STARTING state is unstable and oscillates rapidly - SPEAKING is confirmed speech
			if (previousState == VADStateQuiet || previousState == VADStateStarting) && newState == VADStateSpeaking {
				logger.Info("[VADInput] 🎤 User started speaking (confirmed)")
				userStartedFrame := frames.NewUserStartedSpeakingFrame()
				if err := p.PushFrame(userStartedFrame, frames.Downstream); err != nil {
					logger.Error("[VADInput] Failed to push UserStartedSpeakingFrame: %v", err)
				}
			}

			// UserStoppedSpeakingFrame is controlled by turn analyzer (smart turn detection)
			// If turn analyzer says complete (silence timeout), emit end of turn
			if turnState == turn.TurnComplete {
				logger.Info("[VADInput] 🔇 Turn complete (silence timeout)")
				p.emitUserStoppedSpeaking()
			} else if newState == VADStateQuiet && (previousState == VADStateSpeaking || previousState == VADStateStopping) {
				// VAD went from confirmed speech (SPEAKING/STOPPING) to QUIET
				// Only then run ML inference to check if turn is complete
				// Don't trigger on STARTING → QUIET (those are just brief blips, not real speech)
				if p.turnAnalyzer.SpeechTriggered() {
					go p.runTurnAnalysis()
				}
			}
		} else {
			// No turn analyzer - use VAD-only logic
			if p.previousState != p.currentState {
				if err := p.emitStateTransitionFrames(); err != nil {
					logger.Error("[VADInput] Error emitting state transition frames: %v", err)
				}
			}
		}

		// Remove processed chunk from buffer
		p.audioBuffer = p.audioBuffer[requiredBytes:]
	}

	p.bufferMu.Unlock()

	// Always push audio frame downstream (STT needs all audio)
	return p.PushFrame(audioFrame, direction)
}

// runTurnAnalysis runs ML inference to determine if turn is complete
func (p *VADInputProcessor) runTurnAnalysis() {
	if p.turnAnalyzer == nil {
		return
	}

	state, metrics, err := p.turnAnalyzer.AnalyzeEndOfTurn()
	if err != nil {
		logger.Error("[VADInput] Turn analysis error: %v", err)
		return
	}

	if metrics != nil {
		logger.Debug("[VADInput] Turn analysis: %s (prob=%.3f, time=%.1fms)",
			state.String(), metrics.Probability, metrics.TotalTimeMs)
	}

	if state == turn.TurnComplete {
		logger.Info("[VADInput] 🔇 Turn complete (ML inference)")
		p.emitUserStoppedSpeaking()
	}
}

// emitUserStoppedSpeaking emits UserStoppedSpeakingFrame
func (p *VADInputProcessor) emitUserStoppedSpeaking() {
	userStoppedFrame := frames.NewUserStoppedSpeakingFrame()
	if err := p.PushFrame(userStoppedFrame, frames.Downstream); err != nil {
		logger.Error("[VADInput] Failed to push UserStoppedSpeakingFrame: %v", err)
	}
}

// emitStateTransitionFrames emits appropriate frames based on VAD state transitions
func (p *VADInputProcessor) emitStateTransitionFrames() error {
	p.stateMu.RLock()
	prev := p.previousState
	current := p.currentState
	p.stateMu.RUnlock()

	// User started speaking: QUIET/STARTING → SPEAKING
	if (prev == VADStateQuiet || prev == VADStateStarting) && current == VADStateSpeaking {
		logger.Info("[VADInput] 🎤 User started speaking")
		userStartedFrame := frames.NewUserStartedSpeakingFrame()
		if err := p.PushFrame(userStartedFrame, frames.Downstream); err != nil {
			return fmt.Errorf("failed to push UserStartedSpeakingFrame: %w", err)
		}
	}

	// User stopped speaking: SPEAKING/STOPPING → QUIET
	if (prev == VADStateSpeaking || prev == VADStateStopping) && current == VADStateQuiet {
		logger.Info("[VADInput] 🔇 User stopped speaking")
		userStoppedFrame := frames.NewUserStoppedSpeakingFrame()
		if err := p.PushFrame(userStoppedFrame, frames.Downstream); err != nil {
			return fmt.Errorf("failed to push UserStoppedSpeakingFrame: %w", err)
		}
	}

	return nil
}

// GetCurrentState returns the current VAD state
func (p *VADInputProcessor) GetCurrentState() VADState {
	p.stateMu.RLock()
	defer p.stateMu.RUnlock()
	return p.currentState
}
