package aggregators

import (
	"context"
	"log"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// UserAggregatorParams holds configuration for the user aggregator
type UserAggregatorParams struct {
	AggregationTimeout             time.Duration // Timeout for late transcriptions (default: 500ms)
	TurnEmulatedVADTimeout         time.Duration // Timeout for emulated VAD (default: 800ms)
	EnableEmulatedVADInterruptions bool          // Allow whispered interruptions (default: false)
}

// DefaultUserAggregatorParams returns default parameters
func DefaultUserAggregatorParams() *UserAggregatorParams {
	return &UserAggregatorParams{
		AggregationTimeout:             500 * time.Millisecond,
		TurnEmulatedVADTimeout:         800 * time.Millisecond,
		EnableEmulatedVADInterruptions: false,
	}
}

// LLMUserAggregator accumulates user input and handles interruption decisions
type LLMUserAggregator struct {
	*LLMContextAggregator

	// State tracking
	userSpeaking          bool
	botSpeaking           bool
	wasBotSpeaking        bool
	seenInterimResults    bool
	waitingForAggregation bool
	interruptionSent      bool // Track if interruption already sent for current utterance

	// Aggregation task
	aggregationCtx    context.Context
	aggregationCancel context.CancelFunc
	aggregationEvent  chan struct{}

	// Configuration
	params *UserAggregatorParams
}

// NewLLMUserAggregator creates a new user aggregator
func NewLLMUserAggregator(context *services.LLMContext, params *UserAggregatorParams) *LLMUserAggregator {
	if params == nil {
		params = DefaultUserAggregatorParams()
	}

	u := &LLMUserAggregator{
		userSpeaking:          false,
		botSpeaking:           false,
		wasBotSpeaking:        false,
		seenInterimResults:    false,
		waitingForAggregation: false,
		aggregationEvent:      make(chan struct{}, 1),
		params:                params,
	}

	u.LLMContextAggregator = NewLLMContextAggregator("LLMUserAggregator", context, "user", u)
	return u
}

// HandleFrame processes frames for user aggregation
func (u *LLMUserAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame to configure interruptions and start aggregation task
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		u.HandleStartFrame(startFrame)
		log.Printf("[%s] Interruptions: allowed=%v, strategies=%d", u.Name(), u.InterruptionsAllowed(), len(u.InterruptionStrategies()))

		// Start aggregation task
		u.aggregationCtx, u.aggregationCancel = context.WithCancel(ctx)
		go u.aggregationTaskHandler()

		return u.PushFrame(frame, direction)
	}

	// Handle EndFrame - cleanup
	if _, ok := frame.(*frames.EndFrame); ok {
		log.Printf("[%s] Received EndFrame, cleaning up", u.Name())
		if u.aggregationCancel != nil {
			u.aggregationCancel()
		}
		return u.PushFrame(frame, direction)
	}

	// Handle TTSStartedFrame - bot started speaking
	if _, ok := frame.(*frames.TTSStartedFrame); ok {
		u.botSpeaking = true
		log.Printf("[%s] Bot started speaking", u.Name())
		return u.PushFrame(frame, direction)
	}

	// Handle TTSStoppedFrame - bot stopped speaking
	if _, ok := frame.(*frames.TTSStoppedFrame); ok {
		u.botSpeaking = false
		log.Printf("[%s] Bot stopped speaking", u.Name())

		// Reset interruption strategies when bot stops - clears accumulated text
		for _, strategy := range u.InterruptionStrategies() {
			if err := strategy.Reset(); err != nil {
				log.Printf("[%s] Error resetting strategy on bot stop: %v", u.Name(), err)
			}
		}

		return u.PushFrame(frame, direction)
	}

	// Handle AudioFrame - feed to audio-based interruption strategies
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		// Feed audio to all interruption strategies that support audio
		for _, strategy := range u.InterruptionStrategies() {
			// Check if strategy supports audio input
			if audioStrategy, ok := strategy.(interface {
				AppendAudio([]byte, int) error
			}); ok {
				if err := audioStrategy.AppendAudio(audioFrame.Data, audioFrame.SampleRate); err != nil {
					log.Printf("[%s] Error appending audio to strategy: %v", u.Name(), err)
				}
			}
		}

		// Check if we should interrupt based on audio
		if u.InterruptionsAllowed() && u.botSpeaking && len(u.InterruptionStrategies()) > 0 {
			for _, strategy := range u.InterruptionStrategies() {
				shouldInterrupt, err := strategy.ShouldInterrupt()
				if err != nil {
					log.Printf("[%s] Error checking interruption: %v", u.Name(), err)
					continue
				}

				if shouldInterrupt {
					// Only send interruption if not already sent for this utterance
					if !u.interruptionSent {
						log.Printf("[%s] 🔴 AUDIO-BASED interruption detected!", u.Name())
						u.interruptionSent = true

						// Send InterruptionTaskFrame upstream
						if err := u.PushInterruptionTaskFrame(); err != nil {
							log.Printf("[%s] Error pushing interruption task frame: %v", u.Name(), err)
						}

						// Reset all strategies after interruption
						for _, s := range u.InterruptionStrategies() {
							if err := s.Reset(); err != nil {
								log.Printf("[%s] Error resetting strategy: %v", u.Name(), err)
							}
						}
					}
					break
				}
			}
		}

		// Pass audio frame downstream
		return u.PushFrame(frame, direction)
	}

	// Handle TranscriptionFrame - accumulate user text
	if transcriptionFrame, ok := frame.(*frames.TranscriptionFrame); ok {
		text := transcriptionFrame.Text
		if text == "" {
			// Consume empty transcription frames (don't pass downstream)
			return nil
		}

		log.Printf("[%s] Transcription (final=%v): '%s'", u.Name(), transcriptionFrame.IsFinal, text)

		// Handle aggregation differently for final vs interim
		if transcriptionFrame.IsFinal {
			// Final transcription - append to aggregation for LLM context
			u.AppendToAggregation(text)
			u.seenInterimResults = false // Reset interim flag

			// Feed final transcription to TEXT-BASED strategies (e.g., MinWordsStrategy)
			// This allows hybrid text+audio based interruption detection
			if u.InterruptionsAllowed() && u.botSpeaking && len(u.InterruptionStrategies()) > 0 {
				for _, strategy := range u.InterruptionStrategies() {
					if textStrategy, ok := strategy.(interface {
						AppendText(string) error
					}); ok {
						textStrategy.AppendText(text)
					}
				}

				// Check if any strategy wants to interrupt (text-based or audio-based)
				if !u.interruptionSent {
					for _, strategy := range u.InterruptionStrategies() {
						if shouldInterrupt, _ := strategy.ShouldInterrupt(); shouldInterrupt {
							log.Printf("[%s] 🔴 INTERRUPTION detected (text-based strategy)!", u.Name())
							u.interruptionSent = true
							u.PushInterruptionTaskFrame()
							// Reset all strategies after interruption
							for _, s := range u.InterruptionStrategies() {
								s.Reset()
							}
							break
						}
					}
				}
			}

			// Signal aggregation task
			select {
			case u.aggregationEvent <- struct{}{}:
			default:
			}

			// If not waiting for more transcriptions, push immediately
			if !u.waitingForAggregation && !u.userSpeaking {
				if err := u.pushAggregation(); err != nil {
					log.Printf("[%s] Error pushing aggregation: %v", u.Name(), err)
				}
			}
		} else {
			// Interim result - DO NOT append to aggregation
			// Only set flag to indicate we've seen interim results
			u.seenInterimResults = true
			log.Printf("[%s] Interim transcription received (audio-based strategies handle interruption)", u.Name())
		}

		// Consume the transcription frame - DO NOT pass downstream
		// The aggregator will send LLMContextFrame when ready
		return nil
	}

	// Handle LLMMessagesAppendFrame
	if appendFrame, ok := frame.(*frames.LLMMessagesAppendFrame); ok {
		if messages, ok := appendFrame.Messages.([]services.LLMMessage); ok {
			for _, msg := range messages {
				u.context.Messages = append(u.context.Messages, msg)
			}
			if appendFrame.RunLLM {
				return u.PushContextFrame(frames.Downstream)
			}
		}
		return nil
	}

	// Handle LLMMessagesUpdateFrame
	if updateFrame, ok := frame.(*frames.LLMMessagesUpdateFrame); ok {
		if messages, ok := updateFrame.Messages.([]services.LLMMessage); ok {
			u.context.Messages = messages
			if updateFrame.RunLLM {
				return u.PushContextFrame(frames.Downstream)
			}
		}
		return nil
	}

	// Pass all other frames through
	return u.PushFrame(frame, direction)
}

// pushAggregation pushes the accumulated text to the LLM context
// NOTE: Interruption is handled by AUDIO-BASED strategies, not text-based
// This function only manages aggregation and context updates
func (u *LLMUserAggregator) pushAggregation() error {
	if len(u.aggregation) == 0 {
		return nil
	}

	log.Printf("[%s] pushAggregation called: aggregation='%s'", u.Name(), u.AggregationString())

	// Audio-based strategies have already determined if interruption should occur
	// We just process the aggregation and push context to LLM
	return u.processAggregation()
}

// processAggregation converts aggregation to context and pushes downstream
func (u *LLMUserAggregator) processAggregation() error {
	text := u.AggregationString()
	log.Printf("[%s] Processing aggregation: '%s'", u.Name(), text)

	// Reset aggregation state
	if err := u.Reset(); err != nil {
		return err
	}

	// Add user message to context
	u.context.AddUserMessage(text)

	// Push context frame downstream to trigger LLM
	return u.PushContextFrame(frames.Downstream)
}


// aggregationTaskHandler runs in the background to handle timeouts
func (u *LLMUserAggregator) aggregationTaskHandler() {
	timeout := u.params.AggregationTimeout

	for {
		select {
		case <-u.aggregationCtx.Done():
			log.Printf("[%s] Aggregation task stopped", u.Name())
			return

		case <-time.After(timeout):
			// Timeout - push aggregation if we have text and user is not speaking
			if !u.userSpeaking && len(u.aggregation) > 0 {
				log.Printf("[%s] Aggregation timeout - pushing accumulated text", u.Name())
				if err := u.pushAggregation(); err != nil {
					log.Printf("[%s] Error pushing aggregation on timeout: %v", u.Name(), err)
				}
			}

		case <-u.aggregationEvent:
			// Transcription event - could implement emulated VAD here if needed
			log.Printf("[%s] Aggregation event received", u.Name())
		}
	}
}

// Reset overrides base Reset to also clear user aggregator state
// NOTE: Does NOT reset interruption strategies - they should accumulate text
// while bot is speaking. Strategies are reset on interruption or bot stop.
func (u *LLMUserAggregator) Reset() error {
	u.wasBotSpeaking = false
	u.seenInterimResults = false
	u.waitingForAggregation = false
	u.interruptionSent = false // Clear interruption flag for next utterance

	// DO NOT reset strategies here - they need to accumulate text while bot speaks
	// Strategies are reset in:
	// 1. After interruption is triggered (line 191-193)
	// 2. When bot stops speaking (TTSStoppedFrame handler)

	return u.LLMContextAggregator.Reset()
}
