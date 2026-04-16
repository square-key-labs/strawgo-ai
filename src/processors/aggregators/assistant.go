package aggregators

import (
	"context"
	"encoding/json"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// AssistantAggregatorParams holds configuration for the assistant aggregator
type AssistantAggregatorParams struct {
	EnableAutoContextSummarization bool
	AutoSummarizationConfig        LLMAutoContextSummarizationConfig
	SummaryLLM                     services.LLMService
	MainLLM                        services.LLMService
}

// DefaultAssistantAggregatorParams returns default parameters
func DefaultAssistantAggregatorParams() *AssistantAggregatorParams {
	return &AssistantAggregatorParams{}
}

// LLMAssistantAggregator accumulates assistant responses and tracks function calls
type LLMAssistantAggregator struct {
	*LLMContextAggregator

	// State
	started     int
	botSpeaking bool

	// Function call tracking
	functionCallsInProgress map[string]*frames.FunctionCallInProgressFrame

	// Configuration
	params     *AssistantAggregatorParams
	summarizer *LLMContextSummarizer
	log        *logger.Logger
}

// NewLLMAssistantAggregator creates a new assistant aggregator
func NewLLMAssistantAggregator(context *services.LLMContext, params *AssistantAggregatorParams) *LLMAssistantAggregator {
	if params == nil {
		params = DefaultAssistantAggregatorParams()
	}

	a := &LLMAssistantAggregator{
		started:                 0,
		functionCallsInProgress: make(map[string]*frames.FunctionCallInProgressFrame),
		params:                  params,
		summarizer:              NewLLMContextSummarizer(params.AutoSummarizationConfig, params.SummaryLLM),
		log:                     logger.WithPrefix("AssistantAggregator"),
	}

	a.LLMContextAggregator = NewLLMContextAggregator("LLMAssistantAggregator", context, "assistant", a)
	return a
}

// HandleFrame processes frames for assistant aggregation
func (a *LLMAssistantAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch frame.(type) {
	case *frames.TTSStartedFrame, *frames.BotStartedSpeakingFrame:
		a.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		a.botSpeaking = false
	}

	// Handle InterruptionFrame - clear state and reset
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		a.log.Info("Interruption received - clearing aggregation and resetting state")

		// Push any accumulated aggregation before resetting
		if len(a.aggregation) > 0 {
			if err := a.pushAggregation(ctx); err != nil {
				a.log.Warn("Error pushing aggregation on interruption: %v", err)
			}
		}

		// Reset state
		a.started = 0
		if err := a.Reset(); err != nil {
			a.log.Warn("Error resetting on interruption: %v", err)
		}

		// Handle interruption frame (calls base handler which drains queue)
		a.HandleInterruptionFrame()

		return a.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseStartFrame - increment nesting counter
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		a.started++
		a.log.Info("LLM response started (nesting level: %d)", a.started)
		return a.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseEndFrame - decrement counter and push aggregation
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		// Guard against stale end frames from interrupted contexts
		// These can arrive after an interruption reset the counter to 0
		if a.started <= 0 {
			a.log.Debug("Ignoring stale LLMFullResponseEndFrame (nesting level already %d)", a.started)
			return a.PushFrame(frame, direction)
		}

		a.started--
		a.log.Info("LLM response ended (nesting level: %d)", a.started)

		if a.started == 0 {
			if err := a.pushAggregation(ctx); err != nil {
				a.log.Warn("Error pushing aggregation: %v", err)
			}
		}

		return a.PushFrame(frame, direction)
	}

	// Handle TextFrame (from LLM) - accumulate if response is active
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		if a.started > 0 {
			a.log.Debug("Accumulating text: '%s'", textFrame.Text)
			a.AppendToAggregation(textFrame.Text)
			// Note: We don't set addSpaces here - keep default behavior
		}
		return a.PushFrame(frame, direction)
	}

	// Handle LLMTextFrame (legacy LLM output frame) - accumulate if response is active
	if llmTextFrame, ok := frame.(*frames.LLMTextFrame); ok {
		if a.started > 0 {
			a.log.Debug("Accumulating LLM text: '%s'", llmTextFrame.Text)
			a.AppendToAggregation(llmTextFrame.Text)
		}
		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallsStartedFrame - track function calls
	if callsStartedFrame, ok := frame.(*frames.FunctionCallsStartedFrame); ok {
		a.log.Info("Function calls started: %d calls", len(callsStartedFrame.FunctionCalls))
		for _, call := range callsStartedFrame.FunctionCalls {
			a.functionCallsInProgress[call.ToolCallID] = nil
		}
		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallInProgressFrame - add to context
	if inProgressFrame, ok := frame.(*frames.FunctionCallInProgressFrame); ok {
		a.log.Info("Function call in progress: %s (id: %s)", inProgressFrame.FunctionName, inProgressFrame.ToolCallID)

		// Convert arguments to JSON string
		argsJSON, err := json.Marshal(inProgressFrame.Arguments)
		if err != nil {
			a.log.Warn("Error marshaling function arguments: %v", err)
			return a.PushFrame(frame, direction)
		}

		// Add assistant message with tool calls to context
		a.context.AddMessageWithToolCalls([]services.ToolCall{
			{
				ID:   inProgressFrame.ToolCallID,
				Type: "function",
				Function: services.FunctionCall{
					Name:      inProgressFrame.FunctionName,
					Arguments: string(argsJSON),
				},
			},
		})

		// Add placeholder tool response (will be updated later)
		a.context.AddToolMessage(inProgressFrame.ToolCallID, "IN_PROGRESS")

		// Track this call
		a.functionCallsInProgress[inProgressFrame.ToolCallID] = inProgressFrame

		a.maybeAutoSummarize(ctx)

		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallResultFrame - update context and potentially trigger LLM
	if resultFrame, ok := frame.(*frames.FunctionCallResultFrame); ok {
		a.log.Info("Function call result: %s (id: %s)", resultFrame.FunctionName, resultFrame.ToolCallID)

		// Remove from in-progress tracking
		delete(a.functionCallsInProgress, resultFrame.ToolCallID)

		// Convert result to JSON string
		result := "COMPLETED"
		if resultFrame.Result != nil {
			resultJSON, err := json.Marshal(resultFrame.Result)
			if err != nil {
				a.log.Warn("Error marshaling function result: %v", err)
			} else {
				result = string(resultJSON)
			}
		}

		// Update the tool message in context
		a.updateFunctionCallResult(resultFrame.FunctionName, resultFrame.ToolCallID, result)
		a.maybeAutoSummarize(ctx)

		// Determine if we should run LLM again
		runLLM := false
		if resultFrame.Result != nil {
			if resultFrame.RunLLM != nil {
				runLLM = *resultFrame.RunLLM
			} else {
				// Default: run LLM if no more function calls in progress
				runLLM = len(a.functionCallsInProgress) == 0
			}
		}

		if runLLM {
			a.log.Info("Triggering LLM execution after function result")
			return a.PushContextFrame(frames.Upstream)
		}

		return a.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.LLMSummarizeContextFrame); ok {
		a.forceSummarizeContext(ctx)
		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallCancelFrame - remove from tracking
	if cancelFrame, ok := frame.(*frames.FunctionCallCancelFrame); ok {
		a.log.Info("Function call cancelled: %s (id: %s)", cancelFrame.FunctionName, cancelFrame.ToolCallID)

		if inProgressFrame, exists := a.functionCallsInProgress[cancelFrame.ToolCallID]; exists {
			if inProgressFrame.CancelOnInterruption {
				a.updateFunctionCallResult(cancelFrame.FunctionName, cancelFrame.ToolCallID, "CANCELLED")
				delete(a.functionCallsInProgress, cancelFrame.ToolCallID)
			}
		}

		return a.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return a.PushFrame(frame, direction)
}

// pushAggregation pushes the accumulated assistant response to context
func (a *LLMAssistantAggregator) pushAggregation(ctx context.Context) error {
	if len(a.aggregation) == 0 {
		a.log.Debug("No aggregation to push")
		return nil
	}

	text := a.AggregationString()
	a.log.Info("Pushing aggregation: '%s'", text)

	// Reset aggregation
	if err := a.Reset(); err != nil {
		return err
	}

	// Add assistant message to context if we have content
	if text != "" {
		a.context.AddAssistantMessage(text)
		a.maybeAutoSummarize(ctx)
	}

	// Push context frame downstream
	if err := a.PushContextFrame(frames.Downstream); err != nil {
		return err
	}

	// Push timestamp frame to mark completion
	timestamp := time.Now().Format(time.RFC3339)
	a.log.Info("Assistant response completed at %s", timestamp)

	return nil
}

func (a *LLMAssistantAggregator) maybeAutoSummarize(ctx context.Context) {
	if a.summarizer == nil {
		return
	}
	if !a.params.EnableAutoContextSummarization {
		return
	}
	if !a.summarizer.ShouldAutoSummarize(a.context) {
		return
	}
	a.summarizer.SummarizeContext(ctx, a.context, a.params.MainLLM)
}

func (a *LLMAssistantAggregator) forceSummarizeContext(ctx context.Context) {
	if a.summarizer == nil {
		return
	}
	a.summarizer.SummarizeContext(ctx, a.context, a.params.MainLLM)
}

// updateFunctionCallResult finds and updates a tool message in the context
func (a *LLMAssistantAggregator) updateFunctionCallResult(functionName, toolCallID, result string) {
	for i := range a.context.Messages {
		msg := &a.context.Messages[i]
		if msg.Role == "tool" && msg.ToolCallID == toolCallID {
			msg.Content = result
			a.log.Debug("Updated function result for %s: %s", functionName, result)
			break
		}
	}
}

// Reset overrides base Reset to also clear assistant aggregator state
func (a *LLMAssistantAggregator) Reset() error {
	a.started = 0
	return a.LLMContextAggregator.Reset()
}
