package aggregators

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// AssistantAggregatorParams holds configuration for the assistant aggregator
type AssistantAggregatorParams struct {
	// Currently no specific params needed
}

// DefaultAssistantAggregatorParams returns default parameters
func DefaultAssistantAggregatorParams() *AssistantAggregatorParams {
	return &AssistantAggregatorParams{}
}

// LLMAssistantAggregator accumulates assistant responses and tracks function calls
type LLMAssistantAggregator struct {
	*LLMContextAggregator

	// State
	started int // Nesting counter for LLM responses

	// Function call tracking
	functionCallsInProgress map[string]*frames.FunctionCallInProgressFrame

	// Configuration
	params *AssistantAggregatorParams
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
	}

	a.LLMContextAggregator = NewLLMContextAggregator("LLMAssistantAggregator", context, "assistant", a)
	return a
}

// HandleFrame processes frames for assistant aggregation
func (a *LLMAssistantAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle InterruptionFrame - clear state and reset
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		log.Printf("[%s] ⚠️  Interruption received - clearing aggregation and resetting state", a.Name())

		// Push any accumulated aggregation before resetting
		if len(a.aggregation) > 0 {
			if err := a.pushAggregation(); err != nil {
				log.Printf("[%s] Error pushing aggregation on interruption: %v", a.Name(), err)
			}
		}

		// Reset state
		a.started = 0
		if err := a.Reset(); err != nil {
			log.Printf("[%s] Error resetting on interruption: %v", a.Name(), err)
		}

		// Handle interruption frame (calls base handler which drains queue)
		a.HandleInterruptionFrame()

		return a.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseStartFrame - increment nesting counter
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		a.started++
		log.Printf("[%s] LLM response started (nesting level: %d)", a.Name(), a.started)
		return a.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseEndFrame - decrement counter and push aggregation
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		a.started--
		log.Printf("[%s] LLM response ended (nesting level: %d)", a.Name(), a.started)

		if a.started <= 0 {
			if err := a.pushAggregation(); err != nil {
				log.Printf("[%s] Error pushing aggregation: %v", a.Name(), err)
			}
		}

		return a.PushFrame(frame, direction)
	}

	// Handle TextFrame (from LLM) - accumulate if response is active
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		if a.started > 0 {
			log.Printf("[%s] Accumulating text: '%s'", a.Name(), textFrame.Text)
			a.AppendToAggregation(textFrame.Text)
			// Note: We don't set addSpaces here - keep default behavior
		}
		return a.PushFrame(frame, direction)
	}

	// Handle LLMTextFrame (legacy LLM output frame) - accumulate if response is active
	if llmTextFrame, ok := frame.(*frames.LLMTextFrame); ok {
		if a.started > 0 {
			log.Printf("[%s] Accumulating LLM text: '%s'", a.Name(), llmTextFrame.Text)
			a.AppendToAggregation(llmTextFrame.Text)
		}
		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallsStartedFrame - track function calls
	if callsStartedFrame, ok := frame.(*frames.FunctionCallsStartedFrame); ok {
		log.Printf("[%s] Function calls started: %d calls", a.Name(), len(callsStartedFrame.FunctionCalls))
		for _, call := range callsStartedFrame.FunctionCalls {
			a.functionCallsInProgress[call.ToolCallID] = nil
		}
		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallInProgressFrame - add to context
	if inProgressFrame, ok := frame.(*frames.FunctionCallInProgressFrame); ok {
		log.Printf("[%s] Function call in progress: %s (id: %s)", a.Name(), inProgressFrame.FunctionName, inProgressFrame.ToolCallID)

		// Convert arguments to JSON string
		argsJSON, err := json.Marshal(inProgressFrame.Arguments)
		if err != nil {
			log.Printf("[%s] Error marshaling function arguments: %v", a.Name(), err)
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

		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallResultFrame - update context and potentially trigger LLM
	if resultFrame, ok := frame.(*frames.FunctionCallResultFrame); ok {
		log.Printf("[%s] Function call result: %s (id: %s)", a.Name(), resultFrame.FunctionName, resultFrame.ToolCallID)

		// Remove from in-progress tracking
		delete(a.functionCallsInProgress, resultFrame.ToolCallID)

		// Convert result to JSON string
		result := "COMPLETED"
		if resultFrame.Result != nil {
			resultJSON, err := json.Marshal(resultFrame.Result)
			if err != nil {
				log.Printf("[%s] Error marshaling function result: %v", a.Name(), err)
			} else {
				result = string(resultJSON)
			}
		}

		// Update the tool message in context
		a.updateFunctionCallResult(resultFrame.FunctionName, resultFrame.ToolCallID, result)

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
			log.Printf("[%s] Triggering LLM execution after function result", a.Name())
			return a.PushContextFrame(frames.Upstream)
		}

		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallCancelFrame - remove from tracking
	if cancelFrame, ok := frame.(*frames.FunctionCallCancelFrame); ok {
		log.Printf("[%s] Function call cancelled: %s (id: %s)", a.Name(), cancelFrame.FunctionName, cancelFrame.ToolCallID)

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
func (a *LLMAssistantAggregator) pushAggregation() error {
	if len(a.aggregation) == 0 {
		log.Printf("[%s] No aggregation to push", a.Name())
		return nil
	}

	text := a.AggregationString()
	log.Printf("[%s] Pushing aggregation: '%s'", a.Name(), text)

	// Reset aggregation
	if err := a.Reset(); err != nil {
		return err
	}

	// Add assistant message to context if we have content
	if text != "" {
		a.context.AddAssistantMessage(text)
	}

	// Push context frame downstream
	if err := a.PushContextFrame(frames.Downstream); err != nil {
		return err
	}

	// Push timestamp frame to mark completion
	timestamp := time.Now().Format(time.RFC3339)
	log.Printf("[%s] Assistant response completed at %s", a.Name(), timestamp)

	return nil
}

// updateFunctionCallResult finds and updates a tool message in the context
func (a *LLMAssistantAggregator) updateFunctionCallResult(functionName, toolCallID, result string) {
	for i := range a.context.Messages {
		msg := &a.context.Messages[i]
		if msg.Role == "tool" && msg.ToolCallID == toolCallID {
			msg.Content = result
			log.Printf("[%s] Updated function result for %s: %s", a.Name(), functionName, result)
			break
		}
	}
}

// Reset overrides base Reset to also clear assistant aggregator state
func (a *LLMAssistantAggregator) Reset() error {
	a.started = 0
	return a.LLMContextAggregator.Reset()
}
