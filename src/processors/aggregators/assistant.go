package aggregators

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/google/uuid"

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

	// FunctionRegistry, when set, lets the aggregator look up per-tool policy
	// (TimeoutSecs, CancelOnInterruption) without each LLM service having to
	// thread the registry through. Optional — if nil, all calls use the
	// frame-borne CancelOnInterruption flag and DefaultFunctionCallTimeout.
	FunctionRegistry services.FunctionRegistry

	// DefaultFunctionCallTimeout, when > 0, applies to any in-progress tool
	// call that does not have a per-tool TimeoutSecs in the registry.
	// On expiry the aggregator pushes a FunctionCallCancelFrame upstream,
	// matching pipecat's function_call_timeout_secs default behavior.
	DefaultFunctionCallTimeout time.Duration
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

	// Function call tracking. functionCallsInProgress is keyed by ToolCallID
	// for fast lookup. groupRemaining counts how many calls in each group are
	// still pending; when it hits zero (and the group had >0 to start with)
	// the LLM is triggered exactly once for the whole batch.
	// callTimers tracks per-call cancellation timers when DefaultFunctionCallTimeout
	// (or per-tool TimeoutSecs) is configured.
	functionCallsInProgress map[string]*frames.FunctionCallInProgressFrame
	callGroup               map[string]string // toolCallID → groupID
	groupRemaining          map[string]int    // groupID → remaining call count
	callTimers              map[string]context.CancelFunc
	fnMu                    sync.Mutex

	// Configuration
	params     *AssistantAggregatorParams
	summarizer *LLMContextSummarizer
	log        *logger.Logger
}

// NewLLMAssistantAggregator creates a new assistant aggregator
func NewLLMAssistantAggregator(llmContext *services.LLMContext, params *AssistantAggregatorParams) *LLMAssistantAggregator {
	if params == nil {
		params = DefaultAssistantAggregatorParams()
	}

	a := &LLMAssistantAggregator{
		started:                 0,
		functionCallsInProgress: make(map[string]*frames.FunctionCallInProgressFrame),
		callGroup:               make(map[string]string),
		groupRemaining:          make(map[string]int),
		callTimers:              make(map[string]context.CancelFunc),
		params:                  params,
		summarizer:              NewLLMContextSummarizer(params.AutoSummarizationConfig, params.SummaryLLM),
		log:                     logger.WithPrefix("AssistantAggregator"),
	}

	a.LLMContextAggregator = NewLLMContextAggregator("LLMAssistantAggregator", llmContext, "assistant", a)
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

	// Handle FunctionCallsStartedFrame - track function calls.
	// Synthesize a GroupID if the producer didn't supply one. This GroupID
	// is then attached by the aggregator to subsequent
	// FunctionCallInProgressFrame / FunctionCallResultFrame frames sharing
	// these ToolCallIDs, giving us pipecat-style group_parallel_tools=true
	// semantics: a single LLM trigger after the whole batch completes.
	if callsStartedFrame, ok := frame.(*frames.FunctionCallsStartedFrame); ok {
		a.log.Info("Function calls started: %d calls", len(callsStartedFrame.FunctionCalls))

		groupID := callsStartedFrame.GroupID
		if groupID == "" {
			groupID = uuid.New().String()
			callsStartedFrame.GroupID = groupID
		}

		a.fnMu.Lock()
		for _, call := range callsStartedFrame.FunctionCalls {
			a.functionCallsInProgress[call.ToolCallID] = nil
			a.callGroup[call.ToolCallID] = groupID
			a.groupRemaining[groupID]++
		}
		a.fnMu.Unlock()

		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallInProgressFrame - add to context
	if inProgressFrame, ok := frame.(*frames.FunctionCallInProgressFrame); ok {
		a.log.Info("Function call in progress: %s (id: %s)", inProgressFrame.FunctionName, inProgressFrame.ToolCallID)

		// If the aggregator already assigned a GroupID via FunctionCallsStartedFrame,
		// stamp it on this frame so downstream consumers see it.
		a.fnMu.Lock()
		if gid, ok := a.callGroup[inProgressFrame.ToolCallID]; ok && inProgressFrame.GroupID == "" {
			inProgressFrame.GroupID = gid
		}
		a.fnMu.Unlock()

		// FunctionRegistry, if configured, may override the per-call
		// CancelOnInterruption flag (so the LLM service does not need to
		// know about each tool's policy).
		if a.params != nil && a.params.FunctionRegistry != nil {
			if reg, found := a.params.FunctionRegistry.Lookup(inProgressFrame.FunctionName); found {
				inProgressFrame.CancelOnInterruption = reg.CancelOnInterruption
			}
		}

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
		a.fnMu.Lock()
		a.functionCallsInProgress[inProgressFrame.ToolCallID] = inProgressFrame
		a.fnMu.Unlock()

		a.startCallTimeoutLocked(inProgressFrame)

		a.maybeAutoSummarize(ctx)

		return a.PushFrame(frame, direction)
	}

	// Handle FunctionCallResultFrame - update context and potentially trigger LLM
	if resultFrame, ok := frame.(*frames.FunctionCallResultFrame); ok {
		a.log.Info("Function call result: %s (id: %s)", resultFrame.FunctionName, resultFrame.ToolCallID)

		// Look up the originating in-progress frame so we know group +
		// async semantics for this call. Then prune our state.
		a.fnMu.Lock()
		inProgress := a.functionCallsInProgress[resultFrame.ToolCallID]
		groupID, _ := a.callGroup[resultFrame.ToolCallID]
		delete(a.functionCallsInProgress, resultFrame.ToolCallID)
		delete(a.callGroup, resultFrame.ToolCallID)
		groupComplete := false
		if groupID != "" {
			a.groupRemaining[groupID]--
			if a.groupRemaining[groupID] <= 0 {
				delete(a.groupRemaining, groupID)
				groupComplete = true
			}
		}
		// Stop any pending per-call timeout.
		if cancelTimer, ok := a.callTimers[resultFrame.ToolCallID]; ok {
			cancelTimer()
			delete(a.callTimers, resultFrame.ToolCallID)
		}
		// If the producer left GroupID empty, fill it in for downstream.
		if resultFrame.GroupID == "" {
			resultFrame.GroupID = groupID
		}
		a.fnMu.Unlock()

		isAsync := inProgress != nil && !inProgress.CancelOnInterruption

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

		// For async function calls (CancelOnInterruption=false), the LLM
		// already continued without waiting. We inject the late-arriving
		// result as a "developer" role message so the LLM can react to
		// it on its next inference. We still update the placeholder tool
		// message so the conversation history remains valid.
		a.updateFunctionCallResult(resultFrame.FunctionName, resultFrame.ToolCallID, result)
		if isAsync {
			a.context.Messages = append(a.context.Messages, services.LLMMessage{
				Role:    "developer",
				Content: "Function `" + resultFrame.FunctionName + "` returned: " + result,
			})
		}
		a.maybeAutoSummarize(ctx)

		// Determine if we should run the LLM again.
		// 1. Explicit RunLLM on the frame wins.
		// 2. Async results always trigger an inference (they're the only way
		//    the model learns the result).
		// 3. Otherwise, sync results trigger only when the *whole group*
		//    has completed — pipecat group_parallel_tools=true semantics.
		// 4. With no group at all (empty GroupID), fall back to the legacy
		//    "no calls in progress" gate so existing services keep working.
		runLLM := false
		if resultFrame.RunLLM != nil {
			runLLM = *resultFrame.RunLLM
		} else if isAsync {
			runLLM = true
		} else if groupID != "" {
			runLLM = groupComplete
		} else if resultFrame.Result != nil {
			a.fnMu.Lock()
			runLLM = len(a.functionCallsInProgress) == 0
			a.fnMu.Unlock()
		}

		if runLLM {
			a.log.Info("Triggering LLM execution after function result (group=%q async=%v)", groupID, isAsync)
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

		a.fnMu.Lock()
		inProgressFrame, exists := a.functionCallsInProgress[cancelFrame.ToolCallID]
		if exists && inProgressFrame.CancelOnInterruption {
			a.updateFunctionCallResult(cancelFrame.FunctionName, cancelFrame.ToolCallID, "CANCELLED")
			delete(a.functionCallsInProgress, cancelFrame.ToolCallID)
			if gid, ok := a.callGroup[cancelFrame.ToolCallID]; ok {
				delete(a.callGroup, cancelFrame.ToolCallID)
				a.groupRemaining[gid]--
				if a.groupRemaining[gid] <= 0 {
					delete(a.groupRemaining, gid)
				}
			}
			if cancelTimer, ok := a.callTimers[cancelFrame.ToolCallID]; ok {
				cancelTimer()
				delete(a.callTimers, cancelFrame.ToolCallID)
			}
		}
		a.fnMu.Unlock()

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
	a.fnMu.Lock()
	for _, cancel := range a.callTimers {
		cancel()
	}
	a.callTimers = make(map[string]context.CancelFunc)
	a.fnMu.Unlock()
	return a.LLMContextAggregator.Reset()
}

// startCallTimeoutLocked starts a per-call cancellation timer for an
// in-progress function call. Per-tool TimeoutSecs from the FunctionRegistry
// wins; otherwise falls back to AssistantAggregatorParams.DefaultFunctionCallTimeout.
// If neither is configured, no timer is started (matches pipecat default of
// function_call_timeout_secs=None).
//
// On expiry the aggregator pushes a FunctionCallCancelFrame upstream so the
// LLM service can cancel the in-flight call. The cancel frame routes back
// through HandleFrame via the regular cancel path.
func (a *LLMAssistantAggregator) startCallTimeoutLocked(in *frames.FunctionCallInProgressFrame) {
	timeout := a.params.DefaultFunctionCallTimeout
	if a.params.FunctionRegistry != nil {
		if reg, ok := a.params.FunctionRegistry.Lookup(in.FunctionName); ok {
			if reg.TimeoutSecs != nil {
				timeout = *reg.TimeoutSecs
			}
		}
	}
	if timeout <= 0 {
		return
	}

	timerCtx, cancel := context.WithTimeout(context.Background(), timeout)

	a.fnMu.Lock()
	a.callTimers[in.ToolCallID] = cancel
	a.fnMu.Unlock()

	go func() {
		<-timerCtx.Done()
		if timerCtx.Err() != context.DeadlineExceeded {
			return // cancelled normally on result/cancel arrival
		}

		a.fnMu.Lock()
		_, stillPending := a.functionCallsInProgress[in.ToolCallID]
		delete(a.callTimers, in.ToolCallID)
		a.fnMu.Unlock()

		if !stillPending {
			return
		}

		a.log.Warn("Function call timed out: %s (id: %s, timeout: %v)", in.FunctionName, in.ToolCallID, timeout)
		_ = a.PushFrame(frames.NewFunctionCallCancelFrame(in.ToolCallID, in.FunctionName), frames.Upstream)
	}()
}
