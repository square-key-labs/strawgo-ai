package aggregators

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const (
	toolResultCompleted = "COMPLETED"
	finishedCallsLimit  = 64
)

type finishedCall struct {
	content string
	runLLM  bool
}

type AssistantAggregatorParams struct {
	EnableAutoContextSummarization bool
	AutoSummarizationConfig        LLMAutoContextSummarizationConfig
	Summarize                      SummaryFunc
	SummarizationTimeout           time.Duration
	OnAssistantTurnStopped         func(text string, interrupted bool)
	OnAssistantThought             func(text string)
	OnSummaryApplied               func(beforeCount, afterCount int)
}

func DefaultAssistantAggregatorParams() *AssistantAggregatorParams {
	return &AssistantAggregatorParams{}
}

type LLMAssistantAggregator struct {
	*LLMContextAggregator

	params     AssistantAggregatorParams
	summarizer *LLMContextSummarizer

	depth      int
	turnActive bool
	turnStart  time.Time
	llmText    strings.Builder
	ttsText    strings.Builder

	thinking      bool
	thoughtAppend bool
	thoughtText   strings.Builder

	calls         map[string]*frames.FunctionCallInProgressFrame
	finished      map[string]finishedCall
	finishedOrder []string

	botSpeaking      bool
	userSpeaking     bool
	pushOwed         bool
	pushOnBotStopped bool
}

func NewLLMAssistantAggregator(c *llm.Context, params *AssistantAggregatorParams) *LLMAssistantAggregator {
	if params == nil {
		params = DefaultAssistantAggregatorParams()
	}
	a := &LLMAssistantAggregator{
		params:   *params,
		calls:    make(map[string]*frames.FunctionCallInProgressFrame),
		finished: make(map[string]finishedCall),
	}
	if params.Summarize != nil {
		a.summarizer = NewLLMContextSummarizer(params.AutoSummarizationConfig, params.Summarize)
		a.summarizer.SetTimeout(params.SummarizationTimeout)
		a.summarizer.OnSummaryApplied = params.OnSummaryApplied
	}
	a.LLMContextAggregator = NewLLMContextAggregator("LLMAssistantAggregator", c, llm.RoleAssistant, a)
	return a
}

func (a *LLMAssistantAggregator) Summarizer() *LLMContextSummarizer {
	return a.summarizer
}

func (a *LLMAssistantAggregator) Stop() error {
	if a.summarizer != nil {
		a.summarizer.Cancel()
	}
	return a.BaseProcessor.Stop()
}

func (a *LLMAssistantAggregator) HasFunctionCallsInProgress() bool {
	return len(a.calls) > 0
}

func (a *LLMAssistantAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *selfFrame:
		if f.kind == selfSummary && a.summarizer != nil {
			if res, ok := f.payload.(SummaryResult); ok {
				if res.Err != nil {
					logger.Warn("[%s] context summarization failed: %v", a.Name(), res.Err)
				} else {
					a.summarizer.Apply(a.context, res)
				}
			}
		}
		return nil
	case *frames.InterruptionFrame:
		a.commit(true)
		a.resetResponse()
		return a.PushFrame(frame, direction)
	case *frames.EndFrame:
		a.commit(false)
		a.cancelSummary()
		return a.PushFrame(frame, direction)
	case *frames.CancelFrame:
		a.commit(true)
		a.cancelSummary()
		return a.PushFrame(frame, direction)
	case *frames.LLMAssistantPushAggregationFrame:
		a.commit(false)
		return nil
	case *frames.LLMFullResponseStartFrame:
		a.depth++
		a.pushOwed = false
		a.startTurn()
		return a.PushFrame(frame, direction)
	case *frames.LLMFullResponseEndFrame:
		if a.depth > 0 {
			a.depth--
			if a.depth == 0 {
				a.commit(false)
			}
		}
		return a.PushFrame(frame, direction)
	case *frames.TTSStartedFrame:
		a.startTurn()
		return a.PushFrame(frame, direction)
	case *frames.LLMTextFrame:
		if direction == frames.Downstream && a.turnActive {
			a.llmText.WriteString(f.Text)
		}
		return a.PushFrame(frame, direction)
	case *frames.TextFrame:
		if direction == frames.Downstream && a.turnActive {
			a.llmText.WriteString(f.Text)
		}
		return a.PushFrame(frame, direction)
	case *frames.AggregatedTextFrame:
		if direction == frames.Downstream && a.turnActive && f.AppendToContext {
			text := f.Text
			if f.RawText != "" {
				text = f.RawText
			}
			a.llmText.WriteString(text)
		}
		return a.PushFrame(frame, direction)
	case *frames.TTSTextFrame:
		if direction == frames.Downstream && a.turnActive && f.AppendToContext && f.Text != "" {
			if (f.TextType == frames.TextTypeWord || f.TextType == frames.TextTypeSentence) && a.ttsText.Len() > 0 {
				a.ttsText.WriteByte(' ')
			}
			a.ttsText.WriteString(f.Text)
		}
		return a.PushFrame(frame, direction)
	case *frames.LLMThoughtStartFrame:
		a.thinking = true
		a.thoughtAppend = f.AppendToContext
		a.thoughtText.Reset()
		return a.PushFrame(frame, direction)
	case *frames.LLMThoughtTextFrame:
		if a.thinking {
			a.thoughtText.WriteString(f.Text)
		}
		return a.PushFrame(frame, direction)
	case *frames.LLMThoughtEndFrame:
		a.endThought(f.Signature)
		return a.PushFrame(frame, direction)
	case *frames.LLMRunFrame:
		return a.PushContextFrame(frames.Upstream)
	case *frames.LLMMessagesAppendFrame:
		a.applyMessagesFrame(frame)
		if f.RunLLM {
			a.pushOwed = true
		}
		if a.pushOwed {
			return a.maybePushContext()
		}
		return nil
	case *frames.LLMMessagesUpdateFrame, *frames.LLMMessagesTransformFrame, *frames.LLMSetToolsFrame, *frames.LLMSetToolChoiceFrame:
		if _, run := a.applyMessagesFrame(frame); run {
			return a.PushContextFrame(frames.Upstream)
		}
		return nil
	case *frames.FunctionCallsStartedFrame:
		for _, call := range f.FunctionCalls {
			if _, done := a.finished[call.ToolCallID]; done {
				continue
			}
			if _, ok := a.calls[call.ToolCallID]; !ok {
				a.calls[call.ToolCallID] = nil
			}
		}
		return nil
	case *frames.FunctionCallInProgressFrame:
		return a.handleCallInProgress(f)
	case *frames.FunctionCallResultFrame:
		return a.handleCallResult(f)
	case *frames.FunctionCallCancelFrame:
		return a.handleCallCancel(f)
	case *frames.LLMSummarizeContextFrame:
		a.requestSummary(true)
		return a.PushFrame(frame, direction)
	case *frames.UserStartedSpeakingFrame:
		a.userSpeaking = true
		return a.PushFrame(frame, direction)
	case *frames.UserStoppedSpeakingFrame:
		a.userSpeaking = false
		if err := a.PushFrame(frame, direction); err != nil {
			return err
		}
		if a.pushOwed {
			return a.maybePushContext()
		}
		return nil
	case *frames.BotStartedSpeakingFrame:
		a.botSpeaking = true
		return a.PushFrame(frame, direction)
	case *frames.BotStoppedSpeakingFrame:
		a.botSpeaking = false
		if err := a.PushFrame(frame, direction); err != nil {
			return err
		}
		if a.pushOnBotStopped && !a.userSpeaking {
			return a.pushContextUpstream()
		}
		return nil
	}
	return a.PushFrame(frame, direction)
}

func (a *LLMAssistantAggregator) startTurn() {
	if a.turnActive {
		return
	}
	a.turnActive = true
	a.turnStart = time.Now()
}

func (a *LLMAssistantAggregator) resetResponse() {
	a.depth = 0
	a.llmText.Reset()
	a.ttsText.Reset()
	a.thinking = false
	a.thoughtAppend = false
	a.thoughtText.Reset()
	a.pushOwed = false
	a.pushOnBotStopped = false
}

func (a *LLMAssistantAggregator) commit(interrupted bool) {
	if !a.turnActive {
		return
	}
	a.turnActive = false
	text := a.ttsText.String()
	if text == "" {
		text = a.llmText.String()
	}
	a.ttsText.Reset()
	a.llmText.Reset()

	if strings.TrimSpace(text) != "" {
		a.context.Add(llm.Assistant(text))
		if err := a.PushContextFrame(frames.Downstream); err != nil {
			logger.Error("[%s] context push failed: %v", a.Name(), err)
		}
		_ = a.PushFrame(frames.NewLLMContextAssistantTimestampFrame(time.Now()), frames.Downstream)
		started := a.turnStart
		if err := a.BroadcastFrame(nil, func() frames.Frame { return frames.NewLLMContextAssistantTurnFrame(text, started) }); err != nil {
			logger.Error("[%s] assistant turn broadcast failed: %v", a.Name(), err)
		}
		a.requestSummary(false)
	}
	if a.params.OnAssistantTurnStopped != nil {
		a.params.OnAssistantTurnStopped(text, interrupted)
	}
}

func (a *LLMAssistantAggregator) endThought(signature string) {
	if !a.thinking {
		return
	}
	text := a.thoughtText.String()
	if a.thoughtAppend && (text != "" || signature != "") {
		a.context.Add(llm.Message{
			Role:  llm.RoleAssistant,
			Parts: []llm.Part{llm.ThoughtPart{Text: text, Signature: signature}},
		})
	}
	a.thinking = false
	a.thoughtAppend = false
	a.thoughtText.Reset()
	if a.params.OnAssistantThought != nil {
		a.params.OnAssistantThought(text)
	}
}

func (a *LLMAssistantAggregator) handleCallInProgress(f *frames.FunctionCallInProgressFrame) error {
	args := []byte("{}")
	if len(f.Arguments) > 0 {
		if encoded, err := json.Marshal(f.Arguments); err == nil {
			args = encoded
		} else {
			logger.Warn("[%s] encoding arguments of %s failed: %v", a.Name(), f.FunctionName, err)
		}
	}
	if done, ok := a.finished[f.ToolCallID]; ok {
		if !a.context.ReplaceToolResult(f.ToolCallID, done.content, false) {
			a.context.Add(
				llm.ToolCalls(llm.ToolCallPart{ID: f.ToolCallID, Name: f.FunctionName, Arguments: args, Signature: f.Signature}),
				llm.ToolResult(f.ToolCallID, f.FunctionName, done.content),
			)
		}
		if !done.runLLM {
			return nil
		}
		a.finished[f.ToolCallID] = finishedCall{content: done.content}
		if a.userSpeaking || a.siblingsPending(f.GroupID, f.ToolCallID) {
			return nil
		}
		a.pushOwed = true
		return a.maybePushContext()
	}
	a.context.Add(
		llm.ToolCalls(llm.ToolCallPart{ID: f.ToolCallID, Name: f.FunctionName, Arguments: args, Signature: f.Signature}),
		llm.ToolResult(f.ToolCallID, f.FunctionName, toolResultInProgress),
	)
	a.calls[f.ToolCallID] = f
	a.requestSummary(false)
	return nil
}

func (a *LLMAssistantAggregator) markFinished(id string, call finishedCall) {
	if _, ok := a.finished[id]; !ok {
		if len(a.finishedOrder) >= finishedCallsLimit {
			delete(a.finished, a.finishedOrder[0])
			a.finishedOrder = a.finishedOrder[1:]
		}
		a.finishedOrder = append(a.finishedOrder, id)
	}
	a.finished[id] = call
}

func (a *LLMAssistantAggregator) handleCallResult(f *frames.FunctionCallResultFrame) error {
	if _, done := a.finished[f.ToolCallID]; done {
		logger.Debug("[%s] ignoring result for already settled function call %s (%s)", a.Name(), f.FunctionName, f.ToolCallID)
		return nil
	}
	inProgress, known := a.calls[f.ToolCallID]
	content := encodeToolResult(f.Result)
	if !known || inProgress == nil {
		logger.Warn("[%s] result for unknown function call %s (%s)", a.Name(), f.FunctionName, f.ToolCallID)
		if !known && a.context.ReplaceToolResult(f.ToolCallID, content, f.Error != "") {
			logger.Debug("[%s] recorded late result for %s", a.Name(), f.ToolCallID)
		}
		return nil
	}

	props := f.Properties
	if props == nil || props.IsFinal {
		delete(a.calls, f.ToolCallID)
		a.markFinished(f.ToolCallID, finishedCall{content: content})
		if !a.context.ReplaceToolResult(f.ToolCallID, content, f.Error != "") {
			a.context.Add(llm.Developer(fmt.Sprintf("Result of function %s (call %s): %s", f.FunctionName, f.ToolCallID, content)))
		}
	} else if f.Result != nil {
		a.context.Add(llm.Developer(fmt.Sprintf("Intermediate result of function %s (call %s): %s", f.FunctionName, f.ToolCallID, content)))
	}

	if f.Result != nil {
		run := true
		if props != nil && props.RunLLM != nil {
			run = *props.RunLLM
		} else if f.RunLLM != nil {
			run = *f.RunLLM
		}
		if run {
			a.pushOwed = true
		}
	}

	var err error
	if a.pushOwed && !a.siblingsPending(inProgress.GroupID, f.ToolCallID) {
		err = a.maybePushContext()
	}

	if props != nil && props.OnContextUpdated != nil {
		cb := props.OnContextUpdated
		go func() {
			defer func() {
				if r := recover(); r != nil {
					logger.Error("[%s] OnContextUpdated panicked: %v", a.Name(), r)
				}
			}()
			cb()
		}()
	}
	a.requestSummary(false)
	return err
}

func (a *LLMAssistantAggregator) handleCallCancel(f *frames.FunctionCallCancelFrame) error {
	if _, done := a.finished[f.ToolCallID]; done {
		return nil
	}
	inProgress := a.calls[f.ToolCallID]
	delete(a.calls, f.ToolCallID)
	if inProgress == nil {
		a.markFinished(f.ToolCallID, finishedCall{content: llm.ToolResultInterrupted, runLLM: f.RunLLM})
		return nil
	}
	a.markFinished(f.ToolCallID, finishedCall{content: llm.ToolResultInterrupted})
	a.context.ReplaceToolResult(f.ToolCallID, llm.ToolResultInterrupted, false)
	if !f.RunLLM || a.userSpeaking || a.siblingsPending(inProgress.GroupID, f.ToolCallID) {
		return nil
	}
	a.pushOwed = true
	return a.maybePushContext()
}

func (a *LLMAssistantAggregator) siblingsPending(groupID, self string) bool {
	for id, call := range a.calls {
		if id == self {
			continue
		}
		if call == nil || groupID == "" || call.GroupID == groupID {
			return true
		}
	}
	return false
}

func (a *LLMAssistantAggregator) maybePushContext() error {
	if a.userSpeaking {
		return nil
	}
	if a.botSpeaking {
		a.pushOnBotStopped = true
		return nil
	}
	return a.pushContextUpstream()
}

func (a *LLMAssistantAggregator) pushContextUpstream() error {
	a.pushOwed = false
	a.pushOnBotStopped = false
	return a.PushContextFrame(frames.Upstream)
}

func (a *LLMAssistantAggregator) requestSummary(force bool) {
	if a.summarizer == nil || (!force && !a.params.EnableAutoContextSummarization) {
		return
	}
	snapshot := a.context.Snapshot()
	if !force && !a.summarizer.ShouldAutoSummarize(snapshot) {
		return
	}
	a.summarizer.Start(a.BaseProcessor.Context(), snapshot, func(res SummaryResult) {
		_ = a.QueueFrame(newSelfFrame(selfSummary, 0, res), frames.Downstream)
	})
}

func (a *LLMAssistantAggregator) cancelSummary() {
	if a.summarizer != nil {
		a.summarizer.Cancel()
	}
}

func (a *LLMAssistantAggregator) Reset() error {
	a.resetResponse()
	a.turnActive = false
	return a.LLMContextAggregator.Reset()
}

func encodeToolResult(result any) string {
	switch v := result.(type) {
	case nil:
		return toolResultCompleted
	case string:
		encoded, _ := json.Marshal(v)
		return string(encoded)
	case json.RawMessage:
		return string(v)
	case []byte:
		return string(v)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return fmt.Sprint(result)
	}
	return string(encoded)
}
