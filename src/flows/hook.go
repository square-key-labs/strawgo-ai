package flows

import (
	"context"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type batch struct {
	open       int
	wantRun    bool
	transition *pendingTransition
}

type Processor struct {
	*processors.BaseProcessor
	manager   atomic.Pointer[Manager]
	m         *Manager
	batches   map[string]*batch
	callBatch map[string]string
	early     map[string]bool
}

func NewProcessor() *Processor {
	h := &Processor{
		batches:   make(map[string]*batch),
		callBatch: make(map[string]string),
		early:     make(map[string]bool),
	}
	h.BaseProcessor = processors.NewBaseProcessor("FlowProcessor", h)
	return h
}

func (h *Processor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if h.m == nil {
		h.m = h.manager.Load()
	}
	if direction != frames.Downstream || h.m == nil {
		return h.PushFrame(frame, direction)
	}
	switch f := frame.(type) {
	case *frames.StartFrame:
		h.m.hookStarted.Store(true)
	case *frames.FunctionCallInProgressFrame:
		h.inProgress(f)
	case *frames.FunctionCallResultFrame:
		h.result(f)
	case *frames.FunctionCallCancelFrame:
		return h.cancelled(f)
	case *frames.EndFrame, *frames.CancelFrame:
		clear(h.batches)
		clear(h.callBatch)
		clear(h.early)
	}
	return h.PushFrame(frame, direction)
}

func (h *Processor) inProgress(f *frames.FunctionCallInProgressFrame) {
	if h.early[f.ToolCallID] {
		delete(h.early, f.ToolCallID)
		return
	}
	key := f.GroupID
	if key == "" {
		key = f.ToolCallID
	}
	b := h.batches[key]
	if b == nil {
		b = &batch{}
		h.batches[key] = b
	}
	b.open++
	h.callBatch[f.ToolCallID] = key
}

func (h *Processor) resolve(toolCallID string) (*batch, bool) {
	key, ok := h.callBatch[toolCallID]
	if !ok {
		return nil, false
	}
	delete(h.callBatch, toolCallID)
	b := h.batches[key]
	b.open--
	if b.open == 0 {
		delete(h.batches, key)
	}
	return b, true
}

func (h *Processor) result(f *frames.FunctionCallResultFrame) {
	if f.Properties != nil && !f.Properties.IsFinal {
		return
	}
	b, ok := h.resolve(f.ToolCallID)
	if !ok {
		return
	}
	props := frames.NewFunctionCallResultProperties()
	if f.Properties != nil {
		props = *f.Properties
	}
	want := true
	if props.RunLLM != nil {
		want = *props.RunLLM
	} else if f.RunLLM != nil {
		want = *f.RunLLM
	}
	b.wantRun = b.wantRun || want
	if t, ok := h.m.takePending(f.ToolCallID); ok {
		b.transition = &t
	}

	run := false
	if b.open == 0 {
		if t := b.transition; t != nil {
			prev := props.OnContextUpdated
			props.OnContextUpdated = func() {
				if prev != nil {
					prev()
				}
				h.m.scheduleTransition(*t)
			}
		} else {
			run = b.wantRun
		}
	}
	props.RunLLM = &run
	f.Properties = &props
}

func (h *Processor) cancelled(f *frames.FunctionCallCancelFrame) error {
	h.m.takePending(f.ToolCallID)
	b, ok := h.resolve(f.ToolCallID)
	if !ok {
		h.early[f.ToolCallID] = true
		return h.PushFrame(f, frames.Downstream)
	}
	if err := h.PushFrame(f, frames.Downstream); err != nil {
		return err
	}
	if b.open > 0 || b.transition == nil {
		return nil
	}
	t := *b.transition
	run := false
	barrier := frames.NewLLMMessagesTransformFrame(func(messages []llm.Message) []llm.Message {
		h.m.scheduleTransition(t)
		return messages
	}, &run)
	barrier.SetSurvivesInterruption(true)
	return h.PushFrame(barrier, frames.Downstream)
}
