package flows

import (
	"context"
	"fmt"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type Action interface {
	Execute(ctx context.Context, m *Manager) error
}

type ActionHandler func(ctx context.Context, action Custom, m *Manager) error

type TTSSay struct {
	Text        string
	SkipContext bool
}

type EndConversation struct {
	Text        string
	SkipContext bool
}

type FunctionAction struct {
	Fn func(ctx context.Context, m *Manager) error
}

type Custom struct {
	Type   string
	Params map[string]any
}

const (
	kindTTSSay   = "tts_say"
	kindEnd      = "end_conversation"
	kindFunction = "function"
)

func actionKind(a Action) string {
	switch v := a.(type) {
	case TTSSay, *TTSSay:
		return kindTTSSay
	case EndConversation, *EndConversation:
		return kindEnd
	case FunctionAction, *FunctionAction:
		return kindFunction
	case Custom:
		return v.Type
	case *Custom:
		return v.Type
	case registryAction:
		return v.Type
	}
	return fmt.Sprintf("%T", a)
}

func speakFrame(text string, skipContext bool) *frames.TTSSpeakFrame {
	f := frames.NewTTSSpeakFrame(text)
	f.AppendToContext = !skipContext
	return f
}

func (a TTSSay) Execute(ctx context.Context, m *Manager) error {
	if a.Text == "" {
		return fmt.Errorf("tts_say requires text")
	}
	m.acquire()
	if err := m.task.QueueFrames([]frames.Frame{speakFrame(a.Text, a.SkipContext), m.newDoneFrame()}); err != nil {
		m.release()
		return err
	}
	return nil
}

func (a EndConversation) Execute(ctx context.Context, m *Manager) error {
	var list []frames.Frame
	if a.Text != "" {
		list = append(list, speakFrame(a.Text, a.SkipContext))
	}
	return m.task.QueueFrames(append(list, frames.NewEndFrame()))
}

func (a FunctionAction) Execute(ctx context.Context, m *Manager) error {
	if a.Fn == nil {
		return fmt.Errorf("function action requires a function")
	}
	m.acquire()
	f := &functionActionFrame{
		ControlFrame: &frames.ControlFrame{BaseFrame: frames.NewUninterruptibleBaseFrame("FlowFunctionActionFrame")},
		m:            m,
		fn:           a.Fn,
	}
	if err := m.task.QueueFrames([]frames.Frame{f}); err != nil {
		m.release()
		return err
	}
	return nil
}

func (a Custom) Execute(ctx context.Context, m *Manager) error {
	h, ok := m.actionHandler(a.Type)
	if !ok {
		return fmt.Errorf("no handler registered for action type %q", a.Type)
	}
	return h(ctx, a, m)
}

type actionDoneFrame struct {
	*frames.ControlFrame
	m *Manager
}

type functionActionFrame struct {
	*frames.ControlFrame
	m  *Manager
	fn func(ctx context.Context, m *Manager) error
}

func (m *Manager) newDoneFrame() *actionDoneFrame {
	return &actionDoneFrame{
		ControlFrame: &frames.ControlFrame{BaseFrame: frames.NewUninterruptibleBaseFrame("FlowActionDoneFrame")},
		m:            m,
	}
}

func (m *Manager) acquire() {
	m.trackMu.Lock()
	if m.inflight == 0 {
		m.idle = make(chan struct{})
	}
	m.inflight++
	m.trackMu.Unlock()
}

func (m *Manager) release() {
	m.trackMu.Lock()
	if m.inflight > 0 {
		m.inflight--
		if m.inflight == 0 {
			close(m.idle)
		}
	}
	m.trackMu.Unlock()
}

func (m *Manager) waitIdle(ctx context.Context) error {
	m.trackMu.Lock()
	if m.inflight == 0 {
		m.trackMu.Unlock()
		return nil
	}
	idle := m.idle
	m.trackMu.Unlock()
	select {
	case <-idle:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-m.ctx.Done():
		return m.ctx.Err()
	}
}

func needsWait(prev, next string) bool {
	switch prev {
	case kindTTSSay:
		return next != kindTTSSay && next != kindEnd && next != kindFunction
	case kindFunction:
		return true
	}
	return false
}

func (m *Manager) runActions(ctx context.Context, actions []Action) error {
	prev := ""
	for _, a := range actions {
		kind := actionKind(a)
		if needsWait(prev, kind) {
			if err := m.waitIdle(ctx); err != nil {
				return err
			}
		}
		if err := a.Execute(ctx, m); err != nil {
			return fmt.Errorf("flows: action %s: %w", kind, err)
		}
		prev = kind
		if kind == kindEnd {
			return nil
		}
	}
	if needsWait(prev, "") {
		return m.waitIdle(ctx)
	}
	return nil
}

func (m *Manager) runFunctionAction(f *functionActionFrame) {
	go func() {
		defer m.release()
		defer func() {
			if r := recover(); r != nil {
				logger.Error("[flows] function action panicked: %v", r)
			}
		}()
		if err := f.fn(m.workerContext(m.ctx), m); err != nil {
			logger.Error("[flows] function action failed: %v", err)
		}
	}()
}

func (m *Manager) scheduleDeferred(actions []Action) {
	m.trackMu.Lock()
	m.deferred = actions
	m.trackMu.Unlock()
}

func (m *Manager) onBotStoppedSpeaking() {
	m.trackMu.Lock()
	if m.inflight != 0 || len(m.deferred) == 0 {
		m.trackMu.Unlock()
		return
	}
	actions := m.deferred
	m.deferred = nil
	m.trackMu.Unlock()
	m.enqueue(func(ctx context.Context) error {
		return m.runActions(ctx, actions)
	}, false)
}

func (m *Manager) validateActions(node string, actions []Action) error {
	for _, a := range actions {
		if a == nil {
			return fmt.Errorf("flows: node %q has a nil action", node)
		}
		switch v := a.(type) {
		case Custom:
			if _, ok := m.actionHandler(v.Type); !ok {
				return fmt.Errorf("flows: node %q uses action %q, which is not registered", node, v.Type)
			}
		case *Custom:
			if _, ok := m.actionHandler(v.Type); !ok {
				return fmt.Errorf("flows: node %q uses action %q, which is not registered", node, v.Type)
			}
		}
	}
	return nil
}
