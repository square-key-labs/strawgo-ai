package transcript

import (
	"context"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role      Role
	Content   string
	Timestamp time.Time
	UserID    string
}

type Handler func(Message)

type handlers struct {
	mu   sync.Mutex
	list []Handler
}

func (h *handlers) add(fn Handler) {
	if fn == nil {
		return
	}
	h.mu.Lock()
	h.list = append(h.list[:len(h.list):len(h.list)], fn)
	h.mu.Unlock()
}

func (h *handlers) emit(m Message) {
	h.mu.Lock()
	list := h.list
	h.mu.Unlock()
	for _, fn := range list {
		fn(m)
	}
}

type Processor struct {
	handlers  *handlers
	mu        sync.Mutex
	user      *UserProcessor
	assistant *AssistantProcessor
	assistOps []AssistantOption
}

func NewProcessor(assistantOpts ...AssistantOption) *Processor {
	return &Processor{handlers: &handlers{}, assistOps: assistantOpts}
}

func (p *Processor) OnTranscriptUpdate(fn Handler) {
	p.handlers.add(fn)
}

func (p *Processor) User() *UserProcessor {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.user == nil {
		p.user = newUserProcessor(p.handlers)
	}
	return p.user
}

func (p *Processor) Assistant() *AssistantProcessor {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.assistant == nil {
		p.assistant = newAssistantProcessor(p.handlers, p.assistOps...)
	}
	return p.assistant
}

type UserProcessor struct {
	*processors.BaseProcessor
	handlers *handlers
}

func NewUserProcessor() *UserProcessor {
	return newUserProcessor(&handlers{})
}

func newUserProcessor(h *handlers) *UserProcessor {
	u := &UserProcessor{handlers: h}
	u.BaseProcessor = processors.NewBaseProcessor("UserTranscriptProcessor", u)
	return u
}

func (u *UserProcessor) OnTranscriptUpdate(fn Handler) {
	u.handlers.add(fn)
}

func (u *UserProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if f, ok := frame.(*frames.TranscriptionFrame); ok && direction == frames.Downstream && f.IsFinal {
		if text := strings.TrimSpace(f.Text); text != "" {
			ts := f.Timestamp
			if ts.IsZero() {
				ts = time.Now()
			}
			u.handlers.emit(Message{Role: RoleUser, Content: text, Timestamp: ts})
		}
	}
	return u.PushFrame(frame, direction)
}

type AssistantOption func(*AssistantProcessor)

func WithLLMText() AssistantOption {
	return func(a *AssistantProcessor) { a.useLLMText = true }
}

type AssistantProcessor struct {
	*processors.BaseProcessor
	handlers   *handlers
	useLLMText bool

	text    strings.Builder
	startAt time.Time
}

func NewAssistantProcessor(opts ...AssistantOption) *AssistantProcessor {
	return newAssistantProcessor(&handlers{}, opts...)
}

func newAssistantProcessor(h *handlers, opts ...AssistantOption) *AssistantProcessor {
	a := &AssistantProcessor{handlers: h}
	for _, opt := range opts {
		opt(a)
	}
	a.BaseProcessor = processors.NewBaseProcessor("AssistantTranscriptProcessor", a)
	return a
}

func (a *AssistantProcessor) OnTranscriptUpdate(fn Handler) {
	a.handlers.add(fn)
}

func (a *AssistantProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Downstream {
		switch f := frame.(type) {
		case *frames.TTSTextFrame:
			if !a.useLLMText {
				a.appendWord(f.Text)
			}
		case *frames.LLMTextFrame:
			if a.useLLMText {
				a.appendToken(f.Text)
			}
		case *frames.BotStoppedSpeakingFrame, *frames.InterruptionFrame, *frames.EndFrame, *frames.CancelFrame:
			a.flush()
		case *frames.LLMFullResponseEndFrame:
			if a.useLLMText {
				a.flush()
			}
		}
	} else if _, ok := frame.(*frames.InterruptionFrame); ok {
		a.flush()
	}
	return a.PushFrame(frame, direction)
}

func (a *AssistantProcessor) mark() {
	if a.startAt.IsZero() {
		a.startAt = time.Now()
	}
}

func (a *AssistantProcessor) appendWord(text string) {
	if text == "" {
		return
	}
	a.mark()
	if a.text.Len() > 0 && !startsWithSpace(text) && !endsWithSpace(a.text.String()) {
		a.text.WriteByte(' ')
	}
	a.text.WriteString(text)
}

func (a *AssistantProcessor) appendToken(text string) {
	if text == "" {
		return
	}
	a.mark()
	a.text.WriteString(text)
}

func (a *AssistantProcessor) flush() {
	content := strings.TrimSpace(a.text.String())
	startAt := a.startAt
	a.text.Reset()
	a.startAt = time.Time{}
	if content == "" {
		return
	}
	a.handlers.emit(Message{Role: RoleAssistant, Content: content, Timestamp: startAt})
}

func startsWithSpace(s string) bool {
	for _, r := range s {
		return unicode.IsSpace(r)
	}
	return false
}

func endsWithSpace(s string) bool {
	if s == "" {
		return false
	}
	return unicode.IsSpace(rune(s[len(s)-1]))
}
