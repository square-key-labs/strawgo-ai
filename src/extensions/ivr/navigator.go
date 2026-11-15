package ivr

import (
	"context"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const DefaultIVRStopSecs = 2.0

type Option func(*config)

type config struct {
	ivrVAD           frames.VADParams
	conversationVAD  *frames.VADParams
	classifierPrompt string
	onConversation   ConversationHandler
	onStatus         StatusHandler
}

func WithIVRVADParams(params frames.VADParams) Option {
	return func(c *config) { c.ivrVAD = params }
}

func WithConversationVADParams(params frames.VADParams) Option {
	return func(c *config) { c.conversationVAD = &params }
}

func WithClassifierPrompt(prompt string) Option {
	return func(c *config) { c.classifierPrompt = prompt }
}

func OnConversationDetected(h ConversationHandler) Option {
	return func(c *config) { c.onConversation = h }
}

func OnIVRStatusChanged(h StatusHandler) Option {
	return func(c *config) { c.onStatus = h }
}

func NavigationPrompt(goal string) string {
	return strings.Replace(navigationPromptTemplate, "%s", goal, 1)
}

type Navigator struct {
	*pipeline.Pipeline
	processor *Processor
}

func New(llmService processors.FrameProcessor, goal string, opts ...Option) *Navigator {
	cfg := config{
		ivrVAD:           frames.VADParams{StopSecs: DefaultIVRStopSecs},
		classifierPrompt: ClassifierPrompt,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	proc := newProcessor(cfg.classifierPrompt, NavigationPrompt(goal), cfg.ivrVAD, cfg.conversationVAD)
	proc.onConversation = cfg.onConversation
	proc.onStatus = cfg.onStatus
	tap := &contextTap{target: proc}
	tap.BaseProcessor = processors.NewBaseProcessor("IVRContextTap", tap, processors.WithDirectMode())
	return &Navigator{
		Pipeline:  pipeline.NewPipeline([]processors.FrameProcessor{tap, llmService, proc}),
		processor: proc,
	}
}

func (n *Navigator) Processor() *Processor {
	return n.processor
}

func (n *Navigator) OnConversationDetected(h ConversationHandler) {
	n.processor.handlersMu.Lock()
	n.processor.onConversation = h
	n.processor.handlersMu.Unlock()
}

func (n *Navigator) OnIVRStatusChanged(h StatusHandler) {
	n.processor.handlersMu.Lock()
	n.processor.onStatus = h
	n.processor.handlersMu.Unlock()
}

type contextTap struct {
	*processors.BaseProcessor
	target *Processor
}

func (t *contextTap) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if f, ok := frame.(*frames.LLMContextFrame); ok && direction == frames.Downstream && f.Context != nil {
		t.target.setContext(f.Context)
	}
	return t.PushFrame(frame, direction)
}
