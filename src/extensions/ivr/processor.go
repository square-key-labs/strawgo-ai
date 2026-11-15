package ivr

import (
	"context"
	"strings"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/utils/text"
)

type Status string

const (
	StatusDetected  Status = "detected"
	StatusCompleted Status = "completed"
	StatusStuck     Status = "stuck"
	StatusWait      Status = "wait"
)

func (s Status) Valid() bool {
	switch s {
	case StatusDetected, StatusCompleted, StatusStuck, StatusWait:
		return true
	}
	return false
}

const (
	patternDTMF = "dtmf"
	patternMode = "mode"
	patternIVR  = "ivr"
)

type ConversationHandler func(p *Processor, history []llm.Message)

type StatusHandler func(p *Processor, status Status)

type Processor struct {
	*processors.BaseProcessor

	classifierPrompt string
	ivrPrompt        string
	ivrVAD           frames.VADParams
	conversationVAD  *frames.VADParams

	aggregator *text.PatternPairAggregator
	skipTTS    bool

	ctxMu   sync.Mutex
	context *llm.Context

	handlersMu     sync.RWMutex
	onConversation ConversationHandler
	onStatus       StatusHandler
}

func newProcessor(classifierPrompt, ivrPrompt string, ivrVAD frames.VADParams, conversationVAD *frames.VADParams) *Processor {
	agg, _ := text.NewPatternPairAggregator(
		text.Pattern{Type: patternDTMF, Start: "<dtmf>", End: "</dtmf>", Action: text.MatchRemove},
		text.Pattern{Type: patternMode, Start: "<mode>", End: "</mode>", Action: text.MatchRemove},
		text.Pattern{Type: patternIVR, Start: "<ivr>", End: "</ivr>", Action: text.MatchRemove},
	)
	p := &Processor{
		classifierPrompt: classifierPrompt,
		ivrPrompt:        ivrPrompt,
		ivrVAD:           ivrVAD,
		conversationVAD:  conversationVAD,
		aggregator:       agg,
	}
	p.BaseProcessor = processors.NewBaseProcessor("IVRProcessor", p)
	return p
}

func (p *Processor) setContext(c *llm.Context) {
	p.ctxMu.Lock()
	p.context = c
	p.ctxMu.Unlock()
}

func (p *Processor) conversationHistory() []llm.Message {
	p.ctxMu.Lock()
	c := p.context
	p.ctxMu.Unlock()
	if c == nil {
		return nil
	}
	messages := c.Messages()
	if len(messages) <= 1 {
		return nil
	}
	return append([]llm.Message(nil), messages[1:]...)
}

func (p *Processor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Upstream {
		return p.PushFrame(frame, direction)
	}
	switch f := frame.(type) {
	case *frames.StartFrame:
		if err := p.PushFrame(f, direction); err != nil {
			return err
		}
		return p.PushFrame(frames.NewLLMMessagesUpdateFrame([]llm.Message{llm.Developer(p.classifierPrompt)}, false), frames.Upstream)
	case *frames.InterruptionFrame:
		p.aggregator.Reset()
		return p.PushFrame(f, direction)
	case *frames.LLMTextFrame:
		p.skipTTS = f.SkipTTS
		p.aggregator.Write(f.Text, p.handleSegment)
		return nil
	case *frames.LLMFullResponseEndFrame, *frames.EndFrame:
		p.aggregator.Flush(p.handleSegment)
		return p.PushFrame(frame, direction)
	}
	return p.PushFrame(frame, direction)
}

func (p *Processor) handleSegment(seg text.Segment) {
	if !seg.Match {
		out := frames.NewLLMTextFrame(seg.Text)
		out.SkipTTS = p.skipTTS
		_ = p.PushFrame(out, frames.Downstream)
		return
	}
	switch seg.Type {
	case patternDTMF:
		p.handleDTMF(seg.Text)
	case patternIVR:
		p.handleStatus(seg.Text)
	case patternMode:
		p.handleMode(seg.Text)
	}
}

func (p *Processor) handleDTMF(value string) {
	buttons, err := frames.ParseKeypadEntries(strings.TrimSpace(value))
	if err != nil || len(buttons) == 0 {
		logger.Warn("[%s] Invalid DTMF value %q: must be 0-9, * or #", p.Name(), value)
		return
	}
	_ = p.PushFrame(frames.NewOutputDTMFUrgentFrame(buttons...), frames.Downstream)
	p.pushContextText("<dtmf>" + value + "</dtmf>")
}

func (p *Processor) handleStatus(value string) {
	status := Status(strings.ToLower(value))
	if !status.Valid() {
		logger.Warn("[%s] Unknown IVR status %q", p.Name(), value)
		return
	}
	switch status {
	case StatusDetected:
		p.switchToIVR()
	case StatusCompleted, StatusStuck:
		p.emitStatus(status)
	case StatusWait:
		logger.Debug("[%s] IVR waiting for more transcription", p.Name())
	}
	p.pushContextText("<ivr>" + value + "</ivr>")
}

func (p *Processor) handleMode(value string) {
	switch strings.ToLower(value) {
	case "conversation":
		p.switchToConversation()
	case "ivr":
		p.switchToIVR()
	default:
		logger.Warn("[%s] Unknown mode %q", p.Name(), value)
	}
}

func (p *Processor) pushContextText(s string) {
	t := frames.NewTextFrame(s)
	t.SkipTTS = true
	_ = p.PushFrame(t, frames.Downstream)
}

func (p *Processor) switchToIVR() {
	history := p.conversationHistory()
	messages := make([]llm.Message, 0, len(history)+1)
	messages = append(messages, llm.Developer(p.ivrPrompt))
	messages = append(messages, history...)
	_ = p.PushFrame(frames.NewLLMMessagesUpdateFrame(messages, true), frames.Upstream)
	_ = p.PushFrame(frames.NewVADParamsUpdateFrame(p.ivrVAD), frames.Upstream)
	p.emitStatus(StatusDetected)
}

func (p *Processor) switchToConversation() {
	if p.conversationVAD != nil {
		_ = p.PushFrame(frames.NewVADParamsUpdateFrame(*p.conversationVAD), frames.Upstream)
	}
	p.handlersMu.RLock()
	h := p.onConversation
	p.handlersMu.RUnlock()
	if h != nil {
		h(p, p.conversationHistory())
	}
}

func (p *Processor) emitStatus(status Status) {
	p.handlersMu.RLock()
	h := p.onStatus
	p.handlersMu.RUnlock()
	if h != nil {
		h(p, status)
	}
}
