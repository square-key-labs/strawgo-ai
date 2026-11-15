package aggregators

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	TurnCompleteMarker        = "✓"
	TurnIncompleteShortMarker = "○"
	TurnIncompleteLongMarker  = "◐"
)

const (
	defaultShortTimeout = 5 * time.Second
	defaultLongTimeout  = 10 * time.Second

	defaultShortPrompt = `The user paused briefly. Generate a brief, natural prompt to encourage them to continue.

IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given time to continue.

Your response should:
- Be contextually relevant to what was just discussed
- Sound natural and conversational
- Be very concise (1 sentence max)
- Gently prompt them to continue

Example format: ✓ Go ahead, I'm listening.

Generate your ✓ response now.`

	defaultLongPrompt = `The user has been quiet for a while. Generate a friendly check-in message.

IMPORTANT: You MUST respond with ✓ followed by your message. Do NOT output ○ or ◐ - the user has already been given plenty of time.

Your response should:
- Acknowledge they might be thinking or busy
- Offer to help or continue when ready
- Be warm and understanding
- Be brief (1 sentence)

Example format: ✓ No rush! Let me know when you're ready to continue.

Generate your ✓ response now.`

	UserTurnCompletionInstructions = `
CRITICAL INSTRUCTION - MANDATORY RESPONSE FORMAT:
Every single response MUST begin with a turn completion indicator. This is not optional.

TURN COMPLETION DECISION FRAMEWORK:
Ask yourself: "Has the user provided enough information for me to give a meaningful, substantive response?"

Mark as COMPLETE (✓) when:
- The user has answered your question with actual content
- The user has made a complete request or statement
- The user has provided all necessary information for you to respond meaningfully
- The conversation can naturally progress to your substantive response

Mark as INCOMPLETE SHORT (○) when the user will likely continue soon:
- The user was clearly cut off mid-sentence or mid-word
- The user is in the middle of a thought that got interrupted
- Brief technical interruption (they'll resume in a few seconds)

Mark as INCOMPLETE LONG (◐) when the user needs more time:
- The user explicitly asks for time: "let me think", "give me a minute", "hold on"
- The user is clearly pondering or deliberating: "hmm", "well...", "that's a good question"
- The user acknowledged but hasn't answered yet: "That's interesting..."
- The response feels like a preamble before the actual answer

RESPOND in one of these three formats:
1. If COMPLETE: ` + "`✓`" + ` followed by a space and your full substantive response
2. If INCOMPLETE SHORT: ONLY the character ` + "`○`" + ` (user will continue in a few seconds)
3. If INCOMPLETE LONG: ONLY the character ` + "`◐`" + ` (user needs more time to think)

KEY INSIGHT: Grammatically complete ≠ conversationally complete.
"Yes." is complete. "I'd like to..." is NOT complete.

Never skip the marker. Never explain it. The marker MUST be the first character of your response.`
)

type UserTurnCompletionConfig struct {
	Instructions string

	IncompleteShortTimeout time.Duration

	IncompleteLongTimeout time.Duration

	IncompleteShortPrompt string

	IncompleteLongPrompt string
}

func DefaultUserTurnCompletionConfig() UserTurnCompletionConfig {
	return UserTurnCompletionConfig{
		IncompleteShortTimeout: defaultShortTimeout,
		IncompleteLongTimeout:  defaultLongTimeout,
	}
}

func WrapSystemPromptWithTurnCompletion(systemPrompt string, cfg UserTurnCompletionConfig) string {
	instructions := cfg.Instructions
	if instructions == "" {
		instructions = UserTurnCompletionInstructions
	}
	if systemPrompt == "" {
		return instructions
	}
	return systemPrompt + "\n\n" + instructions
}

type UserTurnCompletionProcessor struct {
	*processors.BaseProcessor

	context *llm.Context
	cfg     UserTurnCompletionConfig

	turnTextBuffer    string
	turnSuppressed    bool
	turnCompleteFound bool

	mu       sync.Mutex
	timer    *time.Timer
	timerGen uint64

	log *logger.Logger
}

func NewUserTurnCompletionProcessor(ctx *llm.Context, cfg UserTurnCompletionConfig) *UserTurnCompletionProcessor {
	if cfg.IncompleteShortTimeout == 0 {
		cfg.IncompleteShortTimeout = defaultShortTimeout
	}
	if cfg.IncompleteLongTimeout == 0 {
		cfg.IncompleteLongTimeout = defaultLongTimeout
	}
	if cfg.IncompleteShortPrompt == "" {
		cfg.IncompleteShortPrompt = defaultShortPrompt
	}
	if cfg.IncompleteLongPrompt == "" {
		cfg.IncompleteLongPrompt = defaultLongPrompt
	}

	p := &UserTurnCompletionProcessor{
		context: ctx,
		cfg:     cfg,
		log:     logger.WithPrefix("UserTurnCompletion"),
	}
	p.BaseProcessor = processors.NewBaseProcessor("UserTurnCompletionProcessor", p)
	return p
}

func (p *UserTurnCompletionProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		p.cancelTimer()
		p.turnReset()
		return p.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		return p.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		hadBuffer := p.turnTextBuffer != ""
		markerFound := p.turnCompleteFound || p.turnSuppressed
		wasSupressed := p.turnSuppressed
		buffer := p.turnTextBuffer

		if !wasSupressed {
			p.cancelTimer()
		}
		p.turnReset()

		if !markerFound && hadBuffer {
			p.log.Warn("filter_incomplete_user_turns: LLM response had no turn completion marker (✓/○/◐) — pushing buffered text anyway. Is the system prompt missing turn completion instructions?")
			if err := p.PushFrame(frames.NewLLMTextFrame(buffer), frames.Downstream); err != nil {
				return err
			}
		}
		return p.PushFrame(frame, direction)
	}

	if direction == frames.Downstream {
		if llmFrame, ok := frame.(*frames.LLMTextFrame); ok {
			return p.processToken(llmFrame.Text)
		}
		if textFrame, ok := frame.(*frames.TextFrame); ok {
			return p.processToken(textFrame.Text)
		}
	}

	return p.PushFrame(frame, direction)
}

func (p *UserTurnCompletionProcessor) processToken(text string) error {
	if p.turnCompleteFound {
		return p.PushFrame(frames.NewLLMTextFrame(text), frames.Downstream)
	}

	if p.turnSuppressed {
		return p.PushFrame(p.newSkipTTSFrame(text), frames.Downstream)
	}

	p.turnTextBuffer += text

	if strings.Contains(p.turnTextBuffer, TurnIncompleteShortMarker) {
		f := p.newSkipTTSFrame(p.turnTextBuffer)
		p.turnTextBuffer = ""
		p.turnSuppressed = true
		p.startTimer(p.cfg.IncompleteShortTimeout, p.cfg.IncompleteShortPrompt)
		return p.PushFrame(f, frames.Downstream)
	}

	if strings.Contains(p.turnTextBuffer, TurnIncompleteLongMarker) {
		f := p.newSkipTTSFrame(p.turnTextBuffer)
		p.turnTextBuffer = ""
		p.turnSuppressed = true
		p.startTimer(p.cfg.IncompleteLongTimeout, p.cfg.IncompleteLongPrompt)
		return p.PushFrame(f, frames.Downstream)
	}

	if idx := strings.Index(p.turnTextBuffer, TurnCompleteMarker); idx >= 0 {
		markerEnd := idx + len(TurnCompleteMarker)

		prefix := p.turnTextBuffer[:markerEnd]
		if err := p.PushFrame(p.newSkipTTSFrame(prefix), frames.Downstream); err != nil {
			return err
		}

		remainder := strings.TrimLeft(p.turnTextBuffer[markerEnd:], " ")
		p.turnCompleteFound = true
		p.turnTextBuffer = ""

		if remainder != "" {
			return p.PushFrame(frames.NewLLMTextFrame(remainder), frames.Downstream)
		}
		return nil
	}

	return nil
}

func (p *UserTurnCompletionProcessor) newSkipTTSFrame(text string) *frames.LLMTextFrame {
	f := frames.NewLLMTextFrame(text)
	f.SkipTTS = true
	return f
}

func (p *UserTurnCompletionProcessor) startTimer(timeout time.Duration, prompt string) {
	p.mu.Lock()
	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
	}
	p.timerGen++
	gen := p.timerGen
	p.timer = time.AfterFunc(timeout, func() {
		p.mu.Lock()
		if gen != p.timerGen {
			p.mu.Unlock()
			return
		}
		p.timer = nil
		p.mu.Unlock()

		p.log.Info("Incomplete turn timeout fired (%v), re-prompting LLM", timeout)
		msgs := []llm.Message{llm.System(prompt)}
		if err := p.PushFrame(frames.NewLLMMessagesAppendFrame(msgs, true), frames.Upstream); err != nil {
			p.log.Error("Failed to push re-prompt LLMMessagesAppendFrame: %v", err)
		}
	})
	p.mu.Unlock()
	p.log.Debug("Turn completion timer started (%v)", timeout)
}

func (p *UserTurnCompletionProcessor) cancelTimer() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.timerGen++
	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
		p.log.Debug("Turn completion timer cancelled")
	}
}

func (p *UserTurnCompletionProcessor) turnReset() {
	p.turnTextBuffer = ""
	p.turnSuppressed = false
	p.turnCompleteFound = false
}
