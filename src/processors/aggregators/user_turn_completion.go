package aggregators

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// Turn completion markers used by UserTurnCompletionProcessor.
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

	// UserTurnCompletionInstructions is the system prompt block injected by
	// WrapSystemPromptWithTurnCompletion. Instructs the LLM to prefix every
	// response with a turn-completion marker (✓/○/◐).
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

// UserTurnCompletionConfig configures the UserTurnCompletionProcessor.
type UserTurnCompletionConfig struct {
	// Instructions is appended to the system prompt. Empty = built-in
	// UserTurnCompletionInstructions block.
	Instructions string

	// IncompleteShortTimeout: wait before re-prompting after ○. Default: 5s.
	IncompleteShortTimeout time.Duration

	// IncompleteLongTimeout: wait before re-prompting after ◐. Default: 10s.
	IncompleteLongTimeout time.Duration

	// IncompleteShortPrompt: system message appended to context and used to
	// re-run the LLM after the short timeout. Empty = built-in default.
	IncompleteShortPrompt string

	// IncompleteLongPrompt: system message used after the long timeout.
	// Empty = built-in default.
	IncompleteLongPrompt string
}

// DefaultUserTurnCompletionConfig returns sensible defaults (5s short timeout, 10s long timeout).
func DefaultUserTurnCompletionConfig() UserTurnCompletionConfig {
	return UserTurnCompletionConfig{
		IncompleteShortTimeout: defaultShortTimeout,
		IncompleteLongTimeout:  defaultLongTimeout,
	}
}

// WrapSystemPromptWithTurnCompletion appends turn-completion instructions to
// the system prompt. Call at pipeline construction time, before NewLLMContext:
//
//	systemPrompt := aggregators.WrapSystemPromptWithTurnCompletion(
//	    "You are a helpful assistant.",
//	    aggregators.DefaultUserTurnCompletionConfig(),
//	)
//	llmContext := services.NewLLMContext(systemPrompt)
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

// UserTurnCompletionProcessor intercepts LLM token output and suppresses TTS
// synthesis when the LLM signals an incomplete user turn (○ or ◐ marker).
// After a configurable timeout it re-prompts the LLM.
//
// Insert between the LLM service and the assistant aggregator:
//
//	pipeline.NewPipeline([]processors.FrameProcessor{
//	    ...,
//	    llmService,
//	    aggregators.NewUserTurnCompletionProcessor(llmContext, cfg),
//	    assistantAgg,
//	    ttsService,
//	    ...,
//	})
type UserTurnCompletionProcessor struct {
	*processors.BaseProcessor

	context *services.LLMContext
	cfg     UserTurnCompletionConfig

	// Turn state — only written/read from HandleFrame's single goroutine.
	// Timer goroutine communicates only via PushFrame (channel op, thread-safe).
	turnTextBuffer    string
	turnSuppressed    bool
	turnCompleteFound bool

	// Timer — protected by mu. Generation counter prevents stale callbacks.
	mu       sync.Mutex
	timer    *time.Timer
	timerGen uint64

	log *logger.Logger
}

// NewUserTurnCompletionProcessor creates a new UserTurnCompletionProcessor.
// ctx is the shared *services.LLMContext (same pointer used by the user and
// assistant aggregators).
func NewUserTurnCompletionProcessor(ctx *services.LLMContext, cfg UserTurnCompletionConfig) *UserTurnCompletionProcessor {
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

// HandleFrame implements processors.ProcessHandler.
func (p *UserTurnCompletionProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// InterruptionFrame: cancel timer, reset state, pass through.
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		p.cancelTimer()
		p.turnReset()
		return p.PushFrame(frame, direction)
	}

	// LLMFullResponseStartFrame: always pass through (never suppressed).
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		return p.PushFrame(frame, direction)
	}

	// LLMFullResponseEndFrame: reset per-response state.
	// IMPORTANT: do NOT cancel the timer when suppressed (○/◐) — the timer must
	// continue running so it can re-prompt the LLM after the user's thinking time.
	// Only cancel if the turn was complete (✓) or there was no marker at all.
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		hadBuffer := p.turnTextBuffer != ""
		markerFound := p.turnCompleteFound || p.turnSuppressed
		wasSupressed := p.turnSuppressed
		buffer := p.turnTextBuffer

		if !wasSupressed {
			// ✓ or no-marker path: cancel any timer (no-op if none running).
			p.cancelTimer()
		}
		// Reset text/flag state; timer keeps running if wasSupressed.
		p.turnReset()

		if !markerFound && hadBuffer {
			p.log.Warn("filter_incomplete_user_turns: LLM response had no turn completion marker (✓/○/◐) — pushing buffered text anyway. Is the system prompt missing turn completion instructions?")
			if err := p.PushFrame(frames.NewLLMTextFrame(buffer), frames.Downstream); err != nil {
				return err
			}
		}
		return p.PushFrame(frame, direction)
	}

	// Downstream text tokens from the LLM — apply marker detection logic.
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

// processToken handles one text token from the LLM stream.
func (p *UserTurnCompletionProcessor) processToken(text string) error {
	// Turn confirmed complete: pass through immediately, no buffering.
	if p.turnCompleteFound {
		return p.PushFrame(frames.NewLLMTextFrame(text), frames.Downstream)
	}

	// Turn suppressed (○/◐ seen): forward with SkipTTS=true — assistant aggregator
	// still stores it in context, TTS services skip synthesis.
	if p.turnSuppressed {
		return p.PushFrame(p.newSkipTTSFrame(text), frames.Downstream)
	}

	// Still searching for marker — accumulate into buffer.
	// Scanning the full buffer each call handles multi-byte Unicode markers that
	// may arrive split across streaming tokens.
	p.turnTextBuffer += text

	// ○ — incomplete short
	if strings.Contains(p.turnTextBuffer, TurnIncompleteShortMarker) {
		f := p.newSkipTTSFrame(p.turnTextBuffer)
		p.turnTextBuffer = ""
		p.turnSuppressed = true
		p.startTimer(p.cfg.IncompleteShortTimeout, p.cfg.IncompleteShortPrompt)
		return p.PushFrame(f, frames.Downstream)
	}

	// ◐ — incomplete long
	if strings.Contains(p.turnTextBuffer, TurnIncompleteLongMarker) {
		f := p.newSkipTTSFrame(p.turnTextBuffer)
		p.turnTextBuffer = ""
		p.turnSuppressed = true
		p.startTimer(p.cfg.IncompleteLongTimeout, p.cfg.IncompleteLongPrompt)
		return p.PushFrame(f, frames.Downstream)
	}

	// ✓ — complete
	if idx := strings.Index(p.turnTextBuffer, TurnCompleteMarker); idx >= 0 {
		markerEnd := idx + len(TurnCompleteMarker)

		// Push the marker prefix with SkipTTS=true (in context, not spoken).
		prefix := p.turnTextBuffer[:markerEnd]
		if err := p.PushFrame(p.newSkipTTSFrame(prefix), frames.Downstream); err != nil {
			return err
		}

		// Push text after the marker normally — this is what TTS will speak.
		remainder := strings.TrimLeft(p.turnTextBuffer[markerEnd:], " ")
		p.turnCompleteFound = true
		p.turnTextBuffer = ""

		if remainder != "" {
			return p.PushFrame(frames.NewLLMTextFrame(remainder), frames.Downstream)
		}
		return nil
	}

	// No marker found yet — keep buffering, nothing pushed downstream.
	return nil
}

// newSkipTTSFrame returns an LLMTextFrame with SkipTTS=true.
func (p *UserTurnCompletionProcessor) newSkipTTSFrame(text string) *frames.LLMTextFrame {
	f := frames.NewLLMTextFrame(text)
	f.SkipTTS = true
	return f
}

// startTimer cancels any previous timer and starts a new re-prompt timer.
// When it fires it appends a system message to the LLM context and pushes
// LLMMessagesAppendFrame{RunLLM:true} upstream, which causes LLMUserAggregator
// to re-run the LLM (same path as function-call re-invocation).
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
		msgs := []services.LLMMessage{{Role: "system", Content: prompt}}
		// Upstream path: UserTurnCompletionProcessor → LLMService (passes through) →
		// LLMUserAggregator.HandleFrame handles LLMMessagesAppendFrame, appends
		// message, and pushes LLMContextFrame downstream to re-run the LLM.
		if err := p.PushFrame(frames.NewLLMMessagesAppendFrame(msgs, true), frames.Upstream); err != nil {
			p.log.Error("Failed to push re-prompt LLMMessagesAppendFrame: %v", err)
		}
	})
	p.mu.Unlock()
	p.log.Debug("Turn completion timer started (%v)", timeout)
}

// cancelTimer stops any running re-prompt timer.
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

// turnReset clears per-turn state. Callers must call cancelTimer() separately.
func (p *UserTurnCompletionProcessor) turnReset() {
	p.turnTextBuffer = ""
	p.turnSuppressed = false
	p.turnCompleteFound = false
}
