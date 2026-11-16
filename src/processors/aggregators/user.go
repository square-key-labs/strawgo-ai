package aggregators

import (
	"context"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
)

type UserAggregatorParams struct {
	Strategies          turns.UserTurnStrategies
	UserTurnStopTimeout time.Duration
	UserIdleTimeout     time.Duration
	OnUserTurnStarted   func()
	OnUserTurnStopped   func(text string)
	OnUserIdle          func()
	OnUserMuteChanged   func(muted bool)
	EmptyUserTurn       *turns.EmptyUserTurnConfig
}

type LLMUserAggregator struct {
	*LLMContextAggregator

	params           UserAggregatorParams
	explicit         bool
	turn             *userTurnController
	idle             *UserIdleController
	muted            bool
	turnStart        time.Time
	botSpeaking      bool
	awaitingResponse bool
	emptyTurn        turns.EmptyUserTurnConfig
	interruptedBot   bool
	emptyRecoveries  int
	assistant        *LLMAssistantAggregator
}

func NewLLMUserAggregator(c *llm.Context, strategies turns.UserTurnStrategies) *LLMUserAggregator {
	return NewLLMUserAggregatorWithParams(c, UserAggregatorParams{Strategies: strategies})
}

func NewLLMUserAggregatorWithParams(c *llm.Context, params UserAggregatorParams) *LLMUserAggregator {
	if params.UserTurnStopTimeout == 0 {
		params.UserTurnStopTimeout = DefaultUserTurnStopTimeout
	}
	emptyTurn := turns.DefaultEmptyUserTurnConfig()
	if params.EmptyUserTurn != nil {
		emptyTurn = *params.EmptyUserTurn
	}
	u := &LLMUserAggregator{params: params, explicit: !params.Strategies.Empty(), emptyTurn: emptyTurn}
	u.LLMContextAggregator = NewLLMContextAggregator("LLMUserAggregator", c, llm.RoleUser, u)

	strategies := params.Strategies
	if strategies.Empty() {
		strategies = turns.DefaultUserTurnStrategies()
	}
	u.turn = newUserTurnController(strategies, params.UserTurnStopTimeout, userTurnHooks{
		started:          u.onUserTurnStarted,
		stopped:          u.onUserTurnStopped,
		resetAggregation: func() { _ = u.LLMContextAggregator.Reset() },
		wake:             func() { _ = u.QueueFrame(newSelfFrame(selfTurnTimer, 0, nil), frames.Downstream) },
		turnMetrics:      u.pushTurnMetrics,
	})
	u.idle = NewUserIdleController(params.UserIdleTimeout, func() {
		_ = u.QueueFrame(newSelfFrame(selfUserIdle, 0, nil), frames.Downstream)
	})
	return u
}

func (u *LLMUserAggregator) Stop() error {
	u.turn.close()
	u.idle.Stop()
	return u.BaseProcessor.Stop()
}

func (u *LLMUserAggregator) UserTurnActive() bool {
	return u.turn.active()
}

func (u *LLMUserAggregator) Muted() bool {
	return u.muted
}

func (u *LLMUserAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *selfFrame:
		return u.handleSelf(f)
	case *frames.StartFrame:
		if !f.TurnStrategies.Empty() {
			u.turn.setStrategies(f.TurnStrategies)
		}
		return u.PushFrame(frame, direction)
	case *frames.EndFrame, *frames.CancelFrame:
		u.turn.close()
		u.idle.Stop()
		return u.PushFrame(frame, direction)
	case *frames.ServiceMetadataFrame:
		if f.UserTurnStrategies != nil && !u.explicit && !f.UserTurnStrategies.Empty() {
			u.turn.setStrategies(*f.UserTurnStrategies)
		}
		return u.PushFrame(frame, direction)
	}

	if u.updateMute(frame) {
		if !isSpeakingState(frame) {
			return nil
		}
		u.turn.trackSpeaking(frame)
		u.idle.ProcessFrame(frame)
		return u.PushFrame(frame, direction)
	}

	var err error
	switch f := frame.(type) {
	case *frames.TranscriptionFrame:
		if f.IsFinal && strings.TrimSpace(f.Text) != "" {
			u.AppendToAggregation(f.Text)
		}
	case *frames.InterimTranscriptionFrame, *frames.TranslationFrame, *frames.EagerTranscriptionFrame:
	case *frames.LLMRunFrame:
		err = u.pushContext()
	case *frames.LLMContextFrame:
		if direction == frames.Upstream && f.Context != nil {
			err = u.pushContextOf(f.Context)
		} else {
			err = u.PushFrame(frame, direction)
		}
	case *frames.LLMSetToolsFrame:
		u.applyMessagesFrame(frame)
		err = u.PushFrame(frame, direction)
	case *frames.LLMMessagesAppendFrame, *frames.LLMMessagesUpdateFrame, *frames.LLMMessagesTransformFrame, *frames.LLMSetToolChoiceFrame:
		if _, run := u.applyMessagesFrame(frame); run {
			err = u.pushContext()
		}
	case *frames.BotStartedSpeakingFrame:
		u.botSpeaking = true
		err = u.PushFrame(frame, direction)
	case *frames.BotStoppedSpeakingFrame:
		u.botSpeaking = false
		u.awaitingResponse = false
		err = u.PushFrame(frame, direction)
	default:
		err = u.PushFrame(frame, direction)
	}

	u.turn.process(frame)
	u.idle.ProcessFrame(frame)
	return err
}

func (u *LLMUserAggregator) handleSelf(f *selfFrame) error {
	switch f.kind {
	case selfTurnTimer:
		u.turn.process(nil)
	case selfUserIdle:
		if u.params.OnUserIdle != nil {
			u.params.OnUserIdle()
		}
		return u.PushFrame(frames.NewUserIdleTimeoutFrame(), frames.Downstream)
	}
	return nil
}

func (u *LLMUserAggregator) updateMute(frame frames.Frame) bool {
	suppress := false
	if u.muted {
		switch frame.(type) {
		case *frames.InterruptionFrame,
			*frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
			*frames.VADUserStartedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame,
			*frames.ProposedUserStartedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame,
			*frames.AudioFrame, *frames.TranscriptionFrame, *frames.InterimTranscriptionFrame,
			*frames.EagerTranscriptionFrame, *frames.UserBackchannelFrame:
			suppress = true
		}
	}

	next := false
	for _, s := range u.turn.strategies.MuteStrategies {
		if s.ShouldMute(frame) {
			next = true
		}
	}
	if next == u.muted {
		return suppress
	}
	u.muted = next
	if u.params.OnUserMuteChanged != nil {
		u.params.OnUserMuteChanged(next)
	}
	build := func() frames.Frame { return frames.NewUserMuteStoppedFrame() }
	if next {
		build = func() frames.Frame { return frames.NewUserMuteStartedFrame() }
	}
	if err := u.BroadcastFrame(nil, build); err != nil {
		logger.Error("[%s] mute broadcast failed: %v", u.Name(), err)
	}
	return suppress
}

func isSpeakingState(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame,
		*frames.VADUserStartedSpeakingFrame, *frames.VADUserStoppedSpeakingFrame:
		return true
	}
	return false
}

func (u *LLMUserAggregator) onUserTurnStarted(_ user_start.UserTurnStartStrategy, enableInterruptions bool) {
	u.turnStart = time.Now()
	u.interruptedBot = enableInterruptions && u.InterruptionsAllowed() && !u.idle.WaitingForUser()
	u.idle.UserTurnStarted()
	if enableInterruptions && u.InterruptionsAllowed() && (u.botSpeaking || u.awaitingResponse) {
		u.awaitingResponse = false
		if err := u.BroadcastInterruption(nil); err != nil {
			logger.Error("[%s] interruption broadcast failed: %v", u.Name(), err)
		}
	}
	if u.params.OnUserTurnStarted != nil {
		u.params.OnUserTurnStarted()
	}
}

func (u *LLMUserAggregator) onUserTurnStopped(_ bool) {
	u.idle.UserTurnStopped()
	text := strings.TrimSpace(u.AggregationString())
	_ = u.LLMContextAggregator.Reset()
	interrupted := u.interruptedBot
	u.interruptedBot = false
	if text == "" {
		if !u.recoverEmptyTurn(interrupted) {
			u.idle.WaitForUser()
		}
		return
	}
	u.emptyRecoveries = 0
	if u.assistant != nil {
		u.assistant.discardInterruptedSpeech()
	}

	u.context.Add(llm.User(text))
	started := u.turnStart
	if err := u.BroadcastFrame(nil, func() frames.Frame { return frames.NewLLMContextUserTurnFrame(text, started) }); err != nil {
		logger.Error("[%s] user turn broadcast failed: %v", u.Name(), err)
	}
	if err := u.pushContext(); err != nil {
		logger.Error("[%s] context push failed: %v", u.Name(), err)
	}
	if u.params.OnUserTurnStopped != nil {
		u.params.OnUserTurnStopped(text)
	}
}

func (u *LLMUserAggregator) recoverEmptyTurn(interrupted bool) bool {
	cfg := u.emptyTurn
	resume := interrupted && cfg.ResumeInterruptedSpeech && !cfg.Disabled && u.assistant != nil
	if !resume && u.assistant != nil {
		u.assistant.discardInterruptedSpeech()
	}
	prompt := cfg.Prompt(interrupted)
	if !resume && prompt == "" {
		return false
	}
	if u.emptyRecoveries >= cfg.MaxRecoveries() {
		logger.Debug("[%s] empty user turn left unanswered (too many in a row)", u.Name())
		return false
	}
	if u.muted || u.idle.FunctionCallsInProgress() {
		return false
	}
	if resume {
		if text, ok := u.assistant.takeInterruptedSpeech(); ok {
			logger.Debug("[%s] empty user turn interrupted the bot, resuming its speech", u.Name())
			u.emptyRecoveries++
			u.awaitingResponse = true
			if err := u.PushFrame(frames.NewTTSSpeakFrame(text), frames.Downstream); err != nil {
				logger.Error("[%s] resume push failed: %v", u.Name(), err)
			}
			return true
		}
	}
	if prompt == "" {
		return false
	}
	logger.Debug("[%s] empty user turn (interrupted=%v), running LLM", u.Name(), interrupted)
	u.emptyRecoveries++
	u.context.Add(llm.Developer(prompt))
	if err := u.pushContext(); err != nil {
		logger.Error("[%s] context push failed: %v", u.Name(), err)
	}
	return true
}

func (u *LLMUserAggregator) pushTurnMetrics(m turn.TurnMetrics) {
	err := u.PushFrame(frames.NewMetricsFrame(frames.TurnMetricsData{
		MetricsBase:       frames.MetricsBase{Processor: u.Name()},
		IsComplete:        m.IsComplete,
		Probability:       m.Probability,
		E2EProcessingTime: time.Duration(m.TotalTimeMs * float64(time.Millisecond)),
	}), frames.Downstream)
	if err != nil {
		logger.Error("[%s] turn metrics push failed: %v", u.Name(), err)
	}
}

func (u *LLMUserAggregator) pushContext() error {
	return u.pushContextOf(u.context)
}

func (u *LLMUserAggregator) pushContextOf(c *llm.Context) error {
	u.awaitingResponse = true
	return u.PushFrame(frames.NewLLMContextFrame(c), frames.Downstream)
}

func (u *LLMUserAggregator) Reset() error {
	return u.LLMContextAggregator.Reset()
}
