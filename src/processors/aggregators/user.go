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
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

const (
	resumeNone = iota
	resumePushed
	resumeSpeaking
)

type UserAggregatorParams struct {
	Strategies          turns.UserTurnStrategies
	UserTurnStopTimeout time.Duration
	UserIdleTimeout     time.Duration
	OnUserTurnStarted   func()
	OnUserTurnStopped   func(text string)
	OnUserIdle          func()
	OnUserMuteChanged   func(muted bool)
	OnUserBackchannel   func(text string)
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
	recoveryPrompts  []string
	ranLLM           bool
	resuming         int
	realtimeLLM      bool
	assistant        *LLMAssistantAggregator
	speculative      *llm.Context
	specVersion      uint64
	confirmed        *llm.Context
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
		started:           u.onUserTurnStarted,
		stopped:           u.onUserTurnStopped,
		speculate:         u.onSpeculation,
		cancelSpeculation: u.cancelSpeculation,
		resetAggregation:  func() { _ = u.LLMContextAggregator.Reset() },
		backchannel:       u.emitBackchannel,
		restoreText:       u.restoreText,
		wake:              func() { _ = u.QueueFrame(newSelfFrame(selfTurnTimer, 0, nil), frames.Downstream) },
		turnMetrics:       u.pushTurnMetrics,
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
		u.dropRecoveryPrompts()
		return u.PushFrame(frame, direction)
	case *frames.LLMServiceMetadataFrame:
		if f.IsRealtimeService != u.realtimeLLM {
			u.realtimeLLM = f.IsRealtimeService
			logger.Debug("[%s] active LLM %s realtime=%v; empty user turn recovery on=%v", u.Name(), f.ServiceName, f.IsRealtimeService, !f.IsRealtimeService)
		}
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
	case *frames.UserBackchannelFrame:
		if u.params.OnUserBackchannel != nil {
			u.params.OnUserBackchannel(f.Text)
		}
		err = u.PushFrame(frame, direction)
	case *frames.LLMRunFrame:
		err = u.pushContext()
	case *frames.UserTurnResponseCancelFrame:
		if f.Context == nil && direction == frames.Downstream {
			f.Context = u.speculative
		}
		err = u.PushFrame(frame, direction)
		if f.Context != nil && f.Context != u.speculative {
			if direction == frames.Upstream && f.Context == u.confirmed && !u.turn.active() {
				u.confirmed = nil
				logger.Debug("[%s] confirmed speculative response withdrawn, answering the committed turn", u.Name())
				err = u.pushContext()
			}
			return err
		}
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
		if u.resuming == resumePushed {
			u.resuming = resumeSpeaking
		}
		err = u.PushFrame(frame, direction)
	case *frames.BotStoppedSpeakingFrame:
		u.botSpeaking = false
		u.awaitingResponse = false
		if u.resuming == resumeSpeaking && !u.turn.active() {
			u.emptyRecoveries = 0
		}
		if u.resuming == resumeSpeaking {
			u.resuming = resumeNone
		}
		err = u.PushFrame(frame, direction)
	default:
		err = u.PushFrame(frame, direction)
	}

	if functionResultRunsLLM(frame) {
		u.ranLLM = true
	}
	u.turn.process(frame)
	u.idle.ProcessFrame(frame)
	return err
}

func functionResultRunsLLM(frame frames.Frame) bool {
	switch f := frame.(type) {
	case *frames.FunctionCallResultFrame:
		props := f.Properties
		if props != nil && !props.IsFinal && f.Result == nil {
			return false
		}
		if props != nil && props.RunLLM != nil {
			return *props.RunLLM
		}
		return f.RunLLM == nil || *f.RunLLM
	case *frames.FunctionCallCancelFrame:
		return f.RunLLM
	}
	return false
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
	u.turn.setMuted(next)
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
	u.ranLLM = false
	u.resuming = resumeNone
	u.speculative = nil
	u.confirmed = nil
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

func (u *LLMUserAggregator) onUserTurnStopped(_ bool, confirmed bool) {
	u.idle.UserTurnStopped()
	text := strings.TrimSpace(u.AggregationString())
	_ = u.LLMContextAggregator.Reset()
	interrupted := u.interruptedBot
	u.interruptedBot = false
	speculative := u.speculative
	if confirmed && (text == "" || speculative == nil || u.context.Snapshot().Version != u.specVersion) {
		logger.Debug("[%s] speculative response no longer answers the turn, running the LLM again", u.Name())
		u.cancelSpeculation()
		confirmed = false
	}
	u.speculative = nil
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
	u.dropRecoveryPrompts()

	u.context.Add(llm.User(text))
	started := u.turnStart
	if err := u.BroadcastFrame(nil, func() frames.Frame { return frames.NewLLMContextUserTurnFrame(text, started) }); err != nil {
		logger.Error("[%s] user turn broadcast failed: %v", u.Name(), err)
	}
	if confirmed {
		u.awaitingResponse = true
		u.ranLLM = true
		u.confirmed = speculative
		if err := u.BroadcastFrame(nil, func() frames.Frame { return frames.NewUserTurnSpeculationConfirmedFrame(speculative, text) }); err != nil {
			logger.Error("[%s] speculation confirmation broadcast failed: %v", u.Name(), err)
		}
	} else if err := u.pushContext(); err != nil {
		logger.Error("[%s] context push failed: %v", u.Name(), err)
	}
	if u.params.OnUserTurnStopped != nil {
		u.params.OnUserTurnStopped(text)
	}
}

func (u *LLMUserAggregator) restoreText(text string) {
	logger.Debug("[%s] user turn keeps the earlier acknowledgement: %q", u.Name(), text)
	u.aggregation = append([]string{text}, u.aggregation...)
}

func (u *LLMUserAggregator) emitBackchannel(text string) {
	logger.Debug("[%s] user backchannel while the bot speaks, not interrupting: %q", u.Name(), text)
	if u.params.OnUserBackchannel != nil {
		u.params.OnUserBackchannel(text)
	}
	if err := u.PushFrame(frames.NewUserBackchannelFrame(text, ""), frames.Downstream); err != nil {
		logger.Error("[%s] backchannel push failed: %v", u.Name(), err)
	}
}

func (u *LLMUserAggregator) onSpeculation(spec user_stop.UserTurnSpeculation) {
	if u.realtimeLLM {
		return
	}
	text := strings.TrimSpace(strings.TrimSpace(u.AggregationString()) + " " + strings.TrimSpace(spec.Text))
	snap := u.context.Snapshot()
	provisional := llm.NewContext(withoutPrompts(append([]llm.Message(nil), snap.Messages...), u.recoveryPrompts)...)
	provisional.SetTools(snap.Tools)
	provisional.SetToolChoice(snap.ToolChoice)
	provisional.Add(llm.User(text))
	u.speculative = provisional
	u.specVersion = snap.Version
	logger.Debug("[%s] eager end of turn, answering speculatively: %q", u.Name(), text)
	started := u.turnStart
	if err := u.BroadcastFrame(nil, func() frames.Frame {
		f := frames.NewLLMContextUserTurnFrame(text, started)
		f.Speculation = true
		return f
	}); err != nil {
		logger.Error("[%s] speculative user turn broadcast failed: %v", u.Name(), err)
	}
	f := frames.NewLLMContextFrame(provisional)
	f.Speculation = true
	if err := u.PushFrame(f, frames.Downstream); err != nil {
		logger.Error("[%s] speculative context push failed: %v", u.Name(), err)
	}
}

func (u *LLMUserAggregator) cancelSpeculation() {
	speculative := u.speculative
	u.speculative = nil
	if speculative == nil {
		return
	}
	if err := u.BroadcastFrame(nil, func() frames.Frame {
		f := frames.NewUserTurnResponseCancelFrame()
		f.Context = speculative
		return f
	}); err != nil {
		logger.Error("[%s] speculation cancel broadcast failed: %v", u.Name(), err)
	}
}

func (u *LLMUserAggregator) recoverEmptyTurn(interrupted bool) bool {
	cfg := u.emptyTurn
	if u.realtimeLLM {
		cfg.Disabled = true
	}
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
	if u.muted || u.idle.FunctionCallsInProgress() || u.botSpeaking || u.awaitingResponse || u.ranLLM {
		if u.assistant != nil {
			u.assistant.discardInterruptedSpeech()
		}
		return false
	}
	if resume {
		if text, ok := u.assistant.takeInterruptedSpeech(); ok {
			logger.Debug("[%s] empty user turn interrupted the bot, resuming its speech", u.Name())
			u.emptyRecoveries++
			u.awaitingResponse = true
			u.resuming = resumePushed
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
	u.recoveryPrompts = append(u.recoveryPrompts, prompt)
	if err := u.pushContext(); err != nil {
		logger.Error("[%s] context push failed: %v", u.Name(), err)
	}
	return true
}

func (u *LLMUserAggregator) dropRecoveryPrompts() {
	if len(u.recoveryPrompts) == 0 {
		return
	}
	pending := u.recoveryPrompts
	u.recoveryPrompts = nil
	u.context.Transform(func(msgs []llm.Message) []llm.Message {
		return withoutPrompts(msgs, pending)
	})
}

func withoutPrompts(msgs []llm.Message, prompts []string) []llm.Message {
	for _, prompt := range prompts {
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == llm.RoleDeveloper && msgs[i].Text() == prompt {
				msgs = append(msgs[:i], msgs[i+1:]...)
				break
			}
		}
	}
	return msgs
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
	u.ranLLM = true
	return u.PushFrame(frames.NewLLMContextFrame(c), frames.Downstream)
}

func (u *LLMUserAggregator) Reset() error {
	return u.LLMContextAggregator.Reset()
}
