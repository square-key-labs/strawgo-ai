package aggregators

import (
	"context"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/turns"
)

const defaultUserAggregationTimeout = 500 * time.Millisecond

type LLMUserAggregator struct {
	*LLMContextAggregator

	turnStrategies turns.UserTurnStrategies

	userSpeaking          bool
	botSpeaking           bool
	userTurnActive        bool
	seenInterimResults    bool
	waitingForAggregation bool
	interruptionSent      bool
	mutedState            bool

	stateMu sync.Mutex

	aggregationCtx    context.Context
	aggregationCancel context.CancelFunc
	aggregationEvent  chan struct{}
}

func NewLLMUserAggregator(context *services.LLMContext, strategies turns.UserTurnStrategies) *LLMUserAggregator {
	u := &LLMUserAggregator{
		turnStrategies:   strategies,
		aggregationEvent: make(chan struct{}, 1),
	}

	u.LLMContextAggregator = NewLLMContextAggregator("LLMUserAggregator", context, "user", u)
	return u
}

func (u *LLMUserAggregator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		u.HandleStartFrame(startFrame)
		if hasTurnStrategies(startFrame.TurnStrategies) {
			u.turnStrategies = startFrame.TurnStrategies
		}

		u.aggregationCtx, u.aggregationCancel = context.WithCancel(ctx)
		go u.aggregationTaskHandler()

		return u.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.EndFrame); ok {
		if u.aggregationCancel != nil {
			u.aggregationCancel()
		}
		return u.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.InterruptionFrame); ok {
		u.HandleInterruptionFrame()
		if err := u.Reset(); err != nil {
			logger.Error("[%s] reset failed on interruption: %v", u.Name(), err)
		}
		return u.PushFrame(frame, direction)
	}

	u.updateBotSpeakingState(frame)

	if err := u.updateMuteState(frame); err != nil {
		logger.Error("[%s] failed to push mute state frame: %v", u.Name(), err)
	}

	if u.shouldSuppressFrame(frame) {
		return nil
	}

	u.updateUserSpeakingState(frame)
	u.handleTurnStart(ctx, frame)
	u.handleTurnStop(frame)

	if transcriptionFrame, ok := frame.(*frames.TranscriptionFrame); ok {
		if transcriptionFrame.Text == "" {
			return nil
		}

		u.stateMu.Lock()
		if transcriptionFrame.IsFinal {
			u.AppendToAggregation(transcriptionFrame.Text)
			u.seenInterimResults = false
		} else {
			u.seenInterimResults = true
		}
		u.stateMu.Unlock()

		if transcriptionFrame.IsFinal {
			select {
			case u.aggregationEvent <- struct{}{}:
			default:
			}

			u.stateMu.Lock()
			shouldPushNow := !u.waitingForAggregation && !u.userSpeaking
			u.stateMu.Unlock()

			if shouldPushNow {
				if err := u.pushAggregation(); err != nil {
					logger.Error("[%s] failed to push aggregation: %v", u.Name(), err)
				}
			}
		}

		return nil
	}

	if appendFrame, ok := frame.(*frames.LLMMessagesAppendFrame); ok {
		if messages, ok := appendFrame.Messages.([]services.LLMMessage); ok {
			u.context.Messages = append(u.context.Messages, messages...)
			if appendFrame.RunLLM {
				return u.PushContextFrame(frames.Downstream)
			}
		}
		return nil
	}

	if updateFrame, ok := frame.(*frames.LLMMessagesUpdateFrame); ok {
		if messages, ok := updateFrame.Messages.([]services.LLMMessage); ok {
			u.context.Messages = messages
			if updateFrame.RunLLM {
				return u.PushContextFrame(frames.Downstream)
			}
		}
		return nil
	}

	if transformFrame, ok := frame.(*frames.LLMMessagesTransformFrame); ok {
		if transform, ok := transformFrame.Transform.(func([]services.LLMMessage) []services.LLMMessage); ok {
			u.context.Messages = transform(u.context.Messages)
			if transformFrame.RunLLM {
				return u.PushContextFrame(frames.Downstream)
			}
		} else {
			logger.Error("[%s] LLMMessagesTransformFrame.Transform has wrong type", u.Name())
		}
		return nil
	}

	return u.PushFrame(frame, direction)
}

func (u *LLMUserAggregator) pushAggregation() error {
	u.stateMu.Lock()
	if len(u.aggregation) == 0 {
		u.stateMu.Unlock()
		return nil
	}
	u.stateMu.Unlock()

	return u.processAggregation()
}

func (u *LLMUserAggregator) processAggregation() error {
	u.stateMu.Lock()
	text := u.AggregationString()
	u.seenInterimResults = false
	u.waitingForAggregation = false
	u.LLMContextAggregator.Reset()
	u.stateMu.Unlock()

	if text == "" {
		return nil
	}

	// Add user message to context
	u.context.AddUserMessage(text)

	// Push context frame downstream to trigger LLM
	return u.PushContextFrame(frames.Downstream)
}

func (u *LLMUserAggregator) aggregationTaskHandler() {
	ticker := time.NewTicker(defaultUserAggregationTimeout / 2)
	defer ticker.Stop()

	for {
		select {
		case <-u.aggregationCtx.Done():
			return

		case <-ticker.C:
			u.handleTurnStop(nil)

			u.stateMu.Lock()
			shouldPush := !u.userSpeaking && len(u.aggregation) > 0
			u.stateMu.Unlock()

			if shouldPush {
				if err := u.pushAggregation(); err != nil {
					logger.Error("[%s] failed to push aggregation on timeout: %v", u.Name(), err)
				}
			}

		case <-u.aggregationEvent:
			continue
		}
	}
}

func (u *LLMUserAggregator) Reset() error {
	u.stateMu.Lock()
	defer u.stateMu.Unlock()

	u.userSpeaking = false
	u.botSpeaking = false
	u.userTurnActive = false
	u.seenInterimResults = false
	u.waitingForAggregation = false
	u.interruptionSent = false
	u.mutedState = false

	for _, strategy := range u.turnStrategies.StartStrategies {
		strategy.Reset()
	}
	for _, strategy := range u.turnStrategies.StopStrategies {
		strategy.Reset()
	}
	for _, strategy := range u.turnStrategies.MuteStrategies {
		strategy.Reset()
	}

	return u.LLMContextAggregator.Reset()
}

func (u *LLMUserAggregator) updateBotSpeakingState(frame frames.Frame) {
	switch frame.(type) {
	case *frames.BotStartedSpeakingFrame, *frames.TTSStartedFrame:
		u.stateMu.Lock()
		u.botSpeaking = true
		u.stateMu.Unlock()
	case *frames.BotStoppedSpeakingFrame:
		u.stateMu.Lock()
		u.botSpeaking = false
		u.stateMu.Unlock()
	}
}

func (u *LLMUserAggregator) updateUserSpeakingState(frame frames.Frame) {
	switch frame.(type) {
	case *frames.UserStartedSpeakingFrame:
		u.stateMu.Lock()
		u.userSpeaking = true
		u.stateMu.Unlock()
	case *frames.UserStoppedSpeakingFrame:
		u.stateMu.Lock()
		u.userSpeaking = false
		u.interruptionSent = false
		u.stateMu.Unlock()
	}
}

func (u *LLMUserAggregator) handleTurnStart(ctx context.Context, frame frames.Frame) {
	u.stateMu.Lock()
	active := u.userTurnActive
	u.stateMu.Unlock()
	if active {
		return
	}

	for _, strategy := range u.turnStrategies.StartStrategies {
		if !strategy.ShouldStart(frame) {
			continue
		}

		u.stateMu.Lock()
		if u.userTurnActive {
			u.stateMu.Unlock()
			return
		}

		u.userTurnActive = true
		shouldInterrupt := u.InterruptionsAllowed() && u.botSpeaking && strategy.EnableInterruptions() && !u.interruptionSent
		if shouldInterrupt {
			u.interruptionSent = true
		}
		u.stateMu.Unlock()

		if shouldInterrupt {
			if err := u.BroadcastInterruption(ctx); err != nil {
				logger.Error("[%s] failed to broadcast interruption: %v", u.Name(), err)
			}
		}

		for _, startStrategy := range u.turnStrategies.StartStrategies {
			startStrategy.Reset()
		}
		return
	}
}

func (u *LLMUserAggregator) handleTurnStop(frame any) {
	u.stateMu.Lock()
	active := u.userTurnActive
	u.stateMu.Unlock()
	if !active {
		return
	}

	for _, strategy := range u.turnStrategies.StopStrategies {
		if !strategy.ShouldStop(frame) {
			continue
		}

		u.stateMu.Lock()
		u.userTurnActive = false
		u.userSpeaking = false
		u.interruptionSent = false
		u.stateMu.Unlock()

		for _, stopStrategy := range u.turnStrategies.StopStrategies {
			stopStrategy.Reset()
		}

		if err := u.pushAggregation(); err != nil {
			logger.Error("[%s] failed to push aggregation on turn stop: %v", u.Name(), err)
		}
		return
	}
}

func (u *LLMUserAggregator) updateMuteState(frame frames.Frame) error {
	shouldMute := false
	if u.InterruptionsAllowed() {
		for _, strategy := range u.turnStrategies.MuteStrategies {
			if strategy.ShouldMute(frame) {
				shouldMute = true
				break
			}
		}
	}

	u.stateMu.Lock()
	mutedChanged := shouldMute != u.mutedState
	u.mutedState = shouldMute
	u.stateMu.Unlock()

	if !mutedChanged {
		return nil
	}

	if shouldMute {
		return u.PushFrame(frames.NewUserMuteStartedFrame(), frames.Downstream)
	}

	return u.PushFrame(frames.NewUserMuteStoppedFrame(), frames.Downstream)
}

func (u *LLMUserAggregator) shouldSuppressFrame(frame frames.Frame) bool {
	u.stateMu.Lock()
	muted := u.mutedState
	u.stateMu.Unlock()

	if !muted {
		return false
	}

	switch frame.(type) {
	case *frames.AudioFrame, *frames.TranscriptionFrame, *frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame:
		return true
	default:
		return false
	}
}

func hasTurnStrategies(strategies turns.UserTurnStrategies) bool {
	return len(strategies.StartStrategies) > 0 || len(strategies.StopStrategies) > 0 || len(strategies.MuteStrategies) > 0
}
