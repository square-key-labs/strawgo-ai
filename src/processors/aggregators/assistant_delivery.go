package aggregators

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

const selfDeliveryWake selfFrameKind = 100

type deliveryState struct {
	mode         llm.ResultDelivery
	interrupting bool
	userMuted    bool
	userTurn     bool
	stoppedAt    time.Time
	gen          uint64
	timer        *time.Timer
}

func deliveryRank(mode llm.ResultDelivery) int {
	switch mode {
	case llm.ResultDeliveryInterrupt:
		return 2
	case llm.ResultDeliveryWhenIdle:
		return 0
	}
	return 1
}

func (a *LLMAssistantAggregator) oweRun(mode llm.ResultDelivery) {
	if !a.pushOwed || deliveryRank(mode) > deliveryRank(a.dl.mode) {
		a.dl.mode = mode
	}
	a.pushOwed = true
}

func (a *LLMAssistantAggregator) clearOwedDelivery() {
	a.dl.mode = llm.ResultDeliveryDefault
}

func (a *LLMAssistantAggregator) deliverOwed() error {
	switch a.dl.mode {
	case llm.ResultDeliveryInterrupt:
		if a.userSpeaking {
			return nil
		}
		if (a.botSpeaking || a.depth > 0) && a.InterruptionsAllowed() && !a.dl.userMuted {
			if a.dl.interrupting {
				return nil
			}
			a.dl.interrupting = true
			a.pushOnBotStopped = true
			return a.PushInterruptionTaskFrame()
		}
	case llm.ResultDeliveryWhenIdle:
		if a.userSpeaking {
			return nil
		}
		if a.dl.userTurn {
			a.armUserTurnTimer()
			return nil
		}
	}
	return a.maybePushContext()
}

func (a *LLMAssistantAggregator) takeInterruptDelivery() bool {
	owed := a.dl.interrupting && a.pushOwed
	a.dl.interrupting = false
	return owed
}

func (a *LLMAssistantAggregator) botStoppedSpeaking() {
	a.dl.interrupting = false
}

func (a *LLMAssistantAggregator) deliverAfterInterruption(owed bool) error {
	if !owed {
		return nil
	}
	a.pushOwed = true
	if a.userSpeaking {
		return nil
	}
	return a.pushContextUpstream()
}

func (a *LLMAssistantAggregator) waitingForUserTurn() bool {
	return a.pushOwed && a.dl.mode == llm.ResultDeliveryWhenIdle && a.dl.userTurn
}

func (a *LLMAssistantAggregator) userTurnStarted() {
	a.dl.userTurn = true
	a.stopUserTurnTimer()
}

func (a *LLMAssistantAggregator) userSpeechStopped() {
	a.dl.stoppedAt = time.Now()
}

func (a *LLMAssistantAggregator) armUserTurnTimer() {
	if a.dl.timer != nil {
		return
	}
	timeout := a.params.WhenIdleUserTurnTimeout
	if timeout <= 0 {
		timeout = DefaultUserTurnStopTimeout
	}
	gen := a.dl.gen
	a.dl.timer = time.AfterFunc(max(time.Until(a.dl.stoppedAt.Add(timeout)), 0), func() {
		_ = a.QueueFrame(newSelfFrame(selfDeliveryWake, gen, nil), frames.Downstream)
	})
}

func (a *LLMAssistantAggregator) userTurnAnswered() {
	a.dl.userTurn = false
	a.stopUserTurnTimer()
}

func (a *LLMAssistantAggregator) stopUserTurnTimer() {
	a.dl.gen++
	if a.dl.timer != nil {
		a.dl.timer.Stop()
		a.dl.timer = nil
	}
}

func (a *LLMAssistantAggregator) handleDeliveryWake(gen uint64) error {
	if gen != a.dl.gen || !a.dl.userTurn {
		return nil
	}
	a.dl.userTurn = false
	a.dl.timer = nil
	if !a.pushOwed || a.userSpeaking {
		return nil
	}
	return a.deliverOwed()
}
