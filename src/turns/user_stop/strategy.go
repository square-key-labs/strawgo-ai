package user_stop

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
)

type UserTurnStopStrategy interface {
	ShouldStop(frame any) bool
	Reset()
}

type Scheduler interface {
	NextDeadline() (time.Time, bool)
}

type TurnLifecycle interface {
	HandleUserTurnStarted()
	HandleUserTurnStopped()
}

type TurnMetricsReporter interface {
	TakeTurnMetrics() (turn.TurnMetrics, bool)
}
