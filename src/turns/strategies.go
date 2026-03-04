package turns

import (
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

// UserTurnStrategies configures how user turns are detected and managed.
type UserTurnStrategies struct {
	// StartStrategies evaluate when a user turn should begin.
	StartStrategies []user_start.UserTurnStartStrategy

	// StopStrategies evaluate when a user turn should end.
	StopStrategies []user_stop.UserTurnStopStrategy

	// MuteStrategies evaluate when the user should be muted.
	MuteStrategies []user_mute.UserMuteStrategy
}
