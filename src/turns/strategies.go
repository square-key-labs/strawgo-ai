package turns

import (
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type UserTurnStrategies struct {
	StartStrategies []user_start.UserTurnStartStrategy
	StopStrategies  []user_stop.UserTurnStopStrategy
	MuteStrategies  []user_mute.UserMuteStrategy
}

func (s UserTurnStrategies) Empty() bool {
	return len(s.StartStrategies) == 0 && len(s.StopStrategies) == 0 && len(s.MuteStrategies) == 0
}

func DefaultUserTurnStrategies() UserTurnStrategies {
	return UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(true),
			user_start.NewTranscriptionUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(user_stop.DefaultUserSpeechTimeout),
		},
	}
}

func BackchannelAwareUserTurnStrategies() UserTurnStrategies {
	return UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewBackchannelUserTurnStartStrategy(user_start.BackchannelParams{}, true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(user_stop.DefaultUserSpeechTimeout),
		},
	}
}
