package turns

const DefaultEmptyUserTurnInterruptedPrompt = "The user may have said something while you were responding, but it was not " +
	"recognized. Your response was cut off there, and the conversation only includes " +
	"the part they heard, which may be none of it. Briefly ask them to repeat what they said, and " +
	"repeat any question of yours they may have missed."

type EmptyUserTurnConfig struct {
	Disabled                   bool
	InterruptedPrompt          string
	LeaveInterruptedUnanswered bool
	IdlePrompt                 string
	MaxConsecutiveRecoveries   int
	ResumeInterruptedSpeech    bool
}

func DefaultEmptyUserTurnConfig() EmptyUserTurnConfig {
	return EmptyUserTurnConfig{
		InterruptedPrompt:        DefaultEmptyUserTurnInterruptedPrompt,
		MaxConsecutiveRecoveries: 1,
	}
}

func (c EmptyUserTurnConfig) Prompt(interruptedBot bool) string {
	if c.Disabled {
		return ""
	}
	if !interruptedBot {
		return c.IdlePrompt
	}
	if c.LeaveInterruptedUnanswered {
		return ""
	}
	if c.InterruptedPrompt == "" {
		return DefaultEmptyUserTurnInterruptedPrompt
	}
	return c.InterruptedPrompt
}

func (c EmptyUserTurnConfig) MaxRecoveries() int {
	if c.MaxConsecutiveRecoveries <= 0 {
		return 1
	}
	return c.MaxConsecutiveRecoveries
}
