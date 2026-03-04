package user_start

// UserTurnStartStrategy decides when a user turn should begin.
type UserTurnStartStrategy interface {
	// ShouldStart returns true when the provided frame indicates a user turn start.
	ShouldStart(frame any) bool

	// EnableInterruptions reports whether interruptions should be enabled.
	EnableInterruptions() bool

	// Reset clears any internal strategy state.
	Reset()
}
