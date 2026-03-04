package user_stop

// UserTurnStopStrategy decides when a user turn should end.
type UserTurnStopStrategy interface {
	// ShouldStop returns true when the provided frame indicates a user turn end.
	ShouldStop(frame any) bool

	// Reset clears any internal strategy state.
	Reset()
}
