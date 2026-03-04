package user_mute

// UserMuteStrategy decides when user audio should be muted.
type UserMuteStrategy interface {
	// ShouldMute returns true when the provided frame indicates muting is needed.
	ShouldMute(frame any) bool

	// Reset clears any internal strategy state.
	Reset()
}
