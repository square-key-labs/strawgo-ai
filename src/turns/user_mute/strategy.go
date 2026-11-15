package user_mute

type UserMuteStrategy interface {
	ShouldMute(frame any) bool
	Reset()
}
