package user_mute

type AlwaysUserMuteStrategy struct {
	enableInterruptions bool
}

func NewAlwaysUserMuteStrategy(enableInterruptions bool) *AlwaysUserMuteStrategy {
	return &AlwaysUserMuteStrategy{enableInterruptions: enableInterruptions}
}

func (s *AlwaysUserMuteStrategy) ShouldMute(_ any) bool {
	return true
}

func (s *AlwaysUserMuteStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *AlwaysUserMuteStrategy) Reset() {}
