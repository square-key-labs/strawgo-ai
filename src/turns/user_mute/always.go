package user_mute

type AlwaysUserMuteStrategy struct{}

func NewAlwaysUserMuteStrategy() *AlwaysUserMuteStrategy {
	return &AlwaysUserMuteStrategy{}
}

func (s *AlwaysUserMuteStrategy) ShouldMute(_ any) bool {
	return true
}

func (s *AlwaysUserMuteStrategy) Reset() {}
