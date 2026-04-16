package user_mute

import (
	"sync"
)

type FunctionCallUserMuteStrategy struct {
	enableInterruptions bool

	mu                 sync.Mutex
	functionCallActive bool
}

func NewFunctionCallUserMuteStrategy(enableInterruptions bool) *FunctionCallUserMuteStrategy {
	return &FunctionCallUserMuteStrategy{enableInterruptions: enableInterruptions}
}

func (s *FunctionCallUserMuteStrategy) ShouldMute(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch frameName(frame) {
	case "FunctionCallsStartedFrame":
		s.functionCallActive = true
	case "FunctionCallResultFrame", "FunctionCallCancelFrame":
		s.functionCallActive = false
	}

	return s.functionCallActive
}

func (s *FunctionCallUserMuteStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *FunctionCallUserMuteStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.functionCallActive = false
}
