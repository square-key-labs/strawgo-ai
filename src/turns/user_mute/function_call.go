package user_mute

import (
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

type FunctionCallUserMuteStrategy struct {
	mu         sync.Mutex
	inProgress map[string]struct{}
}

func NewFunctionCallUserMuteStrategy() *FunctionCallUserMuteStrategy {
	return &FunctionCallUserMuteStrategy{inProgress: make(map[string]struct{})}
}

func (s *FunctionCallUserMuteStrategy) ShouldMute(frame any) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch frameinfo.Name(frame) {
	case "FunctionCallsStartedFrame":
		for _, id := range frameinfo.ToolCallIDs(frame) {
			s.inProgress[id] = struct{}{}
		}
	case "FunctionCallResultFrame", "FunctionCallCancelFrame":
		if id, ok := frameinfo.StringField(frame, "ToolCallID"); ok {
			delete(s.inProgress, id)
		}
	}
	return len(s.inProgress) > 0
}

func (s *FunctionCallUserMuteStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	clear(s.inProgress)
}
