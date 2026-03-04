package user_stop

import (
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
)

type TurnAnalyzerUserTurnStopStrategy struct {
	analyzer            turn.TurnAnalyzer
	enableInterruptions bool
	mu                  sync.Mutex
}

func NewTurnAnalyzerUserTurnStopStrategy(analyzer turn.TurnAnalyzer, enableInterruptions bool) *TurnAnalyzerUserTurnStopStrategy {
	return &TurnAnalyzerUserTurnStopStrategy{
		analyzer:            analyzer,
		enableInterruptions: enableInterruptions,
	}
}

func (s *TurnAnalyzerUserTurnStopStrategy) ShouldStop(frame any) bool {
	if s.analyzer == nil {
		return false
	}

	named, ok := frame.(namedFrame)
	if !ok || named.Name() != "TranscriptionFrame" {
		return false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	state, _, err := s.analyzer.AnalyzeEndOfTurn()
	if err != nil {
		return false
	}

	return state == turn.TurnComplete
}

func (s *TurnAnalyzerUserTurnStopStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *TurnAnalyzerUserTurnStopStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.analyzer != nil {
		s.analyzer.Clear()
	}
}
