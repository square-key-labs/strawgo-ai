package user_stop

import (
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

type TurnAnalyzerUserTurnStopStrategy struct {
	analyzer            turn.TurnAnalyzer
	enableInterruptions bool
	mu                  sync.Mutex
	metrics             turn.TurnMetrics
	hasMetrics          bool
}

func NewTurnAnalyzerUserTurnStopStrategy(analyzer turn.TurnAnalyzer, enableInterruptions bool) *TurnAnalyzerUserTurnStopStrategy {
	return &TurnAnalyzerUserTurnStopStrategy{
		analyzer:            analyzer,
		enableInterruptions: enableInterruptions,
	}
}

func (s *TurnAnalyzerUserTurnStopStrategy) ShouldStop(frame any) bool {
	if s.analyzer == nil || frameinfo.Name(frame) != "TranscriptionFrame" || !frameinfo.IsFinal(frame) {
		return false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	state, metrics, err := s.analyzer.AnalyzeEndOfTurn()
	if err != nil {
		return false
	}
	if metrics != nil {
		s.metrics, s.hasMetrics = *metrics, true
	}
	return state == turn.TurnComplete
}

func (s *TurnAnalyzerUserTurnStopStrategy) TakeTurnMetrics() (turn.TurnMetrics, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	metrics, ok := s.metrics, s.hasMetrics
	s.metrics, s.hasMetrics = turn.TurnMetrics{}, false
	return metrics, ok
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
