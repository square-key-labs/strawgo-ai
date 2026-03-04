package user_stop_test

import (
	"errors"
	"sync"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type mockTurnAnalyzer struct {
	mu          sync.Mutex
	state       turn.EndOfTurnState
	err         error
	clearCalled int
	analyzeCall int
}

func (m *mockTurnAnalyzer) AppendAudio(_ []byte, _ bool) turn.EndOfTurnState {
	return turn.TurnIncomplete
}

func (m *mockTurnAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, *turn.TurnMetrics, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.analyzeCall++
	return m.state, nil, m.err
}

func (m *mockTurnAnalyzer) SpeechTriggered() bool { return false }

func (m *mockTurnAnalyzer) SetSampleRate(_ int) {}

func (m *mockTurnAnalyzer) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.clearCalled++
}

func TestTurnAnalyzerStopStrategy(t *testing.T) {
	mock := &mockTurnAnalyzer{state: turn.TurnComplete}
	strategy := user_stop.NewTurnAnalyzerUserTurnStopStrategy(mock, true)

	if !strategy.ShouldStop(frames.NewTranscriptionFrame("hello", true)) {
		t.Fatalf("expected turn complete from analyzer to stop")
	}

	if strategy.ShouldStop(frames.NewTextFrame("ignore")) {
		t.Fatalf("expected non-transcription frame to not trigger analyzer")
	}

	if !strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be true")
	}
}

func TestTurnAnalyzerStopStrategyHandlesErrorAndReset(t *testing.T) {
	mock := &mockTurnAnalyzer{state: turn.TurnComplete, err: errors.New("boom")}
	strategy := user_stop.NewTurnAnalyzerUserTurnStopStrategy(mock, false)

	if strategy.ShouldStop(frames.NewTranscriptionFrame("hello", true)) {
		t.Fatalf("expected analyzer error to not stop")
	}

	strategy.Reset()

	mock.mu.Lock()
	clearCount := mock.clearCalled
	analyzeCount := mock.analyzeCall
	mock.mu.Unlock()

	if clearCount != 1 {
		t.Fatalf("expected reset to call clear once, got %d", clearCount)
	}

	if analyzeCount != 1 {
		t.Fatalf("expected one analyze call, got %d", analyzeCount)
	}

	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
}
