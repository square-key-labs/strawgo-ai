package user_stop

import (
	"strings"
	"sync"
	"testing"
	"time"
)

// captureWarn returns a warn-function and a thread-safe accessor for the
// captured messages. Designed for the strategy.SetWarnFunc test hook.
func captureWarn(t *testing.T) (func(string, ...interface{}), func() []string) {
	t.Helper()
	var (
		mu       sync.Mutex
		captured []string
	)
	fn := func(format string, args ...interface{}) {
		mu.Lock()
		defer mu.Unlock()
		// We don't need a real format; capturing the format string is
		// enough to assert on the warning text.
		captured = append(captured, format)
	}
	return fn, func() []string {
		mu.Lock()
		defer mu.Unlock()
		out := make([]string, len(captured))
		copy(out, captured)
		return out
	}
}

func TestVADStopSecsExceedsP99Warns(t *testing.T) {
	warn, get := captureWarn(t)
	s := NewSpeechTimeoutUserTurnStopStrategy(0, true)
	s.SetWarnFunc(warn)

	s.SetVADStopSecs(500 * time.Millisecond)
	s.SetTTFSP99Latency(300 * time.Millisecond) // p99 < vad → bad

	msgs := get()
	if len(msgs) == 0 || !strings.Contains(msgs[0], "VAD stop_secs") {
		t.Fatalf("expected warning, got %v", msgs)
	}
}

func TestVADStopSecsBelowP99NoWarning(t *testing.T) {
	warn, get := captureWarn(t)
	s := NewSpeechTimeoutUserTurnStopStrategy(0, true)
	s.SetWarnFunc(warn)

	s.SetVADStopSecs(150 * time.Millisecond)
	s.SetTTFSP99Latency(300 * time.Millisecond) // p99 > vad → ok

	if got := get(); len(got) != 0 {
		t.Fatalf("did not expect warnings, got %v", got)
	}
}

func TestVADStopSecsWarningFiresOnce(t *testing.T) {
	warn, get := captureWarn(t)
	s := NewSpeechTimeoutUserTurnStopStrategy(0, true)
	s.SetWarnFunc(warn)

	s.SetVADStopSecs(500 * time.Millisecond)
	s.SetTTFSP99Latency(300 * time.Millisecond)
	s.SetTTFSP99Latency(300 * time.Millisecond) // same values again

	if got := len(get()); got != 1 {
		t.Fatalf("expected exactly one warning, got %d", got)
	}
}

func TestVADStopSecsChangeResetsWarn(t *testing.T) {
	warn, get := captureWarn(t)
	s := NewSpeechTimeoutUserTurnStopStrategy(0, true)
	s.SetWarnFunc(warn)

	s.SetVADStopSecs(500 * time.Millisecond)
	s.SetTTFSP99Latency(300 * time.Millisecond) // fires
	s.SetVADStopSecs(800 * time.Millisecond)    // changes; eligible again

	if got := len(get()); got != 2 {
		t.Fatalf("expected two warnings, got %d", got)
	}
}
