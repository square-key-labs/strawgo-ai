package deepgram

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// TestConcurrentUpdateSettingsAndSetters stress-tests the lifecycleMu
// paths: parallel UpdateSettings, SetLanguage, SetModel, Cleanup, and
// STTUpdateSettingsFrame handling. No AudioFrame is sent so Initialize
// is never triggered (avoids dialing real Deepgram). The point is the
// race detector: any unsynchronized write/read on language, model,
// encoding, ctx, or cancel surfaces here.
func TestConcurrentUpdateSettingsAndSetters(t *testing.T) {
	s := NewSTTService(STTConfig{
		APIKey:   "k",
		Language: "en",
		Model:    "nova-2",
		Encoding: "linear16",
	})

	stop := make(chan struct{})
	deadline := time.After(150 * time.Millisecond)
	go func() {
		<-deadline
		close(stop)
	}()

	var wg sync.WaitGroup
	worker := func(fn func()) {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				fn()
			}
		}
	}

	wg.Add(6)
	go worker(func() {
		_ = s.UpdateSettings(map[string]interface{}{"language": "es", "model": "nova-3", "encoding": "mulaw"})
	})
	go worker(func() {
		_ = s.UpdateSettings(map[string]interface{}{"language": "en", "model": "nova-2", "encoding": "linear16"})
	})
	go worker(func() {
		s.SetLanguage("fr")
	})
	go worker(func() {
		s.SetModel("nova-3-general")
	})
	go worker(func() {
		_ = s.Cleanup()
	})
	go worker(func() {
		f := frames.NewSTTUpdateSettingsFrame(map[string]interface{}{"language": "de"})
		_ = s.HandleFrame(context.TODO(), f, frames.Downstream)
	})

	wg.Wait()
}
