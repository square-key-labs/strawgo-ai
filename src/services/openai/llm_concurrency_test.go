package openai

import (
	"sync"
	"testing"
	"time"
)

// TestConcurrentUpdateSettingsWithReaders exercises settingsMu under
// parallel writers (UpdateSettings, SetModel, SetTemperature) and
// readers (the snapshot taken at the top of generateResponseFromContext).
// We don't drive a real LLM call — that requires the network. Instead
// the test reader takes the same RLock + reads the same fields the
// generation path does. Race detector enforces correctness.
func TestConcurrentUpdateSettingsWithReaders(t *testing.T) {
	s := NewLLMService(LLMConfig{
		APIKey:      "k",
		Model:       "gpt-4o",
		Temperature: 0.7,
	})

	stop := make(chan struct{})
	deadline := time.After(150 * time.Millisecond)
	go func() { <-deadline; close(stop) }()

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

	wg.Add(5)
	go worker(func() {
		_ = s.UpdateSettings(map[string]interface{}{
			"model":              "gpt-4o-mini",
			"temperature":        0.2,
			"system_instruction": "Be concise.",
		})
	})
	go worker(func() {
		_ = s.UpdateSettings(map[string]interface{}{
			"model":              "gpt-4o",
			"temperature":        0.7,
			"system_instruction": "",
		})
	})
	go worker(func() {
		s.SetModel("gpt-4o-mini-2024-07-18")
	})
	go worker(func() {
		s.SetTemperature(0.5)
	})
	go worker(func() {
		// Snapshot path mirroring generateResponseFromContext.
		s.settingsMu.RLock()
		_ = s.model
		_ = s.temperature
		_ = s.systemInstruction
		s.settingsMu.RUnlock()
	})

	wg.Wait()
}
