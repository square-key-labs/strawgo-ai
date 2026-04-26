package deepgram

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// stt_update_settings_test exercises the runtime UpdateSettings path in
// isolation — no live websocket, just state mutation.

func TestUpdateSettingsAppliesKnownKeys(t *testing.T) {
	s := NewSTTService(STTConfig{
		APIKey:   "key",
		Language: "en",
		Model:    "nova-2",
		Encoding: "linear16",
	})
	if err := s.UpdateSettings(map[string]interface{}{
		"language": "es",
		"model":    "nova-3",
		"encoding": "mulaw",
	}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if s.language != "es" || s.model != "nova-3" || s.encoding != "mulaw" {
		t.Fatalf("expected es/nova-3/mulaw, got %s/%s/%s", s.language, s.model, s.encoding)
	}
}

func TestUpdateSettingsIgnoresEmptyAndUnknown(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "k", Language: "en", Model: "nova-2"})
	if err := s.UpdateSettings(map[string]interface{}{
		"language": "",            // ignored
		"unknown":  "foo",         // ignored
		"model":    map[int]int{}, // wrong type → ignored
	}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if s.language != "en" || s.model != "nova-2" {
		t.Fatalf("expected unchanged en/nova-2, got %s/%s", s.language, s.model)
	}
}

func TestSTTUpdateSettingsFrameTargetingSelf(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "k", Language: "en"})
	frame := frames.NewSTTUpdateSettingsFrameForService(map[string]interface{}{"language": "fr"}, "DeepgramSTT")

	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.language != "fr" {
		t.Fatalf("expected language=fr, got %s", s.language)
	}
}

func TestSTTUpdateSettingsFrameTargetingOther(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "k", Language: "en"})
	frame := frames.NewSTTUpdateSettingsFrameForService(map[string]interface{}{"language": "fr"}, "AzureSTT")

	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.language != "en" {
		t.Fatalf("expected language unchanged (en), got %s", s.language)
	}
}

func TestSTTUpdateSettingsFrameUntargeted(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "k", Language: "en"})
	frame := frames.NewSTTUpdateSettingsFrame(map[string]interface{}{"language": "fr"})

	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.language != "fr" {
		t.Fatalf("expected language=fr (untargeted update), got %s", s.language)
	}
}
