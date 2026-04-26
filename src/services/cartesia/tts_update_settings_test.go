package cartesia

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestUpdateSettingsAppliesKnownKeys(t *testing.T) {
	s := NewTTSService(TTSConfig{
		APIKey:  "k",
		VoiceID: "v1",
	})
	if err := s.UpdateSettings(map[string]interface{}{
		"voice":    "v2",
		"model":    "sonic-2",
		"language": "es",
	}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if s.voiceID != "v2" {
		t.Fatalf("expected voice=v2, got %s", s.voiceID)
	}
	if s.model != "sonic-2" {
		t.Fatalf("expected model=sonic-2, got %s", s.model)
	}
	if s.language != "es" {
		t.Fatalf("expected language=es, got %s", s.language)
	}
}

func TestTTSUpdateSettingsFrameTargetingSelf(t *testing.T) {
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "v1"})
	frame := frames.NewTTSUpdateSettingsFrameForService(
		map[string]interface{}{"voice": "v2"}, s.Name(),
	)
	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.voiceID != "v2" {
		t.Fatalf("expected voice=v2, got %s", s.voiceID)
	}
}

func TestTTSUpdateSettingsFrameTargetingOther(t *testing.T) {
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "v1"})
	frame := frames.NewTTSUpdateSettingsFrameForService(
		map[string]interface{}{"voice": "v2"}, "ElevenLabs",
	)
	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.voiceID != "v1" {
		t.Fatalf("expected voice unchanged (v1), got %s", s.voiceID)
	}
}

func TestTTSUpdateSettingsFrameUntargeted(t *testing.T) {
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "v1"})
	frame := frames.NewTTSUpdateSettingsFrame(map[string]interface{}{"voice": "v2"})
	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.voiceID != "v2" {
		t.Fatalf("expected voice=v2 (untargeted), got %s", s.voiceID)
	}
}
