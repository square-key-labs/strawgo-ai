package frames

import (
	"errors"
	"testing"
)

func TestLLMUpdateSettingsFrameIsSystemCategory(t *testing.T) {
	f := NewLLMUpdateSettingsFrame(map[string]interface{}{"model": "gpt-4o-mini"})
	if f.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", f.Category())
	}
	if f.Service != "" {
		t.Fatalf("expected empty Service for untargeted frame, got %q", f.Service)
	}
}

func TestLLMUpdateSettingsFrameForServiceCarriesTarget(t *testing.T) {
	f := NewLLMUpdateSettingsFrameForService(
		map[string]interface{}{"temperature": 0.3},
		"OpenAI",
	)
	if f.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", f.Category())
	}
	if f.Service != "OpenAI" {
		t.Fatalf("expected Service=OpenAI, got %q", f.Service)
	}
}

func TestTTSUpdateSettingsFrameIsSystemCategory(t *testing.T) {
	f := NewTTSUpdateSettingsFrame(map[string]interface{}{"voice": "alloy"})
	if f.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", f.Category())
	}
	if f.Service != "" {
		t.Fatalf("expected empty Service for untargeted frame, got %q", f.Service)
	}
}

func TestTTSUpdateSettingsFrameForServiceCarriesTarget(t *testing.T) {
	f := NewTTSUpdateSettingsFrameForService(
		map[string]interface{}{"speed": 1.1},
		"Cartesia",
	)
	if f.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", f.Category())
	}
	if f.Service != "Cartesia" {
		t.Fatalf("expected Service=Cartesia, got %q", f.Service)
	}
}

func TestErrorFrameIsFatalDefault(t *testing.T) {
	f := NewErrorFrame(errors.New("boom"))
	if f.IsFatal() {
		t.Fatal("expected IsFatal false when metadata absent")
	}
}

func TestErrorFrameIsFatalSet(t *testing.T) {
	f := NewErrorFrame(errors.New("boom"))
	f.SetMetadata(MetadataKeyFatal, true)
	if !f.IsFatal() {
		t.Fatal("expected IsFatal true after setting metadata")
	}
}

func TestErrorFrameIsFatalNonBoolMetadata(t *testing.T) {
	f := NewErrorFrame(errors.New("boom"))
	f.SetMetadata(MetadataKeyFatal, "yes") // wrong type
	if f.IsFatal() {
		t.Fatal("expected IsFatal false when metadata is not bool")
	}
}
