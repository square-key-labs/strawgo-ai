package frames

import "testing"

func TestSTTUpdateSettingsFrameIsSystemCategory(t *testing.T) {
	f := NewSTTUpdateSettingsFrame(map[string]interface{}{"language": "es"})
	if f.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", f.Category())
	}
	if f.Service != "" {
		t.Fatalf("expected empty Service for untargeted frame, got %q", f.Service)
	}
}

func TestSTTUpdateSettingsFrameForServiceCarriesTarget(t *testing.T) {
	f := NewSTTUpdateSettingsFrameForService(
		map[string]interface{}{"model": "nova-3"},
		"DeepgramSTT",
	)
	if f.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", f.Category())
	}
	if f.Service != "DeepgramSTT" {
		t.Fatalf("expected Service=DeepgramSTT, got %q", f.Service)
	}
	if got := f.Settings["model"]; got != "nova-3" {
		t.Fatalf("expected model=nova-3, got %v", got)
	}
}
