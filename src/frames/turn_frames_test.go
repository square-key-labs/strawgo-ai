package frames

import "testing"

func TestUserMuteFramesMetadata(t *testing.T) {
	started := NewUserMuteStartedFrame()
	if started.Name() != "UserMuteStartedFrame" {
		t.Fatalf("expected UserMuteStartedFrame name, got %q", started.Name())
	}
	if started.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", started.Category())
	}

	stopped := NewUserMuteStoppedFrame()
	if stopped.Name() != "UserMuteStoppedFrame" {
		t.Fatalf("expected UserMuteStoppedFrame name, got %q", stopped.Name())
	}
	if stopped.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", stopped.Category())
	}
}

func TestTTSContextIDFields(t *testing.T) {
	started := NewTTSStartedFrameWithContext("ctx-start")
	if started.Name() != "TTSStartedFrame" {
		t.Fatalf("expected TTSStartedFrame name, got %q", started.Name())
	}
	if started.Category() != ControlCategory {
		t.Fatalf("expected ControlCategory, got %v", started.Category())
	}
	if started.ContextID != "ctx-start" {
		t.Fatalf("expected started context ID to be ctx-start, got %q", started.ContextID)
	}

	stopped := NewTTSStoppedFrame()
	stopped.ContextID = "ctx-stop"
	if stopped.Name() != "TTSStoppedFrame" {
		t.Fatalf("expected TTSStoppedFrame name, got %q", stopped.Name())
	}
	if stopped.Category() != ControlCategory {
		t.Fatalf("expected ControlCategory, got %v", stopped.Category())
	}
	if stopped.ContextID != "ctx-stop" {
		t.Fatalf("expected stopped context ID to be ctx-stop, got %q", stopped.ContextID)
	}

	audio := NewTTSAudioFrame([]byte{1, 2, 3}, 24000, 1)
	audio.ContextID = "ctx-audio"
	if audio.Name() != "TTSAudioFrame" {
		t.Fatalf("expected TTSAudioFrame name, got %q", audio.Name())
	}
	if audio.Category() != DataCategory {
		t.Fatalf("expected DataCategory, got %v", audio.Category())
	}
	if audio.ContextID != "ctx-audio" {
		t.Fatalf("expected audio context ID to be ctx-audio, got %q", audio.ContextID)
	}
}
