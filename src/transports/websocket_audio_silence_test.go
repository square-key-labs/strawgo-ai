package transports

import (
	"testing"
)

func TestAudioOutAutoSilenceDefault(t *testing.T) {
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: &mockSerializer{},
	})
	if !transport.outputProc.audioOutAutoSilence {
		t.Fatalf("expected default %v, got false", AudioOutAutoSilenceDefault)
	}
}

func TestAudioOutAutoSilenceExplicitTrue(t *testing.T) {
	on := true
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:                8081,
		Path:                "/ws",
		Serializer:          &mockSerializer{},
		AudioOutAutoSilence: &on,
	})
	if !transport.outputProc.audioOutAutoSilence {
		t.Fatal("expected true when explicitly set true")
	}
}

func TestAudioOutAutoSilenceExplicitFalse(t *testing.T) {
	off := false
	transport := NewWebSocketTransport(WebSocketConfig{
		Port:                8082,
		Path:                "/ws",
		Serializer:          &mockSerializer{},
		AudioOutAutoSilence: &off,
	})
	if transport.outputProc.audioOutAutoSilence {
		t.Fatal("expected false when explicitly set false")
	}
}
