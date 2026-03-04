package deepgram

import (
	"context"
	"testing"
	"time"
)

func TestNewDeepgramSTTService(t *testing.T) {
	config := STTConfig{
		APIKey:   "test-api-key",
		Language: "en-US",
		Model:    "nova-2",
		Encoding: "linear16",
	}

	service := NewSTTService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.apiKey != "test-api-key" {
		t.Errorf("Expected API key 'test-api-key', got %s", service.apiKey)
	}

	if service.language != "en-US" {
		t.Errorf("Expected language 'en-US', got %s", service.language)
	}

	if service.model != "nova-2" {
		t.Errorf("Expected model 'nova-2', got %s", service.model)
	}

	if service.encoding != "linear16" {
		t.Errorf("Expected encoding 'linear16', got %s", service.encoding)
	}
}

func TestDeepgramSTT_Keepalive(t *testing.T) {
	config := STTConfig{
		APIKey:            "test-api-key",
		KeepaliveInterval: 2 * time.Second,
		KeepaliveTimeout:  10 * time.Second,
	}

	service := NewSTTService(config)

	if service.keepaliveInterval != 2*time.Second {
		t.Errorf("Expected keepaliveInterval 2s, got %v", service.keepaliveInterval)
	}

	if service.keepaliveTimeout != 10*time.Second {
		t.Errorf("Expected keepaliveTimeout 10s, got %v", service.keepaliveTimeout)
	}
}

func TestDeepgramSTT_KeepaliveDefaults(t *testing.T) {
	config := STTConfig{
		APIKey: "test-api-key",
	}

	service := NewSTTService(config)

	if service.keepaliveInterval != 5*time.Second {
		t.Errorf("Expected default keepaliveInterval 5s, got %v", service.keepaliveInterval)
	}

	if service.keepaliveTimeout != 30*time.Second {
		t.Errorf("Expected default keepaliveTimeout 30s, got %v", service.keepaliveTimeout)
	}
}

func TestDeepgramSTT_SetLanguage(t *testing.T) {
	config := STTConfig{
		APIKey:   "test-api-key",
		Language: "en-US",
	}

	service := NewSTTService(config)

	if service.language != "en-US" {
		t.Errorf("Expected initial language 'en-US', got %s", service.language)
	}

	service.SetLanguage("es-ES")

	if service.language != "es-ES" {
		t.Errorf("Expected language to be updated to 'es-ES', got %s", service.language)
	}
}

func TestDeepgramSTT_SetModel(t *testing.T) {
	config := STTConfig{
		APIKey: "test-api-key",
		Model:  "nova-2",
	}

	service := NewSTTService(config)

	if service.model != "nova-2" {
		t.Errorf("Expected initial model 'nova-2', got %s", service.model)
	}

	service.SetModel("nova-3")

	if service.model != "nova-3" {
		t.Errorf("Expected model to be updated to 'nova-3', got %s", service.model)
	}
}

func TestDeepgramSTT_EncodingNormalization(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"mulaw", "mulaw", "mulaw"},
		{"ulaw", "ulaw", "mulaw"},
		{"PCMU", "PCMU", "mulaw"},
		{"alaw", "alaw", "alaw"},
		{"PCMA", "PCMA", "alaw"},
		{"pcm", "pcm", "linear16"},
		{"PCM", "PCM", "linear16"},
		{"linear16", "linear16", "linear16"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := STTConfig{
				APIKey:   "test-api-key",
				Encoding: tt.input,
			}

			service := NewSTTService(config)

			if service.encoding != tt.expected {
				t.Errorf("Expected encoding %s, got %s", tt.expected, service.encoding)
			}
		})
	}
}

// TestDeepgramSTT_ErrorPropagation verifies that API errors are propagated
func TestDeepgramSTT_ErrorPropagation(t *testing.T) {
	config := STTConfig{
		APIKey: "invalid-key",
	}

	service := NewSTTService(config)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Try to initialize with invalid credentials
	// Should return an error (WebSocket dial will fail)
	err := service.Initialize(ctx)

	if err == nil {
		t.Error("Expected Initialize to return an error for invalid API key")
	}
}
