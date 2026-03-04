package azure

import (
	"context"
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestNewTTSService(t *testing.T) {
	config := TTSConfig{
		SubscriptionKey: "test-subscription-key",
		Region:          "westus",
		Voice:           "en-US-AriaNeural",
		OutputFormat:    "riff-24khz-16bit-mono-pcm",
	}

	service := NewTTSService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.subscriptionKey != "test-subscription-key" {
		t.Errorf("Expected subscription key 'test-subscription-key', got %s", service.subscriptionKey)
	}

	if service.region != "westus" {
		t.Errorf("Expected region 'westus', got %s", service.region)
	}

	if service.voice != "en-US-AriaNeural" {
		t.Errorf("Expected voice 'en-US-AriaNeural', got %s", service.voice)
	}

	if service.outputFormat != "riff-24khz-16bit-mono-pcm" {
		t.Errorf("Expected output format 'riff-24khz-16bit-mono-pcm', got %s", service.outputFormat)
	}
}

func TestNewTTSServiceDefaults(t *testing.T) {
	config := TTSConfig{
		SubscriptionKey: "test-key",
	}

	service := NewTTSService(config)

	if service.region != DefaultRegion {
		t.Errorf("Expected default region %s, got %s", DefaultRegion, service.region)
	}

	if service.voice != DefaultVoice {
		t.Errorf("Expected default voice %s, got %s", DefaultVoice, service.voice)
	}

	if service.outputFormat != DefaultOutputFormat {
		t.Errorf("Expected default output format %s, got %s", DefaultOutputFormat, service.outputFormat)
	}
}

func TestTTSContextIDGeneration(t *testing.T) {
	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()

	if contextID1 == "" || contextID2 == "" {
		t.Error("GenerateContextID returned empty string")
	}

	if contextID1 == contextID2 {
		t.Errorf("GenerateContextID should generate unique IDs, got: %s == %s", contextID1, contextID2)
	}

	if len(contextID1) != 36 {
		t.Errorf("Expected UUID length 36, got: %d", len(contextID1))
	}
	if contextID1[8] != '-' || contextID1[13] != '-' || contextID1[18] != '-' || contextID1[23] != '-' {
		t.Errorf("Invalid UUID format: %s", contextID1)
	}
}

func TestTTSContextIDConsistency(t *testing.T) {
	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()
	contextID3 := services.GenerateContextID()

	contextIDs := []string{contextID1, contextID2, contextID3}

	for i, id := range contextIDs {
		if id == "" {
			t.Errorf("Context ID %d is empty", i)
		}

		if len(id) != 36 {
			t.Errorf("Context ID %d has wrong length: expected 36, got %d", i, len(id))
		}

		for j := i + 1; j < len(contextIDs); j++ {
			if id == contextIDs[j] {
				t.Errorf("Context IDs %d and %d are identical: %s", i, j, id)
			}
		}
	}
}

func TestTTSContextIDPattern(t *testing.T) {
	contextIDs := make([]string, 10)
	for i := 0; i < 10; i++ {
		contextIDs[i] = services.GenerateContextID()
	}

	for i, id := range contextIDs {
		if len(id) != 36 {
			t.Errorf("Expected UUID length 36, got: %d", len(id))
		}
		hyphenCount := strings.Count(id, "-")
		if hyphenCount != 4 {
			t.Errorf("Expected 4 hyphens in UUID, got: %d", hyphenCount)
		}

		for j := i + 1; j < len(contextIDs); j++ {
			if id == contextIDs[j] {
				t.Errorf("Duplicate context IDs found at positions %d and %d: %s", i, j, id)
			}
		}
	}
}

func TestTTSGetCodec(t *testing.T) {
	tests := []struct {
		outputFormat string
		expected     string
	}{
		{"riff-16khz-16bit-mono-pcm", "linear16"},
		{"riff-24khz-16bit-mono-pcm", "linear16"},
		{"audio-16khz-128kbitrate-mono-mp3", "mp3"},
		{"audio-24khz-96kbitrate-mono-mp3", "mp3"},
		{"webm-16khz-16bit-mono-opus", "opus"},
		{"ogg-24khz-16bit-mono-opus", "opus"},
		{"unknown-format", "linear16"},
	}

	for _, tt := range tests {
		t.Run(tt.outputFormat, func(t *testing.T) {
			config := TTSConfig{
				SubscriptionKey: "test-key",
				OutputFormat:    tt.outputFormat,
			}

			service := NewTTSService(config)
			codec := service.getCodec()

			if codec != tt.expected {
				t.Errorf("Expected codec %s for output format %s, got %s", tt.expected, tt.outputFormat, codec)
			}
		})
	}
}

func TestTTSParseOutputFormat(t *testing.T) {
	tests := []struct {
		outputFormat     string
		expectedRate     int
		expectedChannels int
	}{
		{"riff-8khz-16bit-mono-pcm", 8000, 1},
		{"riff-16khz-16bit-mono-pcm", 16000, 1},
		{"riff-24khz-16bit-mono-pcm", 24000, 1},
		{"riff-48khz-16bit-mono-pcm", 48000, 1},
		{"audio-16khz-128kbitrate-mono-mp3", 16000, 1},
		{"unknown-format", 16000, 1},
	}

	for _, tt := range tests {
		t.Run(tt.outputFormat, func(t *testing.T) {
			config := TTSConfig{
				SubscriptionKey: "test-key",
				OutputFormat:    tt.outputFormat,
			}

			service := NewTTSService(config)
			rate, channels := service.parseOutputFormat()

			if rate != tt.expectedRate {
				t.Errorf("Expected sample rate %d, got %d", tt.expectedRate, rate)
			}

			if channels != tt.expectedChannels {
				t.Errorf("Expected channels %d, got %d", tt.expectedChannels, channels)
			}
		})
	}
}

func TestTTSVoiceConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		config         TTSConfig
		expectedRegion string
		expectedVoice  string
	}{
		{
			name: "Custom voice configuration",
			config: TTSConfig{
				SubscriptionKey: "test-key",
				Region:          "eastus2",
				Voice:           "en-GB-RyanNeural",
			},
			expectedRegion: "eastus2",
			expectedVoice:  "en-GB-RyanNeural",
		},
		{
			name: "Default voice configuration",
			config: TTSConfig{
				SubscriptionKey: "test-key",
			},
			expectedRegion: DefaultRegion,
			expectedVoice:  DefaultVoice,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewTTSService(tt.config)

			if service.region != tt.expectedRegion {
				t.Errorf("Expected region %s, got %s", tt.expectedRegion, service.region)
			}

			if service.voice != tt.expectedVoice {
				t.Errorf("Expected voice %s, got %s", tt.expectedVoice, service.voice)
			}
		})
	}
}

func TestTTSBuildSSML(t *testing.T) {
	config := TTSConfig{
		SubscriptionKey: "test-key",
		Voice:           "en-US-JennyNeural",
	}

	service := NewTTSService(config)

	text := "Hello, world!"
	ssml := service.buildSSML(text)

	if !strings.Contains(ssml, text) {
		t.Errorf("Expected SSML to contain text '%s'", text)
	}

	if !strings.Contains(ssml, "<speak") {
		t.Error("Expected SSML to contain <speak> tag")
	}

	if !strings.Contains(ssml, "<voice") {
		t.Error("Expected SSML to contain <voice> tag")
	}

	if !strings.Contains(ssml, "en-US-JennyNeural") {
		t.Error("Expected SSML to contain voice name")
	}
}

func TestTTSSetVoice(t *testing.T) {
	config := TTSConfig{
		SubscriptionKey: "test-key",
		Voice:           "en-US-JennyNeural",
	}

	service := NewTTSService(config)

	if service.voice != "en-US-JennyNeural" {
		t.Errorf("Expected initial voice 'en-US-JennyNeural', got %s", service.voice)
	}

	service.SetVoice("en-GB-RyanNeural")

	if service.voice != "en-GB-RyanNeural" {
		t.Errorf("Expected voice to be updated to 'en-GB-RyanNeural', got %s", service.voice)
	}
}

func TestTTSSetModel(t *testing.T) {
	config := TTSConfig{
		SubscriptionKey: "test-key",
	}

	service := NewTTSService(config)

	service.SetModel("neural")
}

// TestAzureTTS_ErrorPropagation verifies that API errors are propagated
func TestAzureTTS_ErrorPropagation(t *testing.T) {
	config := TTSConfig{
		SubscriptionKey: "test-key",
		Region:          "eastus",
	}

	service := NewTTSService(config)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Try to synthesize with cancelled context
	// Should return an error (request creation will fail)
	err := service.synthesize(ctx, "Hello world")

	if err == nil {
		t.Error("Expected synthesize to return an error for cancelled context")
	}
}

