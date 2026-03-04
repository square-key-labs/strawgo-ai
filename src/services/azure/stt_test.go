package azure

import (
	"context"
	"testing"
	"time"
)

func TestNewSTTService(t *testing.T) {
	config := STTConfig{
		SubscriptionKey: "test-subscription-key",
		Region:          "westus",
		Language:        "es-ES",
		Encoding:        "audio/x-wav",
		SampleRate:      16000,
	}

	service := NewSTTService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.subscriptionKey != "test-subscription-key" {
		t.Errorf("Expected subscription key 'test-subscription-key', got %s", service.subscriptionKey)
	}

	if service.region != "westus" {
		t.Errorf("Expected region 'westus', got %s", service.region)
	}

	if service.language != "es-ES" {
		t.Errorf("Expected language 'es-ES', got %s", service.language)
	}

	if service.encoding != "audio/x-wav" {
		t.Errorf("Expected encoding 'audio/x-wav', got %s", service.encoding)
	}

	if service.sampleRate != 16000 {
		t.Errorf("Expected sample rate 16000, got %d", service.sampleRate)
	}
}

func TestNewSTTServiceDefaults(t *testing.T) {
	config := STTConfig{
		SubscriptionKey: "test-key",
	}

	service := NewSTTService(config)

	if service.region != DefaultRegion {
		t.Errorf("Expected default region %s, got %s", DefaultRegion, service.region)
	}

	if service.language != DefaultLanguage {
		t.Errorf("Expected default language %s, got %s", DefaultLanguage, service.language)
	}

	if service.encoding != DefaultEncoding {
		t.Errorf("Expected default encoding %s, got %s", DefaultEncoding, service.encoding)
	}

	if service.sampleRate != DefaultSampleRate {
		t.Errorf("Expected default sample rate %d, got %d", DefaultSampleRate, service.sampleRate)
	}
}

func TestSTTServiceConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		config         STTConfig
		expectedRegion string
		expectedLang   string
		expectedEnc    string
		expectedRate   int
	}{
		{
			name: "Custom configuration",
			config: STTConfig{
				SubscriptionKey: "test-key",
				Region:          "eastus2",
				Language:        "fr-FR",
				Encoding:        "audio/x-wav",
				SampleRate:      8000,
			},
			expectedRegion: "eastus2",
			expectedLang:   "fr-FR",
			expectedEnc:    "audio/x-wav",
			expectedRate:   8000,
		},
		{
			name: "Default configuration",
			config: STTConfig{
				SubscriptionKey: "test-key",
			},
			expectedRegion: DefaultRegion,
			expectedLang:   DefaultLanguage,
			expectedEnc:    DefaultEncoding,
			expectedRate:   DefaultSampleRate,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewSTTService(tt.config)

			if service.region != tt.expectedRegion {
				t.Errorf("Expected region %s, got %s", tt.expectedRegion, service.region)
			}

			if service.language != tt.expectedLang {
				t.Errorf("Expected language %s, got %s", tt.expectedLang, service.language)
			}

			if service.encoding != tt.expectedEnc {
				t.Errorf("Expected encoding %s, got %s", tt.expectedEnc, service.encoding)
			}

			if service.sampleRate != tt.expectedRate {
				t.Errorf("Expected sample rate %d, got %d", tt.expectedRate, service.sampleRate)
			}
		})
	}
}

func TestSTTSetLanguage(t *testing.T) {
	config := STTConfig{
		SubscriptionKey: "test-key",
		Language:        "en-US",
	}

	service := NewSTTService(config)

	if service.language != "en-US" {
		t.Errorf("Expected initial language 'en-US', got %s", service.language)
	}

	service.SetLanguage("de-DE")

	if service.language != "de-DE" {
		t.Errorf("Expected language to be updated to 'de-DE', got %s", service.language)
	}
}

func TestSTTSetModel(t *testing.T) {
	config := STTConfig{
		SubscriptionKey: "test-key",
	}

	service := NewSTTService(config)

	service.SetModel("conversation")
}

// TestAzureSTT_ErrorPropagation verifies that API errors are propagated
func TestAzureSTT_ErrorPropagation(t *testing.T) {
	config := STTConfig{
		SubscriptionKey: "invalid-key",
		Region:          "invalid-region",
	}

	service := NewSTTService(config)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Try to initialize with invalid credentials
	// Should return an error (WebSocket dial will fail)
	err := service.Initialize(ctx)

	if err == nil {
		t.Error("Expected Initialize to return an error for invalid region")
	}
}

// TestAzureSTT_Keepalive verifies that keepalive mechanism is configured correctly
func TestAzureSTT_Keepalive(t *testing.T) {
	config := STTConfig{
		SubscriptionKey:   "test-key",
		KeepaliveInterval: 3 * time.Second,
		KeepaliveTimeout:  15 * time.Second,
	}
	
	service := NewSTTService(config)
	
	if service.keepaliveInterval != 3*time.Second {
		t.Errorf("Expected keepaliveInterval 3s, got %v", service.keepaliveInterval)
	}
	
	if service.keepaliveTimeout != 15*time.Second {
		t.Errorf("Expected keepaliveTimeout 15s, got %v", service.keepaliveTimeout)
	}
}

// TestAzureSTT_KeepaliveDefaults verifies that keepalive defaults are applied
func TestAzureSTT_KeepaliveDefaults(t *testing.T) {
	config := STTConfig{
		SubscriptionKey: "test-key",
	}
	
	service := NewSTTService(config)
	
	if service.keepaliveInterval != 5*time.Second {
		t.Errorf("Expected default keepaliveInterval 5s, got %v", service.keepaliveInterval)
	}
	
	if service.keepaliveTimeout != 30*time.Second {
		t.Errorf("Expected default keepaliveTimeout 30s, got %v", service.keepaliveTimeout)
	}
}
