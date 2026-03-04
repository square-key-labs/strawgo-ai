package google

import (
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestNewGoogleTTSService(t *testing.T) {
	config := TTSConfig{
		APIKey:       "test-api-key",
		LanguageCode: "en-US",
		VoiceName:    "en-US-Neural2-C",
		Gender:       GenderFemale,
		Encoding:     EncodingLinear16,
		SampleRate:   16000,
	}

	service := NewGoogleTTSService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.apiKey != "test-api-key" {
		t.Errorf("Expected API key 'test-api-key', got %s", service.apiKey)
	}

	if service.voiceName != "en-US-Neural2-C" {
		t.Errorf("Expected voice 'en-US-Neural2-C', got %s", service.voiceName)
	}

	if service.languageCode != "en-US" {
		t.Errorf("Expected language code 'en-US', got %s", service.languageCode)
	}

	if service.encoding != EncodingLinear16 {
		t.Errorf("Expected encoding LINEAR16, got %s", service.encoding)
	}

	if service.sampleRate != 16000 {
		t.Errorf("Expected sample rate 16000, got %d", service.sampleRate)
	}
}

func TestNewGoogleTTSServiceDefaults(t *testing.T) {
	config := TTSConfig{
		APIKey: "test-api-key",
	}

	service := NewGoogleTTSService(config)

	if service.languageCode != DefaultLanguageCode {
		t.Errorf("Expected default language code %s, got %s", DefaultLanguageCode, service.languageCode)
	}

	if service.voiceName != DefaultVoiceName {
		t.Errorf("Expected default voice %s, got %s", DefaultVoiceName, service.voiceName)
	}

	if service.encoding != EncodingLinear16 {
		t.Errorf("Expected default encoding LINEAR16, got %s", service.encoding)
	}

	if service.sampleRate != DefaultSampleRate {
		t.Errorf("Expected default sample rate %d, got %d", DefaultSampleRate, service.sampleRate)
	}
}

func TestGoogleTTSContextIDGeneration(t *testing.T) {
	service := NewGoogleTTSService(TTSConfig{
		APIKey:       "test-key",
		LanguageCode: "en-US",
		VoiceName:    "en-US-Neural2-C",
	})

	if service.contextID != "" {
		t.Errorf("Expected empty contextID before synthesis, got: %s", service.contextID)
	}

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

func TestGoogleTTSContextIDConsistency(t *testing.T) {
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

func TestGoogleTTSContextIDPattern(t *testing.T) {
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

func TestGoogleTTSGetCodec(t *testing.T) {
	tests := []struct {
		encoding AudioEncoding
		expected string
	}{
		{EncodingLinear16, "linear16"},
		{EncodingMP3, "mp3"},
		{EncodingOggOpus, "opus"},
		{EncodingMulaw, "mulaw"},
		{EncodingAlaw, "alaw"},
	}

	for _, tt := range tests {
		t.Run(string(tt.encoding), func(t *testing.T) {
			config := TTSConfig{
				APIKey:   "test-api-key",
				Encoding: tt.encoding,
			}

			service := NewGoogleTTSService(config)
			codec := service.getCodec()

			if codec != tt.expected {
				t.Errorf("Expected codec %s for encoding %s, got %s", tt.expected, tt.encoding, codec)
			}
		})
	}
}

func TestGoogleTTSVoiceConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		config         TTSConfig
		expectedLang   string
		expectedName   string
		expectedGender VoiceGender
	}{
		{
			name: "Custom voice configuration",
			config: TTSConfig{
				APIKey:       "test-key",
				LanguageCode: "es-ES",
				VoiceName:    "es-ES-Neural2-A",
				Gender:       GenderMale,
			},
			expectedLang:   "es-ES",
			expectedName:   "es-ES-Neural2-A",
			expectedGender: GenderMale,
		},
		{
			name: "Default voice configuration",
			config: TTSConfig{
				APIKey: "test-key",
			},
			expectedLang:   DefaultLanguageCode,
			expectedName:   DefaultVoiceName,
			expectedGender: GenderFemale,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewGoogleTTSService(tt.config)

			if service.languageCode != tt.expectedLang {
				t.Errorf("Expected language code %s, got %s", tt.expectedLang, service.languageCode)
			}

			if service.voiceName != tt.expectedName {
				t.Errorf("Expected voice name %s, got %s", tt.expectedName, service.voiceName)
			}

			if service.gender != tt.expectedGender {
				t.Errorf("Expected gender %s, got %s", tt.expectedGender, service.gender)
			}
		})
	}
}

func TestGoogleTTSAudioConfiguration(t *testing.T) {
	tests := []struct {
		name         string
		config       TTSConfig
		expectedEnc  AudioEncoding
		expectedRate int
	}{
		{
			name: "Custom audio configuration",
			config: TTSConfig{
				APIKey:     "test-key",
				Encoding:   EncodingMP3,
				SampleRate: 24000,
			},
			expectedEnc:  EncodingMP3,
			expectedRate: 24000,
		},
		{
			name: "Default audio configuration",
			config: TTSConfig{
				APIKey: "test-key",
			},
			expectedEnc:  EncodingLinear16,
			expectedRate: DefaultSampleRate,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewGoogleTTSService(tt.config)

			if service.encoding != tt.expectedEnc {
				t.Errorf("Expected encoding %s, got %s", tt.expectedEnc, service.encoding)
			}

			if service.sampleRate != tt.expectedRate {
				t.Errorf("Expected sample rate %d, got %d", tt.expectedRate, service.sampleRate)
			}
		})
	}
}
