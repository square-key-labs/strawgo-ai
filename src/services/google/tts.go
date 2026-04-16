package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

const (
	// Google Cloud TTS API endpoint
	GoogleTTSURL = "https://texttospeech.googleapis.com/v1/text:synthesize"

	// Default voice settings
	DefaultLanguageCode = "en-US"
	DefaultVoiceName    = "en-US-Neural2-C"
	DefaultGender       = "FEMALE"

	// Default audio config
	DefaultEncoding   = "LINEAR16"
	DefaultSampleRate = 16000
)

// VoiceGender represents the gender of the voice
type VoiceGender string

const (
	GenderUnspecified VoiceGender = "SSML_VOICE_GENDER_UNSPECIFIED"
	GenderMale        VoiceGender = "MALE"
	GenderFemale      VoiceGender = "FEMALE"
	GenderNeutral     VoiceGender = "NEUTRAL"
)

// AudioEncoding represents the audio encoding format
type AudioEncoding string

const (
	EncodingLinear16 AudioEncoding = "LINEAR16"
	EncodingMP3      AudioEncoding = "MP3"
	EncodingOggOpus  AudioEncoding = "OGG_OPUS"
	EncodingMulaw    AudioEncoding = "MULAW"
	EncodingAlaw     AudioEncoding = "ALAW"
)

// GoogleTTSService provides text-to-speech using Google Cloud TTS API
type GoogleTTSService struct {
	*processors.BaseProcessor

	// API configuration
	apiKey         string
	serviceAccount string // Optional: path to service account JSON

	// Voice configuration
	languageCode string
	voiceName    string
	gender       VoiceGender

	// Audio configuration
	encoding   AudioEncoding
	sampleRate int

	// HTTP client
	httpClient *http.Client

	// Context ID for tracking
	contextID string

	// Lifecycle
	started bool
}

// TTSConfig holds configuration for Google TTS
type TTSConfig struct {
	APIKey         string        // Google Cloud API key
	ServiceAccount string        // Optional: path to service account JSON
	LanguageCode   string        // e.g., "en-US", "es-ES", "fr-FR"
	VoiceName      string        // e.g., "en-US-Neural2-C"
	Gender         VoiceGender   // MALE, FEMALE, NEUTRAL
	Encoding       AudioEncoding // LINEAR16, MP3, OGG_OPUS, MULAW, ALAW
	SampleRate     int           // Sample rate in Hz (e.g., 16000, 24000)
}

// NewGoogleTTSService creates a new Google TTS service
func NewGoogleTTSService(config TTSConfig) *GoogleTTSService {
	// Apply defaults
	languageCode := config.LanguageCode
	if languageCode == "" {
		languageCode = DefaultLanguageCode
	}

	voiceName := config.VoiceName
	if voiceName == "" {
		voiceName = DefaultVoiceName
	}

	gender := config.Gender
	if gender == "" {
		gender = GenderFemale
	}

	encoding := config.Encoding
	if encoding == "" {
		encoding = EncodingLinear16
	}

	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultSampleRate
	}

	service := &GoogleTTSService{
		apiKey:         config.APIKey,
		serviceAccount: config.ServiceAccount,
		languageCode:   languageCode,
		voiceName:      voiceName,
		gender:         gender,
		encoding:       encoding,
		sampleRate:     sampleRate,
		httpClient:     &http.Client{},
	}

	service.BaseProcessor = processors.NewBaseProcessor("GoogleTTS", service)
	return service
}

// Initialize initializes the service
func (s *GoogleTTSService) Initialize(ctx context.Context) error {
	logger.Debug("[GoogleTTS] Service initialized")
	return nil
}

// Cleanup cleans up resources
func (s *GoogleTTSService) Cleanup() error {
	logger.Debug("[GoogleTTS] Service cleaned up")
	return nil
}

// SetVoice sets the voice name
func (s *GoogleTTSService) SetVoice(voiceName string) {
	s.voiceName = voiceName
}

// SetModel sets the model (not used for Google TTS, but implements interface)
func (s *GoogleTTSService) SetModel(model string) {
	// Google TTS doesn't have separate models like ElevenLabs
	// Voice name determines the model
}

// HandleFrame processes frames
func (s *GoogleTTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.started = true
		return s.PushFrame(frame, direction)

	case *frames.EndFrame:
		s.started = false
		return s.PushFrame(frame, direction)

	case *frames.InterruptionFrame:
		// Reset context ID on interruption
		s.contextID = ""
		return s.PushFrame(frame, direction)

	case *frames.TextFrame:
		if f.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		return s.synthesize(ctx, f.Text)

	case *frames.LLMTextFrame:
		if f.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		return s.synthesize(ctx, f.Text)

	default:
		return s.PushFrame(frame, direction)
	}
}

// synthesize generates TTS from text
func (s *GoogleTTSService) synthesize(ctx context.Context, text string) error {
	if text == "" {
		return nil
	}

	// Generate context ID for tracking
	contextID := services.GenerateContextID()
	s.contextID = contextID

	// Emit TTSStartedFrame
	startFrame := frames.NewTTSStartedFrameWithContext(contextID)
	if err := s.PushFrame(startFrame, frames.Upstream); err != nil {
		return err
	}
	if err := s.PushFrame(startFrame, frames.Downstream); err != nil {
		return err
	}

	// Build request payload
	requestPayload := map[string]interface{}{
		"input": map[string]string{
			"text": text,
		},
		"voice": map[string]interface{}{
			"languageCode": s.languageCode,
			"name":         s.voiceName,
			"ssmlGender":   string(s.gender),
		},
		"audioConfig": map[string]interface{}{
			"audioEncoding":   string(s.encoding),
			"sampleRateHertz": s.sampleRate,
		},
	}

	requestBody, err := json.Marshal(requestPayload)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", GoogleTTSURL, bytes.NewReader(requestBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Authentication: API key or service account
	if s.apiKey != "" {
		req.Header.Set("X-Goog-Api-Key", s.apiKey)
	}
	// TODO: Add service account authentication if needed

	// Send request
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Google TTS API error (%d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result struct {
		AudioContent string `json:"audioContent"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	// Decode base64 audio
	audioData, err := base64.StdEncoding.DecodeString(result.AudioContent)
	if err != nil {
		return fmt.Errorf("failed to decode audio: %w", err)
	}

	logger.Debug("[GoogleTTS] Received audio: %d bytes", len(audioData))

	// Determine codec from encoding
	codec := s.getCodec()

	// Emit TTSAudioFrame
	audioFrame := frames.NewTTSAudioFrame(audioData, s.sampleRate, 1)
	audioFrame.SetMetadata("codec", codec)
	audioFrame.SetMetadata("context_id", contextID)
	if err := s.PushFrame(audioFrame, frames.Downstream); err != nil {
		return err
	}

	// Emit TTSStoppedFrame
	stopFrame := frames.NewTTSStoppedFrame()
	stopFrame.SetMetadata("context_id", contextID)
	if err := s.PushFrame(stopFrame, frames.Upstream); err != nil {
		return err
	}

	logger.Debug("[GoogleTTS] Synthesis complete for context: %s", contextID)
	return nil
}

// getCodec returns the codec string for the current encoding
func (s *GoogleTTSService) getCodec() string {
	switch s.encoding {
	case EncodingLinear16:
		return "linear16"
	case EncodingMP3:
		return "mp3"
	case EncodingOggOpus:
		return "opus"
	case EncodingMulaw:
		return "mulaw"
	case EncodingAlaw:
		return "alaw"
	default:
		return "linear16"
	}
}
