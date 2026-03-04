package whisper

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	// WhisperAPIURL is the OpenAI Whisper API endpoint
	WhisperAPIURL = "https://api.openai.com/v1/audio/transcriptions"

	// DefaultModel is the default Whisper model
	DefaultModel = "whisper-1"

	// MaxBufferDuration prevents infinite audio accumulation
	MaxBufferDuration = 30 * time.Second

	// DefaultSampleRate for audio (16kHz, 16-bit, mono PCM)
	DefaultSampleRate = 16000

	// DefaultChannels for audio
	DefaultChannels = 1
)

// WhisperSTTService provides speech-to-text using OpenAI Whisper API.
// Unlike streaming STT services (like Deepgram), Whisper requires batch processing:
// accumulate audio until user stops speaking, then transcribe the complete segment.
type WhisperSTTService struct {
	*processors.BaseProcessor

	apiKey     string
	model      string
	language   string
	httpClient *http.Client
	apiURL     string // Configurable for testing

	// Audio accumulation state
	audioBuffer  []byte
	bufferStart  time.Time
	accumulating bool

	// Audio format
	sampleRate int
	channels   int

	// Service lifecycle
	started bool
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewWhisperSTTService creates a new Whisper STT service with default configuration
func NewWhisperSTTService(apiKey string) *WhisperSTTService {
	return NewWhisperSTTServiceWithConfig(WhisperSTTConfig{
		APIKey:     apiKey,
		Model:      DefaultModel,
		Language:   "en",
		SampleRate: DefaultSampleRate,
		Channels:   DefaultChannels,
	})
}

// WhisperSTTConfig holds configuration for Whisper STT
type WhisperSTTConfig struct {
	APIKey     string
	Model      string
	Language   string
	SampleRate int
	Channels   int
}

// NewWhisperSTTServiceWithConfig creates a new Whisper STT service with custom configuration
func NewWhisperSTTServiceWithConfig(config WhisperSTTConfig) *WhisperSTTService {
	model := config.Model
	if model == "" {
		model = DefaultModel
	}

	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultSampleRate
	}

	channels := config.Channels
	if channels == 0 {
		channels = DefaultChannels
	}

	service := &WhisperSTTService{
		apiKey:      config.APIKey,
		model:       model,
		language:    config.Language,
		httpClient:  &http.Client{Timeout: 30 * time.Second},
		apiURL:      WhisperAPIURL,
		audioBuffer: make([]byte, 0),
		sampleRate:  sampleRate,
		channels:    channels,
	}

	service.BaseProcessor = processors.NewBaseProcessor("WhisperSTT", service)
	return service
}

// SetLanguage sets the language for transcription
func (s *WhisperSTTService) SetLanguage(lang string) {
	s.language = lang
}

// SetModel sets the Whisper model to use
func (s *WhisperSTTService) SetModel(model string) {
	s.model = model
}

// Initialize initializes the service
func (s *WhisperSTTService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)
	logger.Info("[WhisperSTT] Initialized")
	return nil
}

// Cleanup cleans up resources
func (s *WhisperSTTService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	s.resetBuffer()
	logger.Info("[WhisperSTT] Cleaned up")
	return nil
}

// HandleFrame processes frames through the Whisper STT pipeline
func (s *WhisperSTTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		s.started = true
		// Lazy initialization on first frame
		if s.ctx == nil {
			if err := s.Initialize(ctx); err != nil {
				logger.Error("[WhisperSTT] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		// Emit STT metadata for auto-tuning turn detection
		s.PushFrame(frames.NewSTTMetadataFrame("whisper", 2000*time.Millisecond), frames.Downstream)
		return s.PushFrame(frame, direction)

	case *frames.EndFrame:
		logger.Info("[WhisperSTT] Received EndFrame, cleaning up")
		s.started = false
		s.resetBuffer()
		if err := s.Cleanup(); err != nil {
			logger.Error("[WhisperSTT] Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)

	case *frames.InterruptionFrame:
		// Discard accumulated audio on interruption
		logger.Info("[WhisperSTT] Received InterruptionFrame, discarding audio buffer")
		s.resetBuffer()
		return s.PushFrame(frame, direction)

	case *frames.AudioFrame:
		// Accumulate audio frames
		return s.accumulateAudio(ctx, f)

	case *frames.UserStartedSpeakingFrame:
		// Start accumulating audio
		logger.Info("[WhisperSTT] User started speaking, starting audio accumulation")
		s.startAccumulation()
		return s.PushFrame(frame, direction)

	case *frames.UserStoppedSpeakingFrame:
		// User stopped speaking - transcribe accumulated audio
		logger.Info("[WhisperSTT] User stopped speaking, transcribing accumulated audio")
		if err := s.transcribeAccumulatedAudio(ctx); err != nil {
			logger.Error("[WhisperSTT] Transcription failed: %v", err)
			return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
		}
		// Always push the UserStoppedSpeakingFrame downstream
		return s.PushFrame(frame, direction)

	default:
		// Pass all other frames through
		return s.PushFrame(frame, direction)
	}
}

// startAccumulation begins audio accumulation
func (s *WhisperSTTService) startAccumulation() {
	s.accumulating = true
	s.bufferStart = time.Now()
	s.audioBuffer = make([]byte, 0, 1024*1024) // Pre-allocate 1MB
}

// accumulateAudio adds audio data to the buffer
func (s *WhisperSTTService) accumulateAudio(ctx context.Context, frame *frames.AudioFrame) error {
	if !s.accumulating {
		// Not accumulating yet, just pass through
		return s.PushFrame(frame, frames.Downstream)
	}

	// Check timeout to prevent infinite accumulation
	if time.Since(s.bufferStart) > MaxBufferDuration {
		logger.Warn("[WhisperSTT] Audio buffer timeout - transcribing accumulated audio")
		if err := s.transcribeAccumulatedAudio(ctx); err != nil {
			logger.Error("[WhisperSTT] Transcription after timeout failed: %v", err)
			return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
		}
		return s.PushFrame(frame, frames.Downstream)
	}

	// Append audio to buffer
	s.audioBuffer = append(s.audioBuffer, frame.Data...)

	// Pass AudioFrame downstream for other processors
	return s.PushFrame(frame, frames.Downstream)
}

// transcribeAccumulatedAudio sends the accumulated audio to Whisper API
func (s *WhisperSTTService) transcribeAccumulatedAudio(ctx context.Context) error {
	if len(s.audioBuffer) == 0 {
		logger.Debug("[WhisperSTT] No audio to transcribe, skipping")
		s.resetBuffer()
		return nil
	}

	logger.Info("[WhisperSTT] Transcribing %d bytes of audio", len(s.audioBuffer))

	// Create WAV file from raw PCM audio
	wavData := s.createWAVFile(s.audioBuffer)

	// Create multipart form data
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Add audio file
	part, err := writer.CreateFormFile("file", "audio.wav")
	if err != nil {
		s.resetBuffer()
		return fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err := part.Write(wavData); err != nil {
		s.resetBuffer()
		return fmt.Errorf("failed to write audio data: %w", err)
	}

	// Add model parameter
	if err := writer.WriteField("model", s.model); err != nil {
		s.resetBuffer()
		return fmt.Errorf("failed to write model field: %w", err)
	}

	// Add language parameter if set
	if s.language != "" {
		if err := writer.WriteField("language", s.language); err != nil {
			s.resetBuffer()
			return fmt.Errorf("failed to write language field: %w", err)
		}
	}

	writer.Close()

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", s.apiURL, &requestBody)
	if err != nil {
		s.resetBuffer()
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	resp, err := s.httpClient.Do(req)
	if err != nil {
		s.resetBuffer()
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		s.resetBuffer()
		return fmt.Errorf("API error: %s - %s", resp.Status, string(body))
	}

	// Parse response
	var result struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		s.resetBuffer()
		return fmt.Errorf("failed to decode response: %w", err)
	}

	logger.Info("[WhisperSTT] Transcription (final=true): %s", result.Text)

	// Emit TranscriptionFrame
	transcriptionFrame := frames.NewTranscriptionFrame(result.Text, true)
	if err := s.PushFrame(transcriptionFrame, frames.Downstream); err != nil {
		s.resetBuffer()
		return err
	}

	// Reset buffer after successful transcription
	s.resetBuffer()

	return nil
}

// createWAVFile creates a WAV file from raw PCM audio data
func (s *WhisperSTTService) createWAVFile(audioData []byte) []byte {
	// WAV file format:
	// RIFF header (12 bytes)
	// fmt chunk (24 bytes)
	// data chunk header (8 bytes)
	// audio data

	dataSize := uint32(len(audioData))
	fileSize := uint32(36 + dataSize) // 44 bytes header - 8 bytes + data size

	byteRate := uint32(s.sampleRate * s.channels * 2) // 2 bytes per sample (16-bit)
	blockAlign := uint16(s.channels * 2)

	buf := new(bytes.Buffer)

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, fileSize)
	buf.WriteString("WAVE")

	// fmt chunk
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))           // fmt chunk size
	binary.Write(buf, binary.LittleEndian, uint16(1))            // PCM format
	binary.Write(buf, binary.LittleEndian, uint16(s.channels))   // Number of channels
	binary.Write(buf, binary.LittleEndian, uint32(s.sampleRate)) // Sample rate
	binary.Write(buf, binary.LittleEndian, byteRate)             // Byte rate
	binary.Write(buf, binary.LittleEndian, blockAlign)           // Block align
	binary.Write(buf, binary.LittleEndian, uint16(16))           // Bits per sample

	// data chunk
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, dataSize)

	// Audio data
	buf.Write(audioData)

	return buf.Bytes()
}

// resetBuffer clears the audio buffer and stops accumulation
func (s *WhisperSTTService) resetBuffer() {
	s.audioBuffer = make([]byte, 0)
	s.accumulating = false
}
