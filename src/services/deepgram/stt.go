package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/url"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/gorilla/websocket"
)

// STTService provides speech-to-text using Deepgram
type STTService struct {
	*processors.BaseProcessor
	apiKey   string
	language string
	model    string
	encoding string
	conn     *websocket.Conn
	ctx      context.Context
	cancel   context.CancelFunc
}

// STTConfig holds configuration for Deepgram
type STTConfig struct {
	APIKey   string
	Language string // e.g., "en-US"
	Model    string // e.g., "nova-2"
	Encoding string // Supported: "mulaw"/"ulaw", "alaw", "linear16" (default: "linear16")
}

// NewSTTService creates a new Deepgram STT service
func NewSTTService(config STTConfig) *STTService {
	encoding := config.Encoding
	if encoding == "" {
		encoding = "linear16" // Default to PCM
	}

	// Normalize codec names for Deepgram API
	encoding = normalizeDeepgramEncoding(encoding)

	ds := &STTService{
		apiKey:   config.APIKey,
		language: config.Language,
		model:    config.Model,
		encoding: encoding,
	}
	ds.BaseProcessor = processors.NewBaseProcessor("DeepgramSTT", ds)
	return ds
}

// normalizeDeepgramEncoding converts codec name variations to Deepgram API format
func normalizeDeepgramEncoding(encoding string) string {
	switch encoding {
	case "ulaw", "PCMU":
		return "mulaw"
	case "PCMA":
		return "alaw"
	case "pcm", "PCM":
		return "linear16"
	default:
		return encoding
	}
}

func (s *STTService) SetLanguage(lang string) {
	s.language = lang
}

func (s *STTService) SetModel(model string) {
	s.model = model
}

func (s *STTService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	// Determine sample rate based on encoding
	sampleRate := "16000" // Default for linear16
	if s.encoding == "mulaw" || s.encoding == "ulaw" || s.encoding == "alaw" {
		sampleRate = "8000" // Telephony codecs (mulaw/alaw) are typically 8kHz
	}

	// Build WebSocket URL
	params := url.Values{}
	params.Set("language", s.language)
	params.Set("model", s.model)
	params.Set("encoding", s.encoding)
	params.Set("sample_rate", sampleRate)
	params.Set("channels", "1")
	params.Set("interim_results", "true")

	wsURL := fmt.Sprintf("wss://api.deepgram.com/v1/listen?%s", params.Encode())

	// Connect to Deepgram
	header := map[string][]string{
		"Authorization": {fmt.Sprintf("Token %s", s.apiKey)},
	}

	var err error
	s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, header)
	if err != nil {
		return fmt.Errorf("failed to connect to Deepgram: %w", err)
	}

	// Start receiving transcriptions
	go s.receiveTranscriptions()

	log.Printf("[DeepgramSTT] Connected and initialized")
	return nil
}

func (s *STTService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	if s.conn != nil {
		s.conn.Close()
	}
	return nil
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Process audio frames
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		// Send audio data to Deepgram
		if s.conn != nil {
			if err := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data); err != nil {
				log.Printf("[DeepgramSTT] Error sending audio: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		// Don't pass audio frame downstream (converted to transcription)
		return nil
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

func (s *STTService) receiveTranscriptions() {
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
			_, message, err := s.conn.ReadMessage()
			if err != nil {
				log.Printf("[DeepgramSTT] Error reading message: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			// Parse Deepgram response
			var response struct {
				IsFinal bool `json:"is_final"`
				Channel struct {
					Alternatives []struct {
						Transcript string  `json:"transcript"`
						Confidence float64 `json:"confidence"`
					} `json:"alternatives"`
				} `json:"channel"`
			}

			if err := json.Unmarshal(message, &response); err != nil {
				log.Printf("[DeepgramSTT] Error parsing response: %v", err)
				continue
			}

			// Extract transcript
			if len(response.Channel.Alternatives) > 0 {
				transcript := response.Channel.Alternatives[0].Transcript
				if transcript != "" {
					transcriptionFrame := frames.NewTranscriptionFrame(transcript, response.IsFinal)
					log.Printf("[DeepgramSTT] Transcription (final=%v): %s", response.IsFinal, transcript)
					s.PushFrame(transcriptionFrame, frames.Downstream)
				}
			}
		}
	}
}
