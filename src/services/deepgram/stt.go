package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
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
	connMu   sync.Mutex // Protects concurrent WebSocket writes
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

	// Start keepalive task to prevent timeout
	go s.keepaliveTask()

	log.Printf("[DeepgramSTT] Connected and initialized")
	return nil
}

func (s *STTService) Cleanup() error {
	// Cancel context first to signal goroutines to stop
	if s.cancel != nil {
		s.cancel()
	}

	// Give goroutines a moment to see the context cancellation
	time.Sleep(50 * time.Millisecond)

	// Now close the connection
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}
	return nil
}

func (s *STTService) reconnect(ctx context.Context) error {
	// Close old connection if exists
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}

	// Cancel old context and create new one
	if s.cancel != nil {
		s.cancel()
	}

	// Reinitialize
	return s.Initialize(ctx)
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Pass StartFrame through without initializing (lazy initialization on first audio)
	if _, ok := frame.(*frames.StartFrame); ok {
		return s.PushFrame(frame, direction)
	}

	// Handle EndFrame - cleanup and close connection
	if _, ok := frame.(*frames.EndFrame); ok {
		log.Printf("[DeepgramSTT] Received EndFrame, cleaning up")
		if err := s.Cleanup(); err != nil {
			log.Printf("[DeepgramSTT] Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - send finalize to reset Deepgram stream
	// This prevents old transcription fragments from arriving after interruption
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		log.Printf("[DeepgramSTT] Received InterruptionFrame, sending finalize to reset stream")
		if s.conn != nil {
			// Send finalize message to tell Deepgram to flush current utterance
			// This prevents stale transcription fragments from leaking through
			finalizeMsg := map[string]interface{}{
				"type": "Finalize",
			}
			s.connMu.Lock()
			err := s.conn.WriteJSON(finalizeMsg)
			s.connMu.Unlock()

			if err != nil {
				log.Printf("[DeepgramSTT] Error sending finalize message: %v", err)
			} else {
				log.Printf("[DeepgramSTT] ✓ Sent finalize message to reset STT stream")
			}
		}
		// Pass the interruption frame downstream
		return s.PushFrame(frame, direction)
	}

	// Process audio frames
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		// Lazy initialization on first audio frame
		if s.conn == nil {
			log.Printf("[DeepgramSTT] Lazy initializing on first AudioFrame")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[DeepgramSTT] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}

		// Send audio data to Deepgram (with mutex protection)
		s.connMu.Lock()
		err := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
		s.connMu.Unlock()

		if err != nil {
			log.Printf("[DeepgramSTT] Error sending audio: %v", err)

			// Attempt to reconnect once
			log.Printf("[DeepgramSTT] Attempting to reconnect...")
			if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
				log.Printf("[DeepgramSTT] Reconnection failed: %v", reconnectErr)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}

			// Retry sending after reconnect (with mutex protection)
			s.connMu.Lock()
			retryErr := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
			s.connMu.Unlock()

			if retryErr != nil {
				log.Printf("[DeepgramSTT] Error sending audio after reconnect: %v", retryErr)
				return s.PushFrame(frames.NewErrorFrame(retryErr), frames.Upstream)
			}
		}

		// IMPORTANT: Pass AudioFrame downstream for audio-based interruption detection
		// LLMUserAggregator needs AudioFrames to analyze user speech patterns
		return s.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

func (s *STTService) receiveTranscriptions() {
	for {
		select {
		case <-s.ctx.Done():
			log.Printf("[DeepgramSTT] Context cancelled, stopping transcription receiver")
			return
		default:
			_, message, err := s.conn.ReadMessage()
			if err != nil {
				// Check if this is a normal closure during shutdown
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					log.Printf("[DeepgramSTT] Connection closed normally")
					return
				}
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

func (s *STTService) keepaliveTask() {
	// Deepgram expects audio or a message within ~10 seconds
	// Send keepalive every 5 seconds to be safe
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if s.conn != nil {
				// Send a JSON keepalive message (with mutex protection)
				keepalive := map[string]string{"type": "KeepAlive"}
				s.connMu.Lock()
				err := s.conn.WriteJSON(keepalive)
				s.connMu.Unlock()

				if err != nil {
					log.Printf("[DeepgramSTT] Error sending keepalive: %v", err)
					return
				}
				// log.Printf("[DeepgramSTT] Sent keepalive")
			}
		}
	}
}
