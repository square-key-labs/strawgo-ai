package assemblyai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	// DefaultSampleRate is the default audio sample rate for AssemblyAI real-time transcription
	DefaultSampleRate = 16000

	// DefaultEndUtteranceSilenceThreshold is the default silence threshold in milliseconds
	// that AssemblyAI uses to detect end of utterance
	DefaultEndUtteranceSilenceThreshold = 700

	// DefaultModel is the latest AssemblyAI real-time model
	DefaultModel = "u3-rt-pro"

	// DefaultBaseURL is the AssemblyAI real-time WebSocket endpoint
	DefaultBaseURL = "wss://api.assemblyai.com/v2/realtime/ws"
)

// STTService provides speech-to-text using AssemblyAI real-time transcription
type STTService struct {
	*processors.BaseProcessor
	apiKey                       string
	language                     string
	model                        string
	sampleRate                   int
	endUtteranceSilenceThreshold int // milliseconds
	baseURL                      string
	conn                         *websocket.Conn
	ctx                          context.Context
	cancel                       context.CancelFunc
	connMu                       sync.Mutex // Protects concurrent WebSocket writes
}

// STTConfig holds configuration for AssemblyAI STT
type STTConfig struct {
	APIKey                       string
	Language                     string // e.g., "en"
	Model                        string // e.g., "u3-rt-pro"
	SampleRate                   int    // Audio sample rate (default: 16000)
	EndUtteranceSilenceThreshold int    // Silence threshold in ms (default: 700)
	BaseURL                      string // WebSocket URL override (for testing)
}

// sessionConfig is the JSON message sent after WebSocket connect to configure the session
type sessionConfig struct {
	EndUtteranceSilenceThreshold int `json:"end_utterance_silence_threshold"`
}

// transcriptMessage represents an AssemblyAI real-time transcript response
type transcriptMessage struct {
	MessageType string  `json:"message_type"`
	Text        string  `json:"text"`
	Confidence  float64 `json:"confidence"`
}

// terminateMessage is sent to gracefully close the session
type terminateMessage struct {
	TerminateSession bool `json:"terminate_session"`
}

// NewSTTService creates a new AssemblyAI STT service
func NewSTTService(config STTConfig) *STTService {
	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultSampleRate
	}

	model := config.Model
	if model == "" {
		model = DefaultModel
	}

	endUtteranceSilenceThreshold := config.EndUtteranceSilenceThreshold
	if endUtteranceSilenceThreshold == 0 {
		endUtteranceSilenceThreshold = DefaultEndUtteranceSilenceThreshold
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	s := &STTService{
		apiKey:                       config.APIKey,
		language:                     config.Language,
		model:                        model,
		sampleRate:                   sampleRate,
		endUtteranceSilenceThreshold: endUtteranceSilenceThreshold,
		baseURL:                      baseURL,
	}
	s.BaseProcessor = processors.NewBaseProcessor("AssemblyAISTT", s)
	return s
}

func (s *STTService) SetLanguage(lang string) {
	s.language = lang
}

func (s *STTService) SetModel(model string) {
	s.model = model
}

func (s *STTService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	// Build WebSocket URL with auth token and sample rate
	wsURL := fmt.Sprintf("%s?sample_rate=%d&token=%s", s.baseURL, s.sampleRate, s.apiKey)

	// Connect to AssemblyAI
	var err error
	s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to AssemblyAI: %w", err)
	}

	// Send session configuration
	cfg := sessionConfig{
		EndUtteranceSilenceThreshold: s.endUtteranceSilenceThreshold,
	}
	s.connMu.Lock()
	err = s.conn.WriteJSON(cfg)
	s.connMu.Unlock()
	if err != nil {
		s.conn.Close()
		s.conn = nil
		return fmt.Errorf("failed to send session config to AssemblyAI: %w", err)
	}

	// Start receiving transcriptions
	go s.receiveTranscriptions()

	log.Printf("[AssemblyAISTT] Connected and initialized (model=%s, sample_rate=%d, silence_threshold=%dms)",
		s.model, s.sampleRate, s.endUtteranceSilenceThreshold)
	return nil
}

func (s *STTService) Cleanup() error {
	// Cancel context first to signal goroutines to stop
	if s.cancel != nil {
		s.cancel()
	}

	// Give goroutines a moment to see the context cancellation
	time.Sleep(50 * time.Millisecond)

	// Send terminate session message before closing
	if s.conn != nil {
		s.connMu.Lock()
		_ = s.conn.WriteJSON(terminateMessage{TerminateSession: true})
		s.connMu.Unlock()

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
		log.Printf("[AssemblyAISTT] Received EndFrame, cleaning up")
		if err := s.Cleanup(); err != nil {
			log.Printf("[AssemblyAISTT] Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - send force end utterance to reset stream
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		log.Printf("[AssemblyAISTT] Received InterruptionFrame, sending force end utterance")
		if s.conn != nil {
			forceEnd := map[string]bool{"force_end_utterance": true}
			s.connMu.Lock()
			err := s.conn.WriteJSON(forceEnd)
			s.connMu.Unlock()

			if err != nil {
				log.Printf("[AssemblyAISTT] Error sending force end utterance: %v", err)
			} else {
				log.Printf("[AssemblyAISTT] ✓ Sent force end utterance to reset STT stream")
			}
		}
		return s.PushFrame(frame, direction)
	}

	// Process audio frames
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		// Lazy initialization on first audio frame
		if s.conn == nil {
			log.Printf("[AssemblyAISTT] Lazy initializing on first AudioFrame")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[AssemblyAISTT] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}

		// Send audio data to AssemblyAI as binary message (with mutex protection)
		s.connMu.Lock()
		err := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
		s.connMu.Unlock()

		if err != nil {
			log.Printf("[AssemblyAISTT] Error sending audio: %v", err)

			// Attempt to reconnect once
			log.Printf("[AssemblyAISTT] Attempting to reconnect...")
			if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
				log.Printf("[AssemblyAISTT] Reconnection failed: %v", reconnectErr)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}

			// Retry sending after reconnect (with mutex protection)
			s.connMu.Lock()
			retryErr := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
			s.connMu.Unlock()

			if retryErr != nil {
				log.Printf("[AssemblyAISTT] Error sending audio after reconnect: %v", retryErr)
				return s.PushFrame(frames.NewErrorFrame(retryErr), frames.Upstream)
			}
		}

		// Pass AudioFrame downstream for audio-based interruption detection
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
			log.Printf("[AssemblyAISTT] Context cancelled, stopping transcription receiver")
			return
		default:
			_, message, err := s.conn.ReadMessage()
			if err != nil {
				// Check if this is a normal closure during shutdown
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					log.Printf("[AssemblyAISTT] Connection closed normally")
					return
				}
				log.Printf("[AssemblyAISTT] Error reading message: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			// Parse AssemblyAI response
			var response transcriptMessage
			if err := json.Unmarshal(message, &response); err != nil {
				log.Printf("[AssemblyAISTT] Error parsing response: %v", err)
				continue
			}

			// Process based on message type
			switch response.MessageType {
			case "PartialTranscript":
				if response.Text != "" {
					transcriptionFrame := frames.NewTranscriptionFrame(response.Text, false)
					log.Printf("[AssemblyAISTT] Partial transcript: %s", response.Text)
					s.PushFrame(transcriptionFrame, frames.Downstream)
				}
			case "FinalTranscript":
				if response.Text != "" {
					transcriptionFrame := frames.NewTranscriptionFrame(response.Text, true)
					log.Printf("[AssemblyAISTT] Final transcript: %s", response.Text)
					s.PushFrame(transcriptionFrame, frames.Downstream)
				}
			case "SessionBegins":
				log.Printf("[AssemblyAISTT] Session started")
			case "SessionTerminated":
				log.Printf("[AssemblyAISTT] Session terminated")
				return
			default:
				// Ignore unknown message types (e.g., "SessionInformation")
				log.Printf("[AssemblyAISTT] Received message type: %s", response.MessageType)
			}
		}
	}
}
