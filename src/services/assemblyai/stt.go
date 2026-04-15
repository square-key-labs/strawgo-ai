package assemblyai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
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
	domain                       string // maps to "language_model" URL param (e.g. "medical-v1")
	sampleRate                   int
	endUtteranceSilenceThreshold int // milliseconds
	baseURL                      string
	onEndOfTurn                  func(transcript string) // called after each FinalTranscript
	conn                         *websocket.Conn
	ctx                          context.Context
	cancel                       context.CancelFunc
	connMu                       sync.Mutex // Protects concurrent WebSocket writes
	log                          *logger.Logger
}

// STTConfig holds configuration for AssemblyAI STT
type STTConfig struct {
	APIKey                       string
	Language                     string // e.g., "en"
	Model                        string // e.g., "u3-rt-pro"
	Domain                       string // Domain-specific acoustic model (e.g. "medical-v1"); maps to language_model URL param
	SampleRate                   int    // Audio sample rate (default: 16000)
	EndUtteranceSilenceThreshold int    // Silence threshold in ms (default: 700)
	BaseURL                      string // WebSocket URL override (for testing)

	// OnEndOfTurn is called after each final transcript is received (end of utterance).
	// It is invoked after the TranscriptionFrame has been pushed, so it cannot race with it.
	// Example use: trigger a pipeline action or log turn boundaries.
	OnEndOfTurn func(transcript string)
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
		domain:                       config.Domain,
		sampleRate:                   sampleRate,
		endUtteranceSilenceThreshold: endUtteranceSilenceThreshold,
		baseURL:                      baseURL,
		onEndOfTurn:                  config.OnEndOfTurn,
		log:                          logger.WithPrefix("AssemblyAISTT"),
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
	if s.domain != "" {
		wsURL += "&language_model=" + url.QueryEscape(s.domain)
	}

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

	s.log.Info("Connected and initialized (model=%s, sample_rate=%d, silence_threshold=%dms)",
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
		s.log.Info("Received EndFrame, cleaning up")
		if err := s.Cleanup(); err != nil {
			s.log.Warn("Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - send force end utterance to reset stream
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.log.Info("Received InterruptionFrame, sending force end utterance")
		if s.conn != nil {
			forceEnd := map[string]bool{"force_end_utterance": true}
			s.connMu.Lock()
			err := s.conn.WriteJSON(forceEnd)
			s.connMu.Unlock()

			if err != nil {
				s.log.Debug("Error sending force end utterance: %v", err)
			} else {
				s.log.Debug("Sent force end utterance to reset STT stream")
			}
		}
		return s.PushFrame(frame, direction)
	}

	// Process audio frames
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		// Lazy initialization on first audio frame
		if s.conn == nil {
			s.log.Info("Lazy initializing on first AudioFrame")
			if err := s.Initialize(ctx); err != nil {
				s.log.Error("Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}

		// Send audio data to AssemblyAI as binary message (with mutex protection)
		s.connMu.Lock()
		err := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
		s.connMu.Unlock()

		if err != nil {
			s.log.Warn("Error sending audio: %v", err)

			// Attempt to reconnect once
			s.log.Warn("Attempting to reconnect...")
			if reconnectErr := s.reconnect(ctx); reconnectErr != nil {
				s.log.Error("Reconnection failed: %v", reconnectErr)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
			s.log.Info("Reconnected successfully")

			// Retry sending after reconnect (with mutex protection)
			s.connMu.Lock()
			retryErr := s.conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
			s.connMu.Unlock()

			if retryErr != nil {
				s.log.Error("Error sending audio after reconnect: %v", retryErr)
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
			s.log.Info("Context cancelled, stopping transcription receiver")
			return
		default:
			_, message, err := s.conn.ReadMessage()
			if err != nil {
				// Check if this is a normal closure during shutdown
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					s.log.Debug("Connection closed normally")
					return
				}
				s.log.Warn("Error reading message: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			// Parse AssemblyAI response
			var response transcriptMessage
			if err := json.Unmarshal(message, &response); err != nil {
				s.log.Warn("Error parsing response: %v", err)
				continue
			}

			// Process based on message type
			switch response.MessageType {
			case "PartialTranscript":
				if response.Text != "" {
					transcriptionFrame := frames.NewTranscriptionFrame(response.Text, false)
					s.log.Debug("Partial transcript: %s", response.Text)
					s.PushFrame(transcriptionFrame, frames.Downstream)
				}
			case "FinalTranscript":
				if response.Text != "" {
					transcriptionFrame := frames.NewTranscriptionFrame(response.Text, true)
					s.log.Info("Final transcript: %s", response.Text)
					s.PushFrame(transcriptionFrame, frames.Downstream)
					// on_end_of_turn fires after the frame is pushed so it cannot
					// race with TranscriptionFrame consumers downstream.
					if s.onEndOfTurn != nil {
						s.onEndOfTurn(response.Text)
					}
				}
			case "SessionBegins":
				s.log.Info("Session started")
			case "SessionTerminated":
				s.log.Info("Session terminated")
				return
			default:
				// Ignore unknown message types (e.g., "SessionInformation")
				s.log.Debug("Received message type: %s", response.MessageType)
			}
		}
	}
}
