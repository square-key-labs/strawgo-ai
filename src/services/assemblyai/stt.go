package assemblyai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"

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
	readWG                       sync.WaitGroup
	connDropped                  atomic.Bool
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

// SetLanguage stores the value but the AssemblyAI real-time websocket URL
// in this build is not derived from it. The setter exists to satisfy the
// services.STTService interface contract.
//
// Deprecated: no-op for AssemblyAI in this implementation. Use the
// "domain" key of UpdateSettings (or the Domain config field) to influence
// recognition behavior.
func (s *STTService) SetLanguage(lang string) {
	s.language = lang
}

// SetModel stores the value but the AssemblyAI real-time websocket URL
// in this build is not derived from it. The setter exists to satisfy the
// services.STTService interface contract.
//
// Deprecated: no-op for AssemblyAI in this implementation.
func (s *STTService) SetModel(model string) {
	s.model = model
}

// UpdateSettings applies a runtime settings update to the STT service.
// Recognized keys: "domain" (the AssemblyAI language_model URL parameter,
// e.g. "medical-v1"). Unknown keys are ignored.
//
// Note: AssemblyAI's real-time websocket URL is derived from sample_rate,
// auth token, and language_model only. Language and model are not part
// of the URL or session config in this build, so we do not accept them
// here. Domain changes trigger a reconnect on the next audio frame.
func (s *STTService) UpdateSettings(settings map[string]interface{}) error {
	changed := false
	for k, v := range settings {
		strVal, _ := v.(string)
		switch k {
		case "domain":
			if strVal != s.domain {
				s.domain = strVal // empty allowed: clears domain
				changed = true
			}
		default:
			s.log.Debug("UpdateSettings: ignoring unknown key %q", k)
		}
	}
	if changed {
		if err := s.Cleanup(); err != nil {
			s.log.Warn("UpdateSettings: cleanup before reconnect failed: %v", err)
		}
	}
	return nil
}

func (s *STTService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	// Build WebSocket URL with auth token and sample rate
	wsURL := fmt.Sprintf("%s?sample_rate=%d&token=%s", s.baseURL, s.sampleRate, s.apiKey)
	if s.domain != "" {
		wsURL += "&language_model=" + url.QueryEscape(s.domain)
	}

	// Connect to AssemblyAI
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to AssemblyAI: %w", err)
	}

	// Send session configuration
	cfg := sessionConfig{
		EndUtteranceSilenceThreshold: s.endUtteranceSilenceThreshold,
	}
	if err = conn.WriteJSON(cfg); err != nil {
		conn.Close()
		return fmt.Errorf("failed to send session config to AssemblyAI: %w", err)
	}

	// Publish the new conn under connMu so a concurrent reader (lazy-init
	// check, audio writer, interruption) sees a coherent value.
	s.connMu.Lock()
	s.conn = conn
	s.connMu.Unlock()

	// Start receiving transcriptions
	s.connDropped.Store(false)
	s.readWG.Add(1)
	go s.receiveTranscriptions(conn)

	s.log.Info("Connected and initialized (model=%s, sample_rate=%d, silence_threshold=%dms)",
		s.model, s.sampleRate, s.endUtteranceSilenceThreshold)
	return nil
}

func (s *STTService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	s.connDropped.Store(true)
	s.disconnect()
	return nil
}

func (s *STTService) disconnect() {
	s.connMu.Lock()
	conn := s.conn
	s.conn = nil
	if conn != nil {
		_ = conn.WriteJSON(terminateMessage{TerminateSession: true})
		conn.Close()
	}
	s.connMu.Unlock()

	s.readWG.Wait()
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Pass StartFrame through without initializing (lazy initialization on first audio)
	if _, ok := frame.(*frames.StartFrame); ok {
		return s.PushFrame(frame, direction)
	}

	// Honor STTUpdateSettingsFrame.
	if updateFrame, ok := frame.(*frames.STTUpdateSettingsFrame); ok {
		if updateFrame.Service == "" || updateFrame.Service == s.Name() {
			if err := s.UpdateSettings(updateFrame.Settings); err != nil {
				s.log.Warn("UpdateSettings failed: %v", err)
			} else {
				s.log.Info("Applied runtime settings: %v", updateFrame.Settings)
			}
		}
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

	// Handle InterruptionFrame - send force end utterance to reset stream.
	// Capture conn under connMu and write through the captured value so a
	// concurrent disconnect cannot nil it underneath us.
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.log.Info("Received InterruptionFrame, sending force end utterance")
		s.connMu.Lock()
		conn := s.conn
		var writeErr error
		if conn != nil {
			forceEnd := map[string]bool{"force_end_utterance": true}
			writeErr = conn.WriteJSON(forceEnd)
		}
		s.connMu.Unlock()

		if conn != nil {
			if writeErr != nil {
				s.log.Debug("Error sending force end utterance: %v", writeErr)
			} else {
				s.log.Debug("Sent force end utterance to reset STT stream")
			}
		}
		return s.PushFrame(frame, direction)
	}

	// Process audio frames
	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		// Lazy initialization on first audio frame.
		// Read s.conn under connMu so a concurrent Cleanup (e.g. from
		// UpdateSettings reconnect) cannot race with this check.
		s.connMu.Lock()
		needInit := s.conn == nil
		s.connMu.Unlock()
		if needInit {
			s.log.Info("Lazy initializing on first AudioFrame")
			if err := s.Initialize(ctx); err != nil {
				s.log.Error("Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}

		if s.connDropped.Load() {
			return s.PushFrame(frame, direction)
		}

		// Send audio data to AssemblyAI as binary message (with mutex protection)
		s.connMu.Lock()
		conn := s.conn
		if conn == nil {
			s.connMu.Unlock()
			return s.PushFrame(frame, direction)
		}
		err := conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
		s.connMu.Unlock()

		if err != nil {
			s.log.Warn("Error sending audio: %v", err)
			s.connDropped.Store(true)
			s.disconnect()
			return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
		}

		// Pass AudioFrame downstream for audio-based interruption detection
		// LLMUserAggregator needs AudioFrames to analyze user speech patterns
		return s.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

func (s *STTService) receiveTranscriptions(conn *websocket.Conn) {
	defer s.readWG.Done()

	for {
		select {
		case <-s.ctx.Done():
			s.log.Info("Context cancelled, stopping transcription receiver")
			return
		default:
			_, message, err := conn.ReadMessage()
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
