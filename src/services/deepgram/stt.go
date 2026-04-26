package deepgram

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// STTService provides speech-to-text using Deepgram
type STTService struct {
	*processors.BaseProcessor
	apiKey            string
	language          string
	model             string
	encoding          string
	keepaliveInterval time.Duration
	keepaliveTimeout  time.Duration
	conn              *websocket.Conn
	ctx               context.Context
	cancel            context.CancelFunc

	// lifecycleMu serializes Initialize / Cleanup / UpdateSettings against
	// each other and against concurrent lazy-init triggered from HandleFrame.
	// It also protects ctx, cancel, and the settings fields (language,
	// model, encoding) from racing with Initialize's reads of those values.
	lifecycleMu sync.Mutex

	connMu      sync.Mutex // Protects concurrent WebSocket writes.
	readWG      sync.WaitGroup
	connDropped atomic.Bool // set on write failure; frames silently dropped until reconnect
	log         *logger.Logger
}

// STTConfig holds configuration for Deepgram
type STTConfig struct {
	APIKey            string
	Language          string        // e.g., "en-US"
	Model             string        // e.g., "nova-2"
	Encoding          string        // Supported: "mulaw"/"ulaw", "alaw", "linear16" (default: "linear16")
	KeepaliveInterval time.Duration // Interval for sending keepalive pings (default: 5s)
	KeepaliveTimeout  time.Duration // Timeout for keepalive (default: 30s)
}

// NewSTTService creates a new Deepgram STT service
func NewSTTService(config STTConfig) *STTService {
	encoding := config.Encoding
	if encoding == "" {
		encoding = "linear16" // Default to PCM
	}

	// Normalize codec names for Deepgram API
	encoding = normalizeDeepgramEncoding(encoding)

	// Set keepalive defaults
	keepaliveInterval := config.KeepaliveInterval
	if keepaliveInterval == 0 {
		keepaliveInterval = 5 * time.Second
	}
	keepaliveTimeout := config.KeepaliveTimeout
	if keepaliveTimeout == 0 {
		keepaliveTimeout = 30 * time.Second
	}

	ds := &STTService{
		apiKey:            config.APIKey,
		language:          config.Language,
		model:             config.Model,
		encoding:          encoding,
		keepaliveInterval: keepaliveInterval,
		keepaliveTimeout:  keepaliveTimeout,
		log:               logger.WithPrefix("DeepgramSTT"),
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

// UpdateSettings applies a runtime settings update to the STT service.
// Recognized keys: "language", "model", "encoding". Unknown keys are
// ignored with a debug log. If any recognized key changes, the existing
// websocket connection is closed; the next audio frame triggers a lazy
// re-init using the new values. (The Deepgram Live API does not accept
// language/model changes mid-stream; reconnect is the documented way.)
func (s *STTService) UpdateSettings(settings map[string]interface{}) error {
	s.lifecycleMu.Lock()
	changed := false
	for k, v := range settings {
		strVal, _ := v.(string)
		switch k {
		case "language":
			if strVal != "" && strVal != s.language {
				s.language = strVal
				changed = true
			}
		case "model":
			if strVal != "" && strVal != s.model {
				s.model = strVal
				changed = true
			}
		case "encoding":
			if strVal != "" {
				normalized := normalizeDeepgramEncoding(strVal)
				if normalized != s.encoding {
					s.encoding = normalized
					changed = true
				}
			}
		default:
			s.log.Debug("UpdateSettings: ignoring unknown key %q", k)
		}
	}
	s.lifecycleMu.Unlock()

	if changed {
		// Force lazy re-init on next audio frame with the new settings.
		if err := s.Cleanup(); err != nil {
			s.log.Warn("UpdateSettings: cleanup before reconnect failed: %v", err)
		}
	}
	return nil
}

func (s *STTService) Initialize(ctx context.Context) error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()

	// Skip if already initialized -- protects against two concurrent
	// lazy-init paths from HandleFrame both trying to dial.
	s.connMu.Lock()
	already := s.conn != nil
	s.connMu.Unlock()
	if already {
		return nil
	}

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

	// Hold connMu across Dial+publish so a concurrent Cleanup blocks until
	// s.conn is published and then closes the dialed conn. Without this,
	// Cleanup firing between Dial returning and the publish below would
	// observe s.conn==nil, skip the close, and leak the dialed conn.
	s.connMu.Lock()
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, header)
	if err != nil {
		s.connMu.Unlock()
		return fmt.Errorf("failed to connect to Deepgram: %w", err)
	}
	s.conn = conn
	s.connDropped.Store(false)
	// Register goroutines under connMu+lifecycleMu so a Cleanup that races
	// in cannot reach readWG.Wait() before they are accounted for.
	s.readWG.Add(2)
	s.connMu.Unlock()

	go s.receiveTranscriptions(conn)
	go s.keepaliveTask(conn)

	s.log.Info("Connected and initialized")
	return nil
}

func (s *STTService) Cleanup() error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()

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
	s.connMu.Unlock()

	if conn != nil {
		conn.Close()
	}

	s.readWG.Wait()
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Pass StartFrame through without initializing (lazy initialization on first audio)
	if _, ok := frame.(*frames.StartFrame); ok {
		// Emit STT metadata for auto-tuning turn detection
		s.PushFrame(frames.NewSTTMetadataFrame("deepgram", 300*time.Millisecond), frames.Downstream)
		return s.PushFrame(frame, direction)
	}

	// Honor STTUpdateSettingsFrame. If the frame targets a specific service
	// and we don't match, pass it through unchanged so other STT services
	// downstream still see it.
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

	// Handle InterruptionFrame - send finalize to reset Deepgram stream
	// This prevents old transcription fragments from arriving after interruption
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.log.Info("Received InterruptionFrame, sending finalize to reset stream")
		// Capture conn under connMu and write through the captured value,
		// not s.conn, so a concurrent disconnect cannot nil it underneath us.
		s.connMu.Lock()
		conn := s.conn
		var writeErr error
		if conn != nil {
			finalizeMsg := map[string]interface{}{
				"type": "Finalize",
			}
			// Bound the write so a stalled socket cannot wedge connMu and
			// starve audio writers, lazy-init readers, keepalives, or Cleanup.
			_ = conn.SetWriteDeadline(time.Now().Add(1 * time.Second))
			writeErr = conn.WriteJSON(finalizeMsg)
			_ = conn.SetWriteDeadline(time.Time{})
		}
		s.connMu.Unlock()

		if conn != nil {
			if writeErr != nil {
				s.log.Debug("Error sending finalize message: %v", writeErr)
			} else {
				s.log.Debug("Sent finalize message to reset STT stream")
			}
		}
		// Pass the interruption frame downstream
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

		// Drop frames silently while connection is down; prevents ~50/sec log flood.
		if s.connDropped.Load() {
			return s.PushFrame(frame, direction)
		}

		// Send audio data to Deepgram (with mutex protection)
		s.connMu.Lock()
		conn := s.conn
		if conn == nil {
			s.connMu.Unlock()
			return s.PushFrame(frame, direction)
		}
		err := conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
		s.connMu.Unlock()

		if err != nil {
			s.connDropped.Store(true)
			s.log.Warn("WebSocket write failed, disconnecting: %v", err)
			s.disconnect()
			return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
		}

		// IMPORTANT: Pass AudioFrame downstream for audio-based interruption detection
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
			s.log.Debug("Context cancelled, stopping transcription receiver")
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
				s.log.Error("Error reading message: %v", err)
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
				s.log.Error("Error parsing response: %v", err)
				continue
			}

			// Extract transcript
			if len(response.Channel.Alternatives) > 0 {
				transcript := response.Channel.Alternatives[0].Transcript
				if transcript != "" {
					transcriptionFrame := frames.NewTranscriptionFrame(transcript, response.IsFinal)
					s.log.Debug("Transcription (final=%v): %s", response.IsFinal, transcript)
					s.PushFrame(transcriptionFrame, frames.Downstream)
				}
			}
		}
	}
}

func (s *STTService) keepaliveTask(conn *websocket.Conn) {
	defer s.readWG.Done()

	ticker := time.NewTicker(s.keepaliveInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if s.connDropped.Load() {
				continue
			}

			// Send a JSON keepalive message (with mutex protection)
			keepalive := map[string]string{"type": "KeepAlive"}
			s.connMu.Lock()
			err := conn.WriteJSON(keepalive)
			s.connMu.Unlock()

			if err != nil {
				s.log.Warn("Error sending keepalive: %v", err)
				return
			}
		}
	}
}
