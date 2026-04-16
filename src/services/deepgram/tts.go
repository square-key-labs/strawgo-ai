package deepgram

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
	"github.com/square-key-labs/strawgo-ai/src/services"
)

const (
	// Deepgram TTS WebSocket endpoint
	DeepgramTTSURL = "wss://api.deepgram.com/v1/speak"

	// Default model (aura family)
	DefaultTTSModel = "aura-asteria-en"

	// Default encoding
	DefaultTTSEncoding = "linear16"

	// Default sample rate
	DefaultTTSSampleRate = 16000
)

// TTSService provides text-to-speech using Deepgram
//
// Context Management:
// ===================
// - Cancel context on InterruptionFrame (ALWAYS, regardless of speaking state)
// - Generate new context_id for each synthesis request
// - Track context_id on all TTS frames for transport layer filtering
//
// Key pattern: Always cancel context on interruption to prevent stale audio
// This prevents old audio from overlapping with new responses.
type TTSService struct {
	*processors.BaseProcessor
	apiKey     string
	model      string
	encoding   string
	sampleRate int

	// WebSocket connection
	conn   *websocket.Conn
	ctx    context.Context
	cancel context.CancelFunc

	// Context management
	contextID            string // Current TTS context ID for tracking
	currentTurnContextID string // Context ID for current LLM turn (reused across multiple TTS invocations)

	// Speaking state tracking
	isSpeaking bool       // Track if we've emitted TTSStartedFrame
	mu         sync.Mutex // Protect concurrent access to isSpeaking and contextID

	// WebSocket write mutex - CRITICAL for thread safety
	// gorilla/websocket is NOT safe for concurrent writes
	wsMu sync.Mutex // Protect concurrent WebSocket writes

	// Metrics tracking
	ttfbStart    time.Time
	ttfbRecorded bool
	log          *logger.Logger
}

// TTSConfig holds configuration for Deepgram TTS
type TTSConfig struct {
	APIKey     string
	Model      string // e.g., "aura-asteria-en", "aura-luna-en", "aura-stella-en"
	Encoding   string // e.g., "linear16", "mulaw", "alaw" (default: "linear16")
	SampleRate int    // e.g., 8000, 16000, 24000, 48000 (default: 16000)
}

// NewTTSService creates a new Deepgram TTS service
func NewTTSService(config TTSConfig) *TTSService {
	// Set defaults
	model := config.Model
	if model == "" {
		model = DefaultTTSModel
	}

	encoding := config.Encoding
	if encoding == "" {
		encoding = DefaultTTSEncoding
	}

	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultTTSSampleRate
	}

	ds := &TTSService{
		apiKey:     config.APIKey,
		model:      model,
		encoding:   encoding,
		sampleRate: sampleRate,
		log:        logger.WithPrefix("DeepgramTTS"),
	}
	ds.BaseProcessor = processors.NewBaseProcessor("DeepgramTTS", ds)
	return ds
}

func (s *TTSService) SetVoice(voiceID string) {
	// Deepgram uses model names instead of voice IDs
	s.model = voiceID
}

func (s *TTSService) SetModel(model string) {
	s.model = model
}

func (s *TTSService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	// Build WebSocket URL with query parameters
	u, err := url.Parse(DeepgramTTSURL)
	if err != nil {
		return fmt.Errorf("failed to parse URL: %w", err)
	}

	q := u.Query()
	q.Set("model", s.model)
	q.Set("encoding", s.encoding)
	q.Set("sample_rate", fmt.Sprintf("%d", s.sampleRate))
	u.RawQuery = q.Encode()

	// Set authorization header
	headers := make(map[string][]string)
	headers["Authorization"] = []string{"Token " + s.apiKey}

	// Connect to Deepgram
	s.conn, _, err = websocket.DefaultDialer.Dial(u.String(), headers)
	if err != nil {
		return fmt.Errorf("failed to connect to Deepgram: %w", err)
	}

	// Start receiving audio responses
	go s.receiveAudio()

	s.log.Info("Connected and initialized (model: %s, encoding: %s, sample_rate: %d)",
		s.model, s.encoding, s.sampleRate)
	return nil
}

func (s *TTSService) Cleanup() error {
	// Cancel context first to signal goroutines to stop
	if s.cancel != nil {
		s.cancel()
	}

	// Give goroutines a moment to see the context cancellation
	time.Sleep(50 * time.Millisecond)

	// Now close the connection
	if s.conn != nil {
		// Send close message (JSON with type: "Close")
		closeMsg := map[string]interface{}{
			"type": "Close",
		}
		s.writeJSON(closeMsg)

		s.conn.Close()
		s.conn = nil
	}

	return nil
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - eager initialization for parallel LLM+TTS processing
	if _, ok := frame.(*frames.StartFrame); ok {
		// Eager initialization for parallel LLM+TTS processing
		if s.ctx == nil {
			s.log.Info("Eager initializing WebSocket for parallel LLM+TTS processing")
			if err := s.Initialize(ctx); err != nil {
				s.log.Error("Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
			s.log.Info("WebSocket ready - zero latency on first token!")
		}

		return s.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseStartFrame - generate context ID for this turn
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		s.mu.Lock()
		s.currentTurnContextID = services.GenerateContextID()
		s.log.Info("LLM response starting, generated turn context ID: %s", s.currentTurnContextID)
		s.mu.Unlock()
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

	// Handle InterruptionFrame - stop synthesis and reset state
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.log.Info("============================================")
		s.log.Info("INTERRUPTION RECEIVED")
		s.log.Info("============================================")

		s.mu.Lock()
		wasSpeaking := s.isSpeaking
		oldContextID := s.contextID
		if s.isSpeaking {
			s.isSpeaking = false
		}
		// Reset metrics
		s.ttfbRecorded = false
		// Reset context IDs
		s.contextID = ""
		s.currentTurnContextID = ""

		s.log.Debug("Step 1: state reset (wasSpeaking=%v, oldContext=%s)", wasSpeaking, oldContextID)

		// Send flush message to Deepgram to clear any pending audio
		if s.conn != nil {
			s.log.Debug("Step 2: sending Flush message to Deepgram")
			flushMsg := map[string]interface{}{
				"type": "Flush",
			}
			if err := s.writeJSON(flushMsg); err != nil {
				s.log.Warn("Error sending flush: %v", err)
			}
		}

		if wasSpeaking {
			s.log.Debug("Step 3: emitting TTSStoppedFrame upstream")
			s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
		}

		s.log.Debug("Interruption complete, passing frame downstream")

		return s.PushFrame(frame, direction)
	}

	// Process text frames (LLM output)
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		if textFrame.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		// Lazy initialization on first text frame
		if s.ctx == nil {
			s.log.Info("Lazy initializing on first TextFrame")
			if err := s.Initialize(ctx); err != nil {
				s.log.Error("Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		return s.synthesizeText(textFrame.Text)
	}

	if llmFrame, ok := frame.(*frames.LLMTextFrame); ok {
		if llmFrame.SkipTTS {
			return s.PushFrame(frame, direction)
		}
		// Lazy initialization on first text frame
		if s.ctx == nil {
			s.log.Info("Lazy initializing on first LLMTextFrame")
			if err := s.Initialize(ctx); err != nil {
				s.log.Error("Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		return s.synthesizeText(llmFrame.Text)
	}

	// Handle LLM response end to flush TTS
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		// Lock to safely read contextID
		s.mu.Lock()
		currentContextID := s.contextID
		wasSpeaking := s.isSpeaking
		s.isSpeaking = false
		s.contextID = ""            // Reset context ID - new one will be generated on next synthesis
		s.currentTurnContextID = "" // Reset turn context ID
		s.ttfbRecorded = false
		s.mu.Unlock()
		s.log.Info("LLM response ended, sending flush to generate final audio")
		// Send flush message to tell Deepgram to finish processing
		flushMsg := map[string]interface{}{
			"type": "Flush",
		}
		if err := s.writeJSON(flushMsg); err != nil {
			s.log.Warn("Error sending flush: %v", err)
		}

		// CRITICAL: Close context after normal completion (not just on interruption)
		// This prevents context accumulation on Deepgram
		s.log.Info("Closing context %s on normal completion (was_speaking=%v)", currentContextID, wasSpeaking)
		closeMsg := map[string]interface{}{
			"type": "Close",
		}
		if err := s.writeJSON(closeMsg); err != nil {
			s.log.Debug("Error closing context: %v", err)
		}

		if wasSpeaking {
			s.log.Info("Synthesis completed, context %s closed", currentContextID)
		}

		return s.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

// synthesizeText generates TTS from text
func (s *TTSService) synthesizeText(text string) error {
	if text == "" {
		return nil
	}

	// Use current turn context ID if available, otherwise generate new one
	s.mu.Lock()
	if s.contextID == "" {
		if s.currentTurnContextID != "" {
			// Reuse context ID from current LLM turn
			s.contextID = s.currentTurnContextID
			s.log.Debug("Reusing turn context ID: %s", s.contextID)
		} else {
			// Generate new context ID if no turn context available (shouldn't happen in normal flow)
			s.contextID = services.GenerateContextID()
			s.log.Debug("Generated new context ID: %s", s.contextID)
		}
	}
	s.mu.Unlock()

	// Emit TTSStartedFrame ONCE (boolean flag pattern)
	s.mu.Lock()
	firstToken := !s.isSpeaking
	contextID := s.contextID
	if firstToken {
		s.isSpeaking = true
		// Start TTFB timer
		s.ttfbStart = time.Now()
		s.ttfbRecorded = false
		s.mu.Unlock()

		s.log.Info("Emitting TTSStartedFrame (first text chunk) with context ID: %s", contextID)
		// Push UPSTREAM so UserAggregator can track bot speaking state
		s.PushFrame(frames.NewTTSStartedFrameWithContext(contextID), frames.Upstream)
		// Push DOWNSTREAM so WebSocketOutput can reset llmResponseEnded flag and set expected context
		s.PushFrame(frames.NewTTSStartedFrameWithContext(contextID), frames.Downstream)
	} else {
		s.mu.Unlock()
	}

	// Log first token latency for monitoring parallel processing performance
	if firstToken {
		s.log.Info("FIRST TOKEN -> Starting audio generation (parallel LLM+TTS)")
	}

	// Send text to Deepgram via WebSocket
	msg := map[string]interface{}{
		"type": "Speak",
		"text": text,
	}

	return s.writeJSON(msg)
}

// writeJSON safely writes JSON to the WebSocket connection with mutex protection
// gorilla/websocket is NOT thread-safe for concurrent writes
func (s *TTSService) writeJSON(v interface{}) error {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	if s.conn == nil {
		return fmt.Errorf("WebSocket connection not established")
	}
	return s.conn.WriteJSON(v)
}

// receiveAudio reads audio from WebSocket
func (s *TTSService) receiveAudio() {
	for {
		select {
		case <-s.ctx.Done():
			s.log.Debug("Context cancelled, stopping audio receiver")
			return
		default:
			if s.conn == nil {
				s.log.Debug("Connection is nil, stopping receiver")
				return
			}

			messageType, message, err := s.conn.ReadMessage()
			if err != nil {
				// Check if this is a normal closure during shutdown
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					s.log.Debug("Connection closed normally")
					return
				}

				s.log.Error("Connection error: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			if messageType == websocket.BinaryMessage {
				// Record TTFB on first audio chunk
				s.mu.Lock()
				if !s.ttfbRecorded && !s.ttfbStart.IsZero() {
					ttfb := time.Since(s.ttfbStart)
					s.ttfbRecorded = true
					s.log.Info("TTFB (Time to First Byte): %v", ttfb)
				}
				contextID := s.contextID
				s.mu.Unlock()

				// Binary audio data
				codec := s.encodingToCodec()
				audioFrame := frames.NewTTSAudioFrame(message, s.sampleRate, 1)
				audioFrame.SetMetadata("codec", codec)
				audioFrame.SetMetadata("context_id", contextID)

				s.PushFrame(audioFrame, frames.Downstream)
			} else if messageType == websocket.TextMessage {
				// Metadata (e.g., completion signal, errors)
				var metadata map[string]interface{}
				if err := json.Unmarshal(message, &metadata); err == nil {
					msgType, ok := metadata["type"].(string)
					if !ok {
						continue
					}

					switch msgType {
					case "Flushed":
						// Flush completed - synthesis is done
						s.log.Info("Received Flushed message - synthesis complete")

						s.mu.Lock()
						if s.isSpeaking {
							s.isSpeaking = false
							s.log.Info("Synthesis completed (WebSocketOutput will emit TTSStoppedFrame after playback)")
						}
						s.mu.Unlock()

					case "Metadata":
						// Metadata about the request (can be ignored or logged)
						s.log.Debug("Received metadata: %v", metadata)

					case "Error":
						// Error message
						errorMsg := "Unknown error"
						if errStr, ok := metadata["error"].(string); ok {
							errorMsg = errStr
						}
						s.log.Error("Error from Deepgram: %s", errorMsg)
						s.PushFrame(frames.NewErrorFrame(fmt.Errorf("Deepgram error: %s", errorMsg)), frames.Upstream)

					default:
						s.log.Warn("Unknown message type: %s", msgType)
					}
				}
			}
		}
	}
}

// encodingToCodec converts Deepgram encoding to internal codec name
func (s *TTSService) encodingToCodec() string {
	switch s.encoding {
	case "mulaw":
		return "mulaw"
	case "alaw":
		return "alaw"
	case "linear16":
		return "linear16"
	default:
		return "linear16"
	}
}
