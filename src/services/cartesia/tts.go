package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// GenerationConfig holds Cartesia Sonic-3 generation parameters
type GenerationConfig struct {
	Volume  float64 `json:"volume,omitempty"`  // Volume multiplier [0.5, 2.0], default 1.0
	Speed   float64 `json:"speed,omitempty"`   // Speed multiplier [0.6, 1.5], default 1.0
	Emotion string  `json:"emotion,omitempty"` // Emotion guidance: neutral, angry, excited, etc.
}

// WordTimestamp represents a word with its playback timing
type WordTimestamp struct {
	Word      string
	StartTime float64 // Start time in seconds
}

// TTSService provides text-to-speech using Cartesia
//
// Context Management:
// ===================
// - Cancel context on InterruptionFrame (ALWAYS, regardless of speaking state)
// - Send "continue": false to finalize context (flush remaining audio)
// - Contexts auto-expire 5 seconds after last input (per Cartesia API)
//
// Key fix: Always cancel context on interruption, even if wasSpeaking=false
// This prevents context accumulation while maintaining efficiency.
//
// Stale-audio filtering: This service does NOT filter incoming audio by
// context_id. The pipeline-level AudioGuard processor is the authoritative
// stale-audio gate (see docs/adr/0001-audio-guard-as-dedicated-processor.md).
// TTS services here only mint context_ids, stamp outgoing TTSAudioFrames,
// and use them to call upstream cancel APIs.
type TTSService struct {
	*processors.BaseProcessor
	*services.AudioContextManager
	apiKey              string
	voiceID             string
	model               string
	cartesiaVersion     string
	language            string
	sampleRate          int
	encoding            string
	container           string
	generationConfig    *GenerationConfig
	aggregateSentences  bool
	pronunciationDictID string
	conn                *websocket.Conn
	ctx                 context.Context
	cancel              context.CancelFunc
	codecDetected       bool // Track if we've auto-detected codec from StartFrame
	log                 *logger.Logger

	// Sentence aggregation
	textBuffer strings.Builder

	// Metrics tracking
	ttfbStart    time.Time
	ttfbRecorded bool

	// Speaking state tracking
	isSpeaking bool       // Track if we've emitted TTSStartedFrame
	mu         sync.Mutex // Protect concurrent access to isSpeaking and service-specific state

	// WebSocket write mutex - CRITICAL for thread safety
	// gorilla/websocket is NOT safe for concurrent writes
	wsMu sync.Mutex // Protect concurrent WebSocket writes

	// Connection generation counter — incremented on each reconnect.
	// Used for debugging/logging only. Stale-goroutine protection is via pointer comparison in receiveAudio().
	connGen uint64

	dialFunc func() (*websocket.Conn, error)
}

// TTSConfig holds configuration for Cartesia TTS
type TTSConfig struct {
	APIKey              string
	VoiceID             string            // e.g., "a0e99841-438c-4a64-b679-ae501e7d6091" (Barbershop Man)
	Model               string            // e.g., "sonic-3", "sonic-2024-10-19"
	CartesiaVersion     string            // e.g., "2025-04-16"
	Language            string            // e.g., "en"
	SampleRate          int               // e.g., 8000, 16000, 22050, 24000, 44100
	Encoding            string            // e.g., "pcm_s16le", "pcm_mulaw", "pcm_alaw"
	Container           string            // e.g., "raw"
	GenerationConfig    *GenerationConfig // Optional: volume, speed, emotion for Sonic-3
	AggregateSentences  bool              // Wait for complete sentences before TTS (default: true)
	PronunciationDictID string            // Optional: UUID of a pre-created pronunciation dictionary (Sonic-3)
}

// NewTTSService creates a new Cartesia TTS service
func NewTTSService(config TTSConfig) *TTSService {
	// Set defaults
	model := config.Model
	if model == "" {
		model = "sonic-3"
	}

	cartesiaVersion := config.CartesiaVersion
	if cartesiaVersion == "" {
		cartesiaVersion = "2025-04-16"
	}

	language := config.Language
	if language == "" {
		language = "en"
	}

	sampleRate := config.SampleRate
	codecDetected := true
	if sampleRate == 0 {
		sampleRate = 24000 // Default PCM at 24kHz
		codecDetected = false
	}

	encoding := config.Encoding
	if encoding == "" {
		encoding = "pcm_s16le"
	}

	container := config.Container
	if container == "" {
		container = "raw"
	}

	// Default to true for sentence aggregation (better audio quality)
	aggregateSentences := true
	if !config.AggregateSentences && config.Model != "" {
		// Only disable if explicitly set and model was explicitly configured
		aggregateSentences = config.AggregateSentences
	}

	cs := &TTSService{
		apiKey:              config.APIKey,
		voiceID:             config.VoiceID,
		model:               model,
		cartesiaVersion:     cartesiaVersion,
		language:            language,
		sampleRate:          sampleRate,
		encoding:            encoding,
		container:           container,
		generationConfig:    config.GenerationConfig,
		aggregateSentences:  aggregateSentences,
		codecDetected:       codecDetected,
		log:                 logger.WithPrefix("CartesiaTTS"),
		pronunciationDictID: config.PronunciationDictID,
		AudioContextManager: services.NewAudioContextManager(),
	}
	cs.BaseProcessor = processors.NewBaseProcessor("CartesiaTTS", cs)
	return cs
}

func (s *TTSService) SetVoice(voiceID string) {
	s.voiceID = voiceID
}

func (s *TTSService) SetModel(model string) {
	s.model = model
}

func (s *TTSService) SetLanguage(language string) {
	s.language = language
}

func (s *TTSService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	// Generate context ID for streaming
	s.SetActiveAudioContextID(services.GenerateContextID())

	// Dial WebSocket outside any lock — network I/O can block
	conn, err := s.dialWebSocket()
	if err != nil {
		return err
	}

	// Install connection under lock
	s.wsMu.Lock()
	s.conn = conn
	s.connGen++
	s.wsMu.Unlock()

	// Start receiving audio
	go s.receiveAudio()

	s.log.Info("Streaming mode connected (context: %s)", s.GetActiveAudioContextID())

	return nil
}

func (s *TTSService) Cleanup() error {
	// Cancel context first to signal goroutines to stop
	if s.cancel != nil {
		s.cancel()
	}

	// Give goroutines a moment to see the context cancellation
	time.Sleep(50 * time.Millisecond)

	// Close the connection under lock (writeJSON may be in flight)
	s.wsMu.Lock()
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}
	s.wsMu.Unlock()

	return nil
}

// isConnected reports whether the WebSocket is currently established.
// Safe for concurrent use.
func (s *TTSService) isConnected() bool {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	return s.conn != nil
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - codec detection AND eager initialization
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		// Auto-detect output format from incoming codec (only if user didn't set SampleRate)
		if !s.codecDetected {
			if meta := startFrame.Metadata(); meta != nil {
				if codec, ok := meta["codec"].(string); ok {
					s.log.Info("Detected incoming codec: %s", codec)
					// Match Cartesia output to incoming codec for compatibility
					switch codec {
					case "mulaw":
						s.sampleRate = 8000
						s.encoding = "pcm_mulaw"
						s.log.Info("Auto-configured output format: pcm_mulaw @ 8000Hz")
					case "alaw":
						s.sampleRate = 8000
						s.encoding = "pcm_alaw"
						s.log.Info("Auto-configured output format: pcm_alaw @ 8000Hz")
					case "linear16":
						s.sampleRate = 16000
						s.encoding = "pcm_s16le"
						s.log.Info("Auto-configured output format: pcm_s16le @ 16000Hz")
					}
					s.codecDetected = true
				}
			}
		}

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
		turnCtxID := s.GetOrCreateTurnContextID()
		s.log.Info("LLM response starting, generated turn context ID: %s", turnCtxID)
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

		oldContextID := s.GetActiveAudioContextID()
		s.mu.Lock()
		wasSpeaking := s.isSpeaking
		textBufferLen := s.textBuffer.Len()
		if s.isSpeaking {
			s.isSpeaking = false
		}
		// Clear text buffer on interruption
		s.textBuffer.Reset()
		// Reset metrics
		s.ttfbRecorded = false
		s.mu.Unlock()
		// Reset context IDs via AudioContextManager
		s.ResetActiveAudioContext()

		s.log.Debug("Step 1: state reset (wasSpeaking=%v, oldContext=%s, textBuffer=%d bytes)", wasSpeaking, oldContextID, textBufferLen)

		// Send upstream cancel for the active context (best-effort, no reconnect).
		// This is API-specific cleanup against the Cartesia API; pipeline-level
		// stale-audio filtering is the AudioGuard processor's job.
		connected := s.isConnected()
		if connected && oldContextID != "" {
			s.log.Debug("Canceling current context %s on Cartesia API", oldContextID)
			cancelMsg := map[string]interface{}{
				"context_id": oldContextID,
				"cancel":     true,
			}
			if err := s.writeJSONBestEffort(cancelMsg); err != nil {
				s.log.Debug("Error canceling context: %v", err)
			}
			s.log.Debug("Step 2: sent cancel to Cartesia for current context")
		} else {
			s.log.Debug("Step 2: no context to cancel (connected=%v, oldContextID=%q)", connected, oldContextID)
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
		return s.processTextInput(textFrame.Text)
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
		return s.processTextInput(llmFrame.Text)
	}

	// Handle LLM response end to flush TTS
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		// Flush any remaining text in buffer (protected by mutex)
		s.mu.Lock()
		hasRemainingText := s.textBuffer.Len() > 0
		var remainingText string
		if hasRemainingText {
			remainingText = s.textBuffer.String()
			s.textBuffer.Reset()
		}
		s.mu.Unlock()

		if hasRemainingText {
			s.log.Debug("Flushing remaining text: %s", remainingText)
			if err := s.synthesizeText(remainingText); err != nil {
				s.log.Warn("Error synthesizing remaining text: %v", err)
			}
		}

		currentContextID := s.GetActiveAudioContextID()
		turnContextID := s.GetTurnContextID() // capture before ResetActiveAudioContext clears it
		logContextID := currentContextID
		if logContextID == "" {
			logContextID = turnContextID
		}
		hasValidContext := s.isConnected() && currentContextID != ""

		if hasValidContext {
			s.log.Info("LLM response ended, sending final flush to generate remaining audio")
			// Send final message with continue=false to signal end of transcript
			flushMsg := s.buildMessageWithContextID("", false, currentContextID)
			if err := s.writeJSON(flushMsg); err != nil {
				s.log.Warn("Error sending flush: %v", err)
			}
		}

		// CRITICAL: Close context after normal completion (not just on interruption)
		// This prevents context accumulation on Cartesia
		s.mu.Lock()
		wasSpeaking := s.isSpeaking
		s.isSpeaking = false
		s.ttfbRecorded = false
		s.mu.Unlock()
		s.ResetActiveAudioContext()

		s.log.Info("Closing context %s on normal completion (was_speaking=%v)", logContextID, wasSpeaking)
		if currentContextID != "" {
			cancelMsg := map[string]interface{}{
				"context_id": currentContextID,
				"cancel":     true,
			}
			if err := s.writeJSONBestEffort(cancelMsg); err != nil {
				s.log.Debug("Error closing context: %v", err)
			}
		}

		if wasSpeaking {
			s.log.Info("Synthesis completed, context %s closed", currentContextID)
		}
		if !wasSpeaking && logContextID != "" {
			s.log.Info("Context %s completed: 0 audio frames, 0 bytes, 0 words (zero-frame turn)", logContextID)
			s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
		}
		return s.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

// processTextInput handles incoming text with optional sentence aggregation
func (s *TTSService) processTextInput(text string) error {
	if text == "" {
		return nil
	}

	if !s.aggregateSentences {
		// No aggregation - send immediately
		return s.synthesizeText(text)
	}

	// Sentence aggregation mode - protect textBuffer with mutex
	s.mu.Lock()
	s.textBuffer.WriteString(text)
	bufferedText := s.textBuffer.String()
	s.mu.Unlock()

	// Extract complete sentences (doesn't need lock - working on local copy)
	sentences, remainder := s.extractSentences(bufferedText)

	// Update buffer with remainder (protected by mutex)
	s.mu.Lock()
	s.textBuffer.Reset()
	s.textBuffer.WriteString(remainder)
	s.mu.Unlock()

	// Synthesize complete sentences
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence != "" {
			s.log.Debug("Synthesizing sentence: %s", sentence)
			if err := s.synthesizeText(sentence); err != nil {
				return err
			}
		}
	}

	return nil
}

// extractSentences splits text into complete sentences and remainder
func (s *TTSService) extractSentences(text string) ([]string, string) {
	var sentences []string
	var currentSentence strings.Builder

	sentenceEnders := map[rune]bool{
		'.': true,
		'!': true,
		'?': true,
		';': true,
	}

	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		currentSentence.WriteRune(r)

		if sentenceEnders[r] {
			// Check if this is really end of sentence (not abbreviation like "Dr.")
			// Simple heuristic: if followed by space and uppercase or end of text
			if i == len(runes)-1 {
				// End of text
				sentences = append(sentences, currentSentence.String())
				currentSentence.Reset()
			} else if i+1 < len(runes) && unicode.IsSpace(runes[i+1]) {
				// Followed by space - likely end of sentence
				sentences = append(sentences, currentSentence.String())
				currentSentence.Reset()
			}
		}
	}

	// Return sentences and any remaining incomplete text
	return sentences, currentSentence.String()
}

func (s *TTSService) synthesizeText(text string) error {
	if text == "" {
		return nil
	}

	// Use AudioContextManager to get or create context ID
	// Reuses turn context ID if available, otherwise generates new one
	ctxID := s.GetOrCreateContextID()

	// Emit TTSStartedFrame ONCE (boolean flag pattern)
	s.mu.Lock()
	firstToken := !s.isSpeaking
	if firstToken {
		s.isSpeaking = true
		// Start TTFB timer
		s.ttfbStart = time.Now()
		s.ttfbRecorded = false
		s.mu.Unlock()

		s.log.Info("Emitting TTSStartedFrame (first text chunk) with context ID: %s", ctxID)
		// Push UPSTREAM so UserAggregator can track bot speaking state
		s.PushFrame(frames.NewTTSStartedFrameWithContext(ctxID), frames.Upstream)
		// Push DOWNSTREAM so WebSocketOutput can reset llmResponseEnded flag and set expected context
		s.PushFrame(frames.NewTTSStartedFrameWithContext(ctxID), frames.Downstream)
	} else {
		s.mu.Unlock()
	}

	// Log first token latency for monitoring parallel processing performance
	if firstToken {
		s.log.Info("FIRST TOKEN -> Starting audio generation (parallel LLM+TTS)")
	}

	// Send text chunk via WebSocket (writeJSON handles nil conn check)
	msg := s.buildMessageWithContextID(text, true, ctxID)
	return s.writeJSON(msg)
}

// writeJSON safely writes JSON to the WebSocket with mutex protection.
// If the connection is dead (nil or ErrCloseSent from Cartesia idle timeout),
// it reconnects, starts a new reader goroutine, and retries the write once.
// For fire-and-forget messages (cancel, cleanup), use writeJSONBestEffort instead.
func (s *TTSService) writeJSON(v interface{}) error {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()

	// Reconnect if connection was marked dead by receiveAudio()
	if s.conn == nil {
		if s.ctx != nil && s.ctx.Err() != nil {
			return fmt.Errorf("WebSocket connection closed (shutting down)")
		}
		s.log.Warn("Connection nil on write, reconnecting...")
		if err := s.reconnectLocked(); err != nil {
			return fmt.Errorf("WebSocket reconnection failed: %w", err)
		}
	}

	s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	err := s.conn.WriteJSON(v)
	if err == nil {
		return nil
	}

	// Connection dead (gorilla auto-echoed close frame → ErrCloseSent permanently).
	// Reconnect and retry the write once.
	if errors.Is(err, websocket.ErrCloseSent) {
		s.log.Warn("Write failed (ErrCloseSent), reconnecting...")
		if reconnErr := s.reconnectLocked(); reconnErr != nil {
			return fmt.Errorf("write failed and reconnection failed: %w", reconnErr)
		}
		s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
		return s.conn.WriteJSON(v)
	}

	return err
}

// writeJSONBestEffort writes JSON without reconnecting on failure.
// Used for cancel/cleanup messages where reconnection would be wasteful.
func (s *TTSService) writeJSONBestEffort(v interface{}) error {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	if s.conn == nil {
		return fmt.Errorf("WebSocket connection not established")
	}
	s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	return s.conn.WriteJSON(v)
}

func (s *TTSService) buildMessage(text string, continueTranscript bool) map[string]interface{} {
	ctxID := s.GetActiveAudioContextID()
	return s.buildMessageWithContextID(text, continueTranscript, ctxID)
}

// buildMessageWithContextID builds a Cartesia message with an explicit context ID
// Use this when you've already captured the contextID under lock
func (s *TTSService) buildMessageWithContextID(text string, continueTranscript bool, contextID string) map[string]interface{} {
	voiceConfig := map[string]interface{}{
		"mode": "id",
		"id":   s.voiceID,
	}

	msg := map[string]interface{}{
		"transcript": text,
		"continue":   continueTranscript,
		"context_id": contextID,
		"model_id":   s.model,
		"voice":      voiceConfig,
		"output_format": map[string]interface{}{
			"container":   s.container,
			"encoding":    s.encoding,
			"sample_rate": s.sampleRate,
		},
		"language":                s.language,
		"add_timestamps":          true,
		"use_original_timestamps": true, // Use original timestamps for non-sonic models
	}

	// Add generation config if provided (Sonic-3 features: volume, speed, emotion)
	if s.generationConfig != nil {
		genConfig := map[string]interface{}{}
		if s.generationConfig.Volume != 0 {
			genConfig["volume"] = s.generationConfig.Volume
		}
		if s.generationConfig.Speed != 0 {
			genConfig["speed"] = s.generationConfig.Speed
		}
		if s.generationConfig.Emotion != "" {
			genConfig["emotion"] = s.generationConfig.Emotion
		}
		if len(genConfig) > 0 {
			msg["generation_config"] = genConfig
		}
	}

	// Add pronunciation dictionary if configured (Sonic-3 feature)
	if s.pronunciationDictID != "" {
		msg["pronunciation_dict_id"] = s.pronunciationDictID
	}

	return msg
}

// pushWordTimestamps emits aligned text frames for each word timestamp.
// Stale-audio filtering for these is handled at the pipeline level by AudioGuard.
func (s *TTSService) pushWordTimestamps(contextID string, timestamps []WordTimestamp) {
	for _, ts := range timestamps {
		textFrame := frames.NewTextFrame(ts.Word + " ")
		textFrame.SetMetadata("word_start_time", ts.StartTime)
		textFrame.SetMetadata("context_id", contextID)
		s.PushFrame(textFrame, frames.Upstream)
	}
}

func (s *TTSService) receiveAudio() {
	// Capture our connection pointer under lock. If writeJSON() reconnects
	// and swaps s.conn while we're reading, we detect it via pointer comparison
	// and exit cleanly without nulling the newer connection.
	s.wsMu.Lock()
	myConn := s.conn
	s.wsMu.Unlock()

	if myConn == nil {
		return
	}

	for {
		select {
		case <-s.ctx.Done():
			s.log.Debug("Context cancelled, stopping audio receiver")
			return
		default:
			_, message, err := myConn.ReadMessage()
			if err != nil {
				// Intentional shutdown — Cleanup() called cancel + closed conn
				if s.ctx.Err() != nil {
					s.log.Debug("Connection closed (shutdown)")
					return
				}

				// Log with speaking state for ops observability.
				// Mid-synthesis loss = audio truncation (pipeline's turn-stop handles the gap).
				// Idle loss = transparent (writeJSON reconnects on next call).
				s.mu.Lock()
				speaking := s.isSpeaking
				s.mu.Unlock()

				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					s.log.Debug("Server closed connection (idle timeout?), was_speaking=%v, marking for write-path reconnect", speaking)
				} else {
					s.log.Warn("Connection error: %v, was_speaking=%v, marking for write-path reconnect", err, speaking)
				}

				// Mark connection dead so writeJSON() reconnects on next call.
				// Only nil out if this is still our connection (prevents nulling
				// a newer connection already established by writeJSON's reconnect).
				s.wsMu.Lock()
				if s.conn == myConn {
					s.conn = nil
				}
				s.wsMu.Unlock()
				return
			}

			// Parse JSON message
			var response map[string]interface{}
			if err := json.Unmarshal(message, &response); err != nil {
				s.log.Error("Error parsing response: %v", err)
				continue
			}

			// Get message type
			msgType, ok := response["type"].(string)
			if !ok {
				s.log.Warn("Unknown message format: %v", response)
				continue
			}

			// Get context ID from response. We stamp it on outgoing frames; the
			// pipeline-level AudioGuard processor handles stale-audio filtering
			// (see docs/adr/0001-audio-guard-as-dedicated-processor.md).
			receivedCtxID, hasCtxID := response["context_id"].(string)

			switch msgType {
			case "chunk":
				// Record TTFB on first audio chunk
				s.mu.Lock()
				if !s.ttfbRecorded && !s.ttfbStart.IsZero() {
					ttfb := time.Since(s.ttfbStart)
					s.ttfbRecorded = true
					s.log.Info("TTFB (Time to First Byte): %v", ttfb)
				}
				s.mu.Unlock()

				// Audio chunk - decode base64 audio
				if audioB64, ok := response["data"].(string); ok && audioB64 != "" {
					audioData, err := base64.StdEncoding.DecodeString(audioB64)
					if err != nil {
						s.log.Error("Error decoding base64 audio: %v", err)
						continue
					}

					codec := s.encodingToCodec()
					audioFrame := frames.NewTTSAudioFrame(audioData, s.sampleRate, 1)
					audioFrame.SetMetadata("codec", codec)
					audioFrame.SetMetadata("context_id", receivedCtxID)

					s.PushFrame(audioFrame, frames.Downstream)
				}

			case "timestamps":
				// Word timestamps - aligned text output
				if wordTimestamps, ok := response["word_timestamps"].(map[string]interface{}); ok {
					words, wordsOK := wordTimestamps["words"].([]interface{})
					starts, startsOK := wordTimestamps["start"].([]interface{})

					if wordsOK && startsOK && len(words) == len(starts) {
						timestamps := make([]WordTimestamp, 0, len(words))
						for i := 0; i < len(words); i++ {
							word, wordOK := words[i].(string)
							start, startOK := starts[i].(float64)
							if wordOK && startOK {
								timestamps = append(timestamps, WordTimestamp{
									Word:      word,
									StartTime: start,
								})
							}
						}

						if hasCtxID && len(timestamps) > 0 {
							s.log.Debug("Received %d word timestamps", len(timestamps))
							s.pushWordTimestamps(receivedCtxID, timestamps)
						}
					}
				}

			case "done":
				// Context completed
				s.log.Info("Received done message for context: %s", receivedCtxID)

				s.mu.Lock()
				if s.isSpeaking {
					s.isSpeaking = false
					s.log.Info("Synthesis completed (WebSocketOutput will emit TTSStoppedFrame after playback)")
				}
				s.mu.Unlock()

			case "error":
				// Error message
				errorMsg := ""
				if errStr, ok := response["error"].(string); ok {
					errorMsg = errStr
				}
				s.log.Error("Error from Cartesia: %s", errorMsg)
				s.PushFrame(frames.NewErrorFrame(fmt.Errorf("Cartesia error: %s", errorMsg)), frames.Upstream)

			default:
				s.log.Warn("Unknown message type: %s", msgType)
			}
		}
	}
}

// dialWebSocket creates a new WebSocket connection to Cartesia.
// Does NOT hold any locks — safe to call from any goroutine.
func (s *TTSService) dialWebSocket() (*websocket.Conn, error) {
	if s.dialFunc != nil {
		return s.dialFunc()
	}

	wsURL := fmt.Sprintf("wss://api.cartesia.ai/tts/websocket?api_key=%s&cartesia_version=%s",
		s.apiKey, s.cartesiaVersion)

	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Cartesia: %w", err)
	}
	return conn, nil
}

// reconnectLocked closes the current connection and establishes a new one.
// Caller MUST hold wsMu. Temporarily releases wsMu during network dial to
// avoid blocking writers. Starts a new receiveAudio() goroutine on success.
func (s *TTSService) reconnectLocked() error {
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}

	// Release lock during dial — network I/O can block
	s.wsMu.Unlock()
	newConn, err := s.dialWebSocket()
	s.wsMu.Lock()

	if err != nil {
		return err
	}

	// Shutdown occurred while we were dialing — discard the new connection
	if s.ctx != nil && s.ctx.Err() != nil {
		newConn.Close()
		return fmt.Errorf("shutting down, discarding new connection")
	}

	// Another goroutine may have reconnected while we were dialing
	if s.conn != nil {
		newConn.Close()
		return nil
	}

	s.conn = newConn
	s.connGen++
	go s.receiveAudio()

	s.log.Info("WebSocket reconnected (gen %d)", s.connGen)
	return nil
}

// reconnect is the public thread-safe method for re-establishing the connection.
func (s *TTSService) reconnect() error {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	return s.reconnectLocked()
}

// encodingToCodec converts Cartesia encoding to internal codec name
func (s *TTSService) encodingToCodec() string {
	switch s.encoding {
	case "pcm_mulaw":
		return "mulaw"
	case "pcm_alaw":
		return "alaw"
	case "pcm_s16le":
		return "linear16"
	case "pcm_f32le":
		return "float32"
	default:
		return "linear16"
	}
}
