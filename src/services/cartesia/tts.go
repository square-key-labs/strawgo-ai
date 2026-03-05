package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
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

// AudioContext tracks audio playback state for a context
type AudioContext struct {
	ID              string
	AudioFrames     []*frames.TTSAudioFrame
	WordTimestamps  []WordTimestamp
	TotalAudioBytes int
	StartTime       time.Time
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

	// Sentence aggregation
	textBuffer strings.Builder

	// Audio context management
	audioContexts map[string]*AudioContext
	contextMu     sync.RWMutex

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

	// Rate-limiting for "IGNORING old context" logs
	ignoredAudioCount    int    // Count of ignored audio messages for current old context
	lastIgnoredContextID string // The context ID we're currently ignoring
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
		pronunciationDictID: config.PronunciationDictID,
		audioContexts:       make(map[string]*AudioContext),
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

	log.Printf("[CartesiaTTS] Streaming mode connected (context: %s)", s.GetActiveAudioContextID())

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

	// Clear audio contexts
	s.contextMu.Lock()
	s.audioContexts = make(map[string]*AudioContext)
	s.contextMu.Unlock()

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
					log.Printf("[CartesiaTTS] Detected incoming codec: %s", codec)
					// Match Cartesia output to incoming codec for compatibility
					switch codec {
					case "mulaw":
						s.sampleRate = 8000
						s.encoding = "pcm_mulaw"
						log.Printf("[CartesiaTTS] Auto-configured output format: pcm_mulaw @ 8000Hz")
					case "alaw":
						s.sampleRate = 8000
						s.encoding = "pcm_alaw"
						log.Printf("[CartesiaTTS] Auto-configured output format: pcm_alaw @ 8000Hz")
					case "linear16":
						s.sampleRate = 16000
						s.encoding = "pcm_s16le"
						log.Printf("[CartesiaTTS] Auto-configured output format: pcm_s16le @ 16000Hz")
					}
					s.codecDetected = true
				}
			}
		}

		// Eager initialization for parallel LLM+TTS processing
		if s.ctx == nil {
			log.Printf("[CartesiaTTS] Eager initializing WebSocket for parallel LLM+TTS processing")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[CartesiaTTS] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
			log.Printf("[CartesiaTTS] WebSocket ready - zero latency on first token!")
		}

		return s.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseStartFrame - generate context ID for this turn
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		turnCtxID := s.GetOrCreateTurnContextID()
		log.Printf("[CartesiaTTS] LLM response starting, generated turn context ID: %s", turnCtxID)
		return s.PushFrame(frame, direction)
	}

	// Handle EndFrame - cleanup and close connection
	if _, ok := frame.(*frames.EndFrame); ok {
		log.Printf("[CartesiaTTS] Received EndFrame, cleaning up")
		if err := s.Cleanup(); err != nil {
			log.Printf("[CartesiaTTS] Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - stop synthesis and reset state
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		log.Printf("[CartesiaTTS] ════════════════════════════════════════════")
		log.Printf("[CartesiaTTS] 🔴 INTERRUPTION RECEIVED")
		log.Printf("[CartesiaTTS] ════════════════════════════════════════════")

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
		// Log final summary of ignored audio messages if any
		if s.ignoredAudioCount > 0 {
			log.Printf("[CartesiaTTS] (ignored %d total audio messages from old context %s)", s.ignoredAudioCount, s.lastIgnoredContextID)
			s.ignoredAudioCount = 0
			s.lastIgnoredContextID = ""
		}
		s.mu.Unlock()
		// Reset context IDs via AudioContextManager
		s.ResetActiveAudioContext()

		log.Printf("[CartesiaTTS]   ├─ Step 1: State reset (wasSpeaking=%v, oldContext=%s, textBuffer=%d bytes)", wasSpeaking, oldContextID, textBufferLen)

		// CRITICAL: Clear ALL audio contexts to prevent stale audio from leaking through
		// This is necessary because contextID may have been cleared by LLMFullResponseEndFrame
		// before the interruption, but audio contexts still exist and allow old audio through
		s.contextMu.Lock()
		allContextIDs := make([]string, 0, len(s.audioContexts))
		for ctxID := range s.audioContexts {
			allContextIDs = append(allContextIDs, ctxID)
		}
		s.audioContexts = make(map[string]*AudioContext) // Clear all contexts
		s.contextMu.Unlock()

		log.Printf("[CartesiaTTS]   ├─ Step 2: Found %d active audio contexts to cancel", len(allContextIDs))

		// Send cancel messages for all active contexts (best-effort, no reconnect)
		connected := s.isConnected()
		if connected && len(allContextIDs) > 0 {
			for _, ctxID := range allContextIDs {
				log.Printf("[CartesiaTTS]   │   └─ Canceling context %s on Cartesia API", ctxID)
				cancelMsg := map[string]interface{}{
					"context_id": ctxID,
					"cancel":     true,
				}
				if err := s.writeJSONBestEffort(cancelMsg); err != nil {
					log.Printf("[CartesiaTTS]   │   └─ ⚠️ Error canceling context %s: %v", ctxID, err)
				}
			}
			log.Printf("[CartesiaTTS]   ├─ Step 3: Sent cancel to Cartesia for %d contexts", len(allContextIDs))
		} else if connected && oldContextID != "" {
			// Fallback: cancel the current context if no audio contexts exist
			log.Printf("[CartesiaTTS]   │   └─ Canceling current context %s on Cartesia API", oldContextID)
			cancelMsg := map[string]interface{}{
				"context_id": oldContextID,
				"cancel":     true,
			}
			if err := s.writeJSONBestEffort(cancelMsg); err != nil {
				log.Printf("[CartesiaTTS]   │   └─ ⚠️ Error canceling context: %v", err)
			}
			log.Printf("[CartesiaTTS]   ├─ Step 3: Sent cancel to Cartesia for current context")
		} else {
			log.Printf("[CartesiaTTS]   ├─ Step 3: No contexts to cancel (connected=%v)", connected)
		}

		if wasSpeaking {
			log.Printf("[CartesiaTTS]   ├─ Step 4: Emitting TTSStoppedFrame upstream")
			s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
		}

		log.Printf("[CartesiaTTS]   └─ Interruption complete, passing frame downstream")

		return s.PushFrame(frame, direction)
	}

	// Process text frames (LLM output)
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		// Lazy initialization on first text frame
		if s.ctx == nil {
			log.Printf("[CartesiaTTS] Lazy initializing on first TextFrame")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[CartesiaTTS] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		return s.processTextInput(textFrame.Text)
	}

	if llmFrame, ok := frame.(*frames.LLMTextFrame); ok {
		// Lazy initialization on first text frame
		if s.ctx == nil {
			log.Printf("[CartesiaTTS] Lazy initializing on first LLMTextFrame")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[CartesiaTTS] Failed to initialize: %v", err)
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
			log.Printf("[CartesiaTTS] Flushing remaining text: %s", remainingText)
			if err := s.synthesizeText(remainingText); err != nil {
				log.Printf("[CartesiaTTS] Error synthesizing remaining text: %v", err)
			}
		}

		currentContextID := s.GetActiveAudioContextID()
		hasValidContext := s.isConnected() && currentContextID != ""

		if hasValidContext {
			log.Printf("[CartesiaTTS] LLM response ended, sending final flush to generate remaining audio")
			// Send final message with continue=false to signal end of transcript
			flushMsg := s.buildMessageWithContextID("", false, currentContextID)
			if err := s.writeJSON(flushMsg); err != nil {
				log.Printf("[CartesiaTTS] Error sending flush: %v", err)
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

		log.Printf("[CartesiaTTS] Closing context %s on normal completion (was_speaking=%v)", currentContextID, wasSpeaking)
		cancelMsg := map[string]interface{}{
			"context_id": currentContextID,
			"cancel":     true,
		}
		if err := s.writeJSONBestEffort(cancelMsg); err != nil {
			log.Printf("[CartesiaTTS] Error closing context: %v", err)
		}

		if wasSpeaking {
			log.Printf("[CartesiaTTS] Synthesis completed, context %s closed", currentContextID)
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
			log.Printf("[CartesiaTTS] Synthesizing sentence: %s", sentence)
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

		log.Printf("[CartesiaTTS] Emitting TTSStartedFrame (first text chunk) with context ID: %s", ctxID)
		// Push UPSTREAM so UserAggregator can track bot speaking state
		s.PushFrame(frames.NewTTSStartedFrameWithContext(ctxID), frames.Upstream)
		// Push DOWNSTREAM so WebSocketOutput can reset llmResponseEnded flag and set expected context
		s.PushFrame(frames.NewTTSStartedFrameWithContext(ctxID), frames.Downstream)

		// Create audio context for this synthesis
		s.createAudioContext(ctxID)
	} else {
		s.mu.Unlock()
	}

	// Log first token latency for monitoring parallel processing performance
	if firstToken {
		log.Printf("[CartesiaTTS] FIRST TOKEN -> Starting audio generation (parallel LLM+TTS)")
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
		log.Printf("[CartesiaTTS] Connection nil on write, reconnecting...")
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
		log.Printf("[CartesiaTTS] Write failed (ErrCloseSent), reconnecting...")
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

// Audio Context Management

func (s *TTSService) createAudioContext(contextID string) {
	s.contextMu.Lock()
	defer s.contextMu.Unlock()

	s.audioContexts[contextID] = &AudioContext{
		ID:             contextID,
		AudioFrames:    make([]*frames.TTSAudioFrame, 0),
		WordTimestamps: make([]WordTimestamp, 0),
		StartTime:      time.Now(),
	}
	log.Printf("[CartesiaTTS] Created audio context: %s", contextID)
}

func (s *TTSService) removeAudioContext(contextID string) {
	s.contextMu.Lock()
	defer s.contextMu.Unlock()

	delete(s.audioContexts, contextID)
	log.Printf("[CartesiaTTS] Removed audio context: %s", contextID)
}

func (s *TTSService) audioContextAvailable(contextID string) bool {
	s.contextMu.RLock()
	defer s.contextMu.RUnlock()

	_, exists := s.audioContexts[contextID]
	return exists
}

func (s *TTSService) appendToAudioContext(contextID string, audioFrame *frames.TTSAudioFrame) {
	s.contextMu.Lock()
	defer s.contextMu.Unlock()

	if ctx, exists := s.audioContexts[contextID]; exists {
		ctx.AudioFrames = append(ctx.AudioFrames, audioFrame)
		ctx.TotalAudioBytes += len(audioFrame.Data)
	}
}

func (s *TTSService) addWordTimestamps(contextID string, timestamps []WordTimestamp) {
	s.contextMu.Lock()
	defer s.contextMu.Unlock()

	if ctx, exists := s.audioContexts[contextID]; exists {
		ctx.WordTimestamps = append(ctx.WordTimestamps, timestamps...)

		// Push text frames aligned with word timestamps
		for _, ts := range timestamps {
			// Create TextFrame for this word
			textFrame := frames.NewTextFrame(ts.Word + " ")
			// Set metadata with timing info
			textFrame.SetMetadata("word_start_time", ts.StartTime)
			textFrame.SetMetadata("context_id", contextID)
			s.PushFrame(textFrame, frames.Upstream)
		}
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
			log.Printf("[CartesiaTTS] Context cancelled, stopping audio receiver")
			return
		default:
			_, message, err := myConn.ReadMessage()
			if err != nil {
				// Intentional shutdown — Cleanup() called cancel + closed conn
				if s.ctx.Err() != nil {
					log.Printf("[CartesiaTTS] Connection closed (shutdown)")
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
					log.Printf("[CartesiaTTS] Server closed connection (idle timeout?), was_speaking=%v, marking for write-path reconnect", speaking)
				} else {
					log.Printf("[CartesiaTTS] Connection error: %v, was_speaking=%v, marking for write-path reconnect", err, speaking)
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
				log.Printf("[CartesiaTTS] Error parsing response: %v", err)
				continue
			}

			// Get message type
			msgType, ok := response["type"].(string)
			if !ok {
				log.Printf("[CartesiaTTS] Unknown message format: %v", response)
				continue
			}

			// Get context ID from response
			receivedCtxID, hasCtxID := response["context_id"].(string)

			// Validate context ID to avoid processing old/stale messages
			if hasCtxID {
				currentCtxID := s.GetActiveAudioContextID()

				if receivedCtxID != currentCtxID && !s.audioContextAvailable(receivedCtxID) {
					// Rate-limit "IGNORING old context" logs to reduce noise
					s.mu.Lock()
					if s.lastIgnoredContextID != receivedCtxID {
						// New old context being ignored - log first occurrence and reset counter
						if s.ignoredAudioCount > 0 {
							log.Printf("[CartesiaTTS] (ignored %d total audio messages from old context %s)", s.ignoredAudioCount, s.lastIgnoredContextID)
						}
						s.lastIgnoredContextID = receivedCtxID
						s.ignoredAudioCount = 1
						log.Printf("[CartesiaTTS] IGNORING audio from OLD context: %s (current: %s)", receivedCtxID, currentCtxID)
					} else {
						s.ignoredAudioCount++
						// Only log every 20th occurrence to reduce spam
						if s.ignoredAudioCount%20 == 0 {
							log.Printf("[CartesiaTTS] ... still ignoring audio from old context %s (%d messages so far)", receivedCtxID, s.ignoredAudioCount)
						}
					}
					s.mu.Unlock()
					continue // CRITICAL: Skip this message, it's from an interrupted context
				}
			}

			switch msgType {
			case "chunk":
				// Record TTFB on first audio chunk
				s.mu.Lock()
				if !s.ttfbRecorded && !s.ttfbStart.IsZero() {
					ttfb := time.Since(s.ttfbStart)
					s.ttfbRecorded = true
					log.Printf("[CartesiaTTS] TTFB (Time to First Byte): %v", ttfb)
				}
				s.mu.Unlock()

				// Audio chunk - decode base64 audio
				if audioB64, ok := response["data"].(string); ok && audioB64 != "" {
					audioData, err := base64.StdEncoding.DecodeString(audioB64)
					if err != nil {
						log.Printf("[CartesiaTTS] Error decoding base64 audio: %v", err)
						continue
					}

					codec := s.encodingToCodec()
					audioFrame := frames.NewTTSAudioFrame(audioData, s.sampleRate, 1)
					audioFrame.SetMetadata("codec", codec)
					audioFrame.SetMetadata("context_id", receivedCtxID)

					// Add to audio context for tracking
					if hasCtxID {
						s.appendToAudioContext(receivedCtxID, audioFrame)
					}

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
							logger.Debug("[CartesiaTTS] Received %d word timestamps", len(timestamps))
							s.addWordTimestamps(receivedCtxID, timestamps)
						}
					}
				}

			case "done":
				// Context completed
				log.Printf("[CartesiaTTS] Received done message for context: %s", receivedCtxID)

				// Get audio context stats before removing
				s.contextMu.RLock()
				if ctx, exists := s.audioContexts[receivedCtxID]; exists {
					duration := time.Since(ctx.StartTime)
					log.Printf("[CartesiaTTS] Context %s completed: %d audio frames, %d bytes, %d words, duration: %v",
						receivedCtxID, len(ctx.AudioFrames), ctx.TotalAudioBytes, len(ctx.WordTimestamps), duration)
				}
				s.contextMu.RUnlock()

				// Remove audio context
				s.removeAudioContext(receivedCtxID)

				s.mu.Lock()
				if s.isSpeaking {
					s.isSpeaking = false
					log.Printf("[CartesiaTTS] Synthesis completed (WebSocketOutput will emit TTSStoppedFrame after playback)")
				}
				s.mu.Unlock()

			case "error":
				// Error message
				errorMsg := ""
				if errStr, ok := response["error"].(string); ok {
					errorMsg = errStr
				}
				log.Printf("[CartesiaTTS] Error from Cartesia: %s", errorMsg)
				s.PushFrame(frames.NewErrorFrame(fmt.Errorf("Cartesia error: %s", errorMsg)), frames.Upstream)

			default:
				log.Printf("[CartesiaTTS] Unknown message type: %s", msgType)
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

	log.Printf("[CartesiaTTS] WebSocket reconnected (gen %d)", s.connGen)
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
