package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
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
	apiKey             string
	voiceID            string
	model              string
	cartesiaVersion    string
	language           string
	sampleRate         int
	encoding           string
	container          string
	generationConfig   *GenerationConfig
	aggregateSentences bool
	conn               *websocket.Conn
	ctx                context.Context
	cancel             context.CancelFunc
	codecDetected      bool   // Track if we've auto-detected codec from StartFrame
	contextID          string // Cartesia context ID for streaming

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
	mu         sync.Mutex // Protect concurrent access to isSpeaking
}

// TTSConfig holds configuration for Cartesia TTS
type TTSConfig struct {
	APIKey             string
	VoiceID            string            // e.g., "a0e99841-438c-4a64-b679-ae501e7d6091" (Barbershop Man)
	Model              string            // e.g., "sonic-3", "sonic-2024-10-19"
	CartesiaVersion    string            // e.g., "2025-04-16"
	Language           string            // e.g., "en"
	SampleRate         int               // e.g., 8000, 16000, 22050, 24000, 44100
	Encoding           string            // e.g., "pcm_s16le", "pcm_mulaw", "pcm_alaw"
	Container          string            // e.g., "raw"
	GenerationConfig   *GenerationConfig // Optional: volume, speed, emotion for Sonic-3
	AggregateSentences bool              // Wait for complete sentences before TTS (default: true)
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
		apiKey:             config.APIKey,
		voiceID:            config.VoiceID,
		model:              model,
		cartesiaVersion:    cartesiaVersion,
		language:           language,
		sampleRate:         sampleRate,
		encoding:           encoding,
		container:          container,
		generationConfig:   config.GenerationConfig,
		aggregateSentences: aggregateSentences,
		codecDetected:      codecDetected,
		audioContexts:      make(map[string]*AudioContext),
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
	s.contextID = uuid.New().String()

	// Build WebSocket URL with API key and version
	wsURL := fmt.Sprintf("wss://api.cartesia.ai/tts/websocket?api_key=%s&cartesia_version=%s",
		s.apiKey, s.cartesiaVersion)

	var err error
	s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to connect to Cartesia: %w", err)
	}

	// Start receiving audio
	go s.receiveAudio()

	log.Printf("[CartesiaTTS] Streaming mode connected (context: %s)", s.contextID)

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
		s.conn.Close()
		s.conn = nil
	}

	// Clear audio contexts
	s.contextMu.Lock()
	s.audioContexts = make(map[string]*AudioContext)
	s.contextMu.Unlock()

	return nil
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

	// Handle LLMFullResponseStartFrame - just pass through
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		log.Printf("[CartesiaTTS] LLM response starting (context: %s)", s.contextID)
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
		log.Printf("[CartesiaTTS] INTERRUPTION RECEIVED - Stopping TTS synthesis")
		s.mu.Lock()
		wasSpeaking := s.isSpeaking
		oldContextID := s.contextID
		if s.isSpeaking {
			s.isSpeaking = false
		}
		// Clear text buffer on interruption
		s.textBuffer.Reset()
		// Reset metrics
		s.ttfbRecorded = false
		// Reset context ID to ensure new one is generated
		s.contextID = ""
		s.mu.Unlock()

		// CRITICAL: Always cancel the context if it exists, regardless of wasSpeaking
		// This prevents context accumulation (same pattern as ElevenLabs fix)
		if s.conn != nil && oldContextID != "" {
			log.Printf("[CartesiaTTS]   Canceling context %s on Cartesia (was_speaking=%v)", oldContextID, wasSpeaking)
			cancelMsg := map[string]interface{}{
				"context_id": oldContextID,
				"cancel":     true,
			}
			if err := s.conn.WriteJSON(cancelMsg); err != nil {
				log.Printf("[CartesiaTTS]   Error canceling context: %v", err)
			}

			// Remove old audio context
			s.removeAudioContext(oldContextID)
		}

		if wasSpeaking {
			log.Printf("[CartesiaTTS]   Emitting TTSStoppedFrame upstream to notify aggregators")
			s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
		}

		log.Printf("[CartesiaTTS] Interruption handled (was_speaking=%v, canceled_context=%s)", wasSpeaking, oldContextID)

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
		// Flush any remaining text in buffer
		if s.textBuffer.Len() > 0 {
			remainingText := s.textBuffer.String()
			s.textBuffer.Reset()
			log.Printf("[CartesiaTTS] Flushing remaining text: %s", remainingText)
			if err := s.synthesizeText(remainingText); err != nil {
				log.Printf("[CartesiaTTS] Error synthesizing remaining text: %v", err)
			}
		}

		if s.conn != nil && s.contextID != "" {
			log.Printf("[CartesiaTTS] LLM response ended, sending final flush to generate remaining audio")
			// Send final message with continue=false to signal end of transcript
			flushMsg := s.buildMessage("", false)
			if err := s.conn.WriteJSON(flushMsg); err != nil {
				log.Printf("[CartesiaTTS] Error sending flush: %v", err)
			}

			// CRITICAL: Reset context ID after flush for proper lifecycle
			oldContextID := s.contextID
			s.mu.Lock()
			wasSpeaking := s.isSpeaking
			s.isSpeaking = false
			s.contextID = "" // Reset context ID - new one will be generated on next synthesis
			s.ttfbRecorded = false
			s.mu.Unlock()

			if wasSpeaking {
				log.Printf("[CartesiaTTS] Synthesis completed, context %s ended", oldContextID)
			}
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

	// Sentence aggregation mode
	s.textBuffer.WriteString(text)
	bufferedText := s.textBuffer.String()

	// Extract complete sentences
	sentences, remainder := s.extractSentences(bufferedText)

	// Update buffer with remainder
	s.textBuffer.Reset()
	s.textBuffer.WriteString(remainder)

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

	// Generate new context ID if needed (after flush or first use)
	if s.contextID == "" {
		s.mu.Lock()
		s.contextID = uuid.New().String()
		log.Printf("[CartesiaTTS] Generated new context ID: %s", s.contextID)
		s.mu.Unlock()
	}

	// Emit TTSStartedFrame ONCE (boolean flag pattern)
	s.mu.Lock()
	firstToken := !s.isSpeaking
	if firstToken {
		s.isSpeaking = true
		// Start TTFB timer
		s.ttfbStart = time.Now()
		s.ttfbRecorded = false
		s.mu.Unlock()

		log.Printf("[CartesiaTTS] Emitting TTSStartedFrame (first text chunk)")
		// Push UPSTREAM so UserAggregator can track bot speaking state
		s.PushFrame(frames.NewTTSStartedFrame(), frames.Upstream)
		// Push DOWNSTREAM so WebSocketOutput can reset llmResponseEnded flag
		s.PushFrame(frames.NewTTSStartedFrame(), frames.Downstream)

		// Create audio context for this synthesis
		s.createAudioContext(s.contextID)
	} else {
		s.mu.Unlock()
	}

	// Log first token latency for monitoring parallel processing performance
	if firstToken {
		log.Printf("[CartesiaTTS] FIRST TOKEN -> Starting audio generation (parallel LLM+TTS)")
	}

	if s.conn != nil {
		// Send text chunk via WebSocket
		msg := s.buildMessage(text, true)
		return s.conn.WriteJSON(msg)
	}

	return fmt.Errorf("WebSocket connection not established")
}

func (s *TTSService) buildMessage(text string, continueTranscript bool) map[string]interface{} {
	voiceConfig := map[string]interface{}{
		"mode": "id",
		"id":   s.voiceID,
	}

	msg := map[string]interface{}{
		"transcript": text,
		"continue":   continueTranscript,
		"context_id": s.contextID,
		"model_id":   s.model,
		"voice":      voiceConfig,
		"output_format": map[string]interface{}{
			"container":   s.container,
			"encoding":    s.encoding,
			"sample_rate": s.sampleRate,
		},
		"language":               s.language,
		"add_timestamps":         true,
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
	for {
		select {
		case <-s.ctx.Done():
			log.Printf("[CartesiaTTS] Context cancelled, stopping audio receiver")
			return
		default:
			if s.conn == nil {
				log.Printf("[CartesiaTTS] Connection is nil, attempting reconnect")
				if err := s.reconnect(); err != nil {
					log.Printf("[CartesiaTTS] Reconnection failed: %v", err)
					time.Sleep(1 * time.Second) // Back off before retry
					continue
				}
			}

			_, message, err := s.conn.ReadMessage()
			if err != nil {
				// Check if this is a normal closure during shutdown
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					log.Printf("[CartesiaTTS] Connection closed normally")
					return
				}

				// Cartesia times out after 5 minutes of inactivity - attempt reconnect
				log.Printf("[CartesiaTTS] Connection error (timeout?): %v, attempting reconnect", err)
				if reconnectErr := s.reconnect(); reconnectErr != nil {
					log.Printf("[CartesiaTTS] Reconnection failed: %v", reconnectErr)
					s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
					return
				}
				log.Printf("[CartesiaTTS] Reconnected successfully after timeout")
				continue
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
				s.mu.Lock()
				currentCtxID := s.contextID
				s.mu.Unlock()

				if receivedCtxID != currentCtxID && !s.audioContextAvailable(receivedCtxID) {
					log.Printf("[CartesiaTTS] IGNORING audio from OLD context: %s (current: %s)", receivedCtxID, currentCtxID)
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
							log.Printf("[CartesiaTTS] Received %d word timestamps", len(timestamps))
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

// reconnect attempts to re-establish the WebSocket connection
func (s *TTSService) reconnect() error {
	// Close existing connection if any
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}

	// Build WebSocket URL with API key and version
	wsURL := fmt.Sprintf("wss://api.cartesia.ai/tts/websocket?api_key=%s&cartesia_version=%s",
		s.apiKey, s.cartesiaVersion)

	var err error
	s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("failed to reconnect to Cartesia: %w", err)
	}

	log.Printf("[CartesiaTTS] WebSocket reconnected successfully")
	return nil
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
