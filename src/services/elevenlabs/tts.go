package elevenlabs

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// VoiceSettings holds configurable voice parameters
type VoiceSettings struct {
	Stability       float64 `json:"stability,omitempty"`        // 0.0 to 1.0
	SimilarityBoost float64 `json:"similarity_boost,omitempty"` // 0.0 to 1.0
	Style           float64 `json:"style,omitempty"`            // 0.0 to 1.0
	UseSpeakerBoost bool    `json:"use_speaker_boost,omitempty"`
	Speed           float64 `json:"speed,omitempty"` // 0.7 to 1.2 for WebSocket, 0.25 to 4.0 for HTTP
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

// TTSService provides text-to-speech using ElevenLabs
//
// Context Management:
// ===================
// - Close context on InterruptionFrame (ALWAYS, regardless of speaking state)
// - Let contexts persist across response end (don't close on flush)
// - Only reset contextID on interruption
//
// Key fix: Always close context on interruption, even if wasSpeaking=false
// This prevents context accumulation while maintaining efficiency.
type TTSService struct {
	*processors.BaseProcessor
	apiKey             string
	voiceID            string
	model              string
	outputFormat       string
	useStreaming       bool
	voiceSettings      *VoiceSettings
	language           string // Language code for multilingual models
	aggregateSentences bool
	conn               *websocket.Conn
	ctx                context.Context
	cancel             context.CancelFunc
	codecDetected      bool   // Track if we've auto-detected codec from StartFrame
	contextID          string // ElevenLabs context ID for multi-stream mode

	// Sentence aggregation
	textBuffer strings.Builder

	// Word timestamp tracking
	cumulativeTime       float64 // Track cumulative audio time
	partialWord          string  // Partial word across chunks
	partialWordStartTime float64

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

// TTSConfig holds configuration for ElevenLabs
type TTSConfig struct {
	APIKey             string
	VoiceID            string         // e.g., "21m00Tcm4TlvDq8ikWAM" (Rachel)
	Model              string         // e.g., "eleven_turbo_v2_5", "eleven_flash_v2_5"
	OutputFormat       string         // Supported: "ulaw_8000", "alaw_8000", "pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100" (default: "pcm_24000")
	UseStreaming       bool           // Use WebSocket streaming for lower latency
	VoiceSettings      *VoiceSettings // Optional: stability, similarity_boost, style, speed
	Language           string         // Language code for multilingual models (e.g., "en", "es", "fr")
	AggregateSentences bool           // Wait for complete sentences before TTS (default: true)
}

// Multilingual models that support language codes
var multilingualModels = map[string]bool{
	"eleven_flash_v2_5": true,
	"eleven_turbo_v2_5": true,
}

// NewTTSService creates a new ElevenLabs TTS service
func NewTTSService(config TTSConfig) *TTSService {
	outputFormat := config.OutputFormat
	codecDetected := true // Assume user explicitly set format

	if outputFormat == "" {
		outputFormat = "pcm_24000" // Default PCM at 24kHz
		codecDetected = false      // Will auto-detect from StartFrame
	}

	// Default voice settings if not provided
	voiceSettings := config.VoiceSettings
	if voiceSettings == nil {
		voiceSettings = &VoiceSettings{
			Stability:       0.5,
			SimilarityBoost: 0.75,
		}
	}

	// Default to true for sentence aggregation (better audio quality)
	aggregateSentences := true
	if !config.AggregateSentences && config.VoiceID != "" {
		// Only disable if explicitly set and voice was explicitly configured
		aggregateSentences = config.AggregateSentences
	}

	es := &TTSService{
		apiKey:             config.APIKey,
		voiceID:            config.VoiceID,
		model:              config.Model,
		outputFormat:       outputFormat,
		useStreaming:       config.UseStreaming,
		voiceSettings:      voiceSettings,
		language:           config.Language,
		aggregateSentences: aggregateSentences,
		codecDetected:      codecDetected,
		audioContexts:      make(map[string]*AudioContext),
	}
	es.BaseProcessor = processors.NewBaseProcessor("ElevenLabsTTS", es)
	return es
}

func (s *TTSService) SetVoice(voiceID string) {
	s.voiceID = voiceID
}

func (s *TTSService) SetModel(model string) {
	s.model = model
}

func (s *TTSService) SetVoiceSettings(settings *VoiceSettings) {
	s.voiceSettings = settings
}

func (s *TTSService) SetLanguage(language string) {
	s.language = language
}

func (s *TTSService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	if s.useStreaming {
		// Generate context ID for multi-stream mode
		s.contextID = uuid.New().String()

		// Build WebSocket URL with multi-stream-input endpoint and output_format
		wsURL := fmt.Sprintf("wss://api.elevenlabs.io/v1/text-to-speech/%s/multi-stream-input?model_id=%s&output_format=%s&auto_mode=true",
			s.voiceID, s.model, s.outputFormat)

		// Add language code for multilingual models
		if s.language != "" && multilingualModels[s.model] {
			wsURL += fmt.Sprintf("&language_code=%s", s.language)
			log.Printf("[ElevenLabsTTS] Using language code: %s", s.language)
		}

		header := http.Header{}
		header.Set("xi-api-key", s.apiKey)

		var err error
		s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, header)
		if err != nil {
			return fmt.Errorf("failed to connect to ElevenLabs: %w", err)
		}

		// Send initial config with context_id and voice settings
		config := map[string]interface{}{
			"text":       " ",
			"context_id": s.contextID,
		}

		// Add voice settings
		if s.voiceSettings != nil {
			voiceSettingsMap := map[string]interface{}{}
			if s.voiceSettings.Stability != 0 {
				voiceSettingsMap["stability"] = s.voiceSettings.Stability
			}
			if s.voiceSettings.SimilarityBoost != 0 {
				voiceSettingsMap["similarity_boost"] = s.voiceSettings.SimilarityBoost
			}
			if s.voiceSettings.Style != 0 {
				voiceSettingsMap["style"] = s.voiceSettings.Style
			}
			if s.voiceSettings.UseSpeakerBoost {
				voiceSettingsMap["use_speaker_boost"] = s.voiceSettings.UseSpeakerBoost
			}
			if s.voiceSettings.Speed != 0 {
				voiceSettingsMap["speed"] = s.voiceSettings.Speed
			}
			if len(voiceSettingsMap) > 0 {
				config["voice_settings"] = voiceSettingsMap
			}
		}

		if err := s.conn.WriteJSON(config); err != nil {
			return fmt.Errorf("failed to send config: %w", err)
		}

		// Start receiving audio
		go s.receiveAudio()

		// Start keepalive to prevent timeout
		go s.keepaliveLoop()

		log.Printf("[ElevenLabsTTS] Streaming mode connected (context: %s)", s.contextID)
	} else {
		log.Printf("[ElevenLabsTTS] Non-streaming mode initialized")
	}

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
		// Send close message before closing socket (for ElevenLabs)
		if s.contextID != "" {
			closeMsg := map[string]interface{}{
				"close_socket": true,
			}
			s.conn.WriteJSON(closeMsg)
		}
		s.conn.Close()
		s.conn = nil
	}

	// Clear audio contexts
	s.contextMu.Lock()
	s.audioContexts = make(map[string]*AudioContext)
	s.contextMu.Unlock()

	return nil
}

func (s *TTSService) keepaliveLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if s.conn != nil && s.contextID != "" {
				keepaliveMsg := map[string]interface{}{
					"text":       "",
					"context_id": s.contextID,
				}
				if err := s.conn.WriteJSON(keepaliveMsg); err != nil {
					log.Printf("[ElevenLabsTTS] Keepalive error: %v", err)
					return
				}
			}
		}
	}
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - codec detection AND eager initialization
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		// Auto-detect output format from incoming codec (only if user didn't set OutputFormat)
		if !s.codecDetected {
			if meta := startFrame.Metadata(); meta != nil {
				if codec, ok := meta["codec"].(string); ok {
					log.Printf("[ElevenLabsTTS] Detected incoming codec: %s", codec)
					// Match ElevenLabs output to incoming codec for compatibility
					switch codec {
					case "mulaw":
						s.outputFormat = "ulaw_8000"
						log.Printf("[ElevenLabsTTS] Auto-configured output format: ulaw_8000")
					case "alaw":
						s.outputFormat = "alaw_8000"
						log.Printf("[ElevenLabsTTS] Auto-configured output format: alaw_8000")
					case "linear16":
						s.outputFormat = "pcm_16000"
						log.Printf("[ElevenLabsTTS] Auto-configured output format: pcm_16000")
					}
					s.codecDetected = true
				}
			}
		}

		// Eager initialization for parallel LLM+TTS processing
		if s.useStreaming && s.ctx == nil {
			log.Printf("[ElevenLabsTTS] Eager initializing WebSocket for parallel LLM+TTS processing")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[ElevenLabsTTS] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
			log.Printf("[ElevenLabsTTS] WebSocket ready - zero latency on first token!")
		}

		return s.PushFrame(frame, direction)
	}

	// Handle LLMFullResponseStartFrame - just pass through
	if _, ok := frame.(*frames.LLMFullResponseStartFrame); ok {
		log.Printf("[ElevenLabsTTS] LLM response starting (context: %s)", s.contextID)
		return s.PushFrame(frame, direction)
	}

	// Handle EndFrame - cleanup and close connection
	if _, ok := frame.(*frames.EndFrame); ok {
		log.Printf("[ElevenLabsTTS] Received EndFrame, cleaning up")
		if err := s.Cleanup(); err != nil {
			log.Printf("[ElevenLabsTTS] Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - stop synthesis and reset state
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		log.Printf("[ElevenLabsTTS] INTERRUPTION RECEIVED - Stopping TTS synthesis")
		s.mu.Lock()
		wasSpeaking := s.isSpeaking
		oldContextID := s.contextID
		if s.isSpeaking {
			s.isSpeaking = false
		}
		// Clear text buffer and word tracking on interruption
		s.textBuffer.Reset()
		s.partialWord = ""
		s.partialWordStartTime = 0.0
		s.cumulativeTime = 0
		s.ttfbRecorded = false
		// Reset context ID to ensure new one is generated
		s.contextID = ""
		s.mu.Unlock()

		// CRITICAL: Always close the context if it exists, regardless of wasSpeaking
		// This prevents context accumulation on ElevenLabs
		if s.useStreaming && s.conn != nil && oldContextID != "" {
			log.Printf("[ElevenLabsTTS]   Closing context %s on ElevenLabs (was_speaking=%v)", oldContextID, wasSpeaking)
			closeMsg := map[string]interface{}{
				"context_id":    oldContextID,
				"close_context": true,
			}
			if err := s.conn.WriteJSON(closeMsg); err != nil {
				log.Printf("[ElevenLabsTTS]   Error closing context: %v", err)
			}

			// Remove old audio context
			s.removeAudioContext(oldContextID)
		}

		if wasSpeaking {
			log.Printf("[ElevenLabsTTS]   Emitting TTSStoppedFrame upstream to notify aggregators")
			s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
		}

		log.Printf("[ElevenLabsTTS] Interruption handled (was_speaking=%v, closed_context=%s)", wasSpeaking, oldContextID)

		return s.PushFrame(frame, direction)
	}

	// Process text frames (LLM output)
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		// Lazy initialization on first text frame
		if s.ctx == nil {
			log.Printf("[ElevenLabsTTS] Lazy initializing on first TextFrame")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[ElevenLabsTTS] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		return s.processTextInput(textFrame.Text)
	}

	if llmFrame, ok := frame.(*frames.LLMTextFrame); ok {
		// Lazy initialization on first text frame
		if s.ctx == nil {
			log.Printf("[ElevenLabsTTS] Lazy initializing on first LLMTextFrame")
			if err := s.Initialize(ctx); err != nil {
				log.Printf("[ElevenLabsTTS] Failed to initialize: %v", err)
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
			log.Printf("[ElevenLabsTTS] Flushing remaining text: %s", remainingText)
			if err := s.synthesizeText(remainingText); err != nil {
				log.Printf("[ElevenLabsTTS] Error synthesizing remaining text: %v", err)
			}
		}

		if s.useStreaming && s.conn != nil && s.contextID != "" {
			log.Printf("[ElevenLabsTTS] LLM response ended, sending flush to generate final audio")
			// Send flush message with context_id
			flushMsg := map[string]interface{}{
				"text":       "",
				"context_id": s.contextID,
				"flush":      true,
			}
			if err := s.conn.WriteJSON(flushMsg); err != nil {
				log.Printf("[ElevenLabsTTS] Error sending flush: %v", err)
			}

			// Note: We do NOT close context here - let it persist for efficiency
			// Context will be closed on next interruption

			s.mu.Lock()
			wasSpeaking := s.isSpeaking
			s.isSpeaking = false
			s.cumulativeTime = 0
			s.partialWord = ""
			s.partialWordStartTime = 0.0
			s.ttfbRecorded = false
			s.mu.Unlock()

			if wasSpeaking {
				log.Printf("[ElevenLabsTTS] Synthesis completed, context %s flushed (will persist)", s.contextID)
			}
		} else {
			// Non-streaming mode - reset flag and emit stopped frame immediately
			s.mu.Lock()
			if s.isSpeaking {
				s.isSpeaking = false
				s.mu.Unlock()
				log.Printf("[ElevenLabsTTS] Emitting TTSStoppedFrame (LLM response ended)")
				s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
			} else {
				s.mu.Unlock()
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
			log.Printf("[ElevenLabsTTS] Synthesizing sentence: %s", sentence)
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

	return sentences, currentSentence.String()
}

func (s *TTSService) synthesizeText(text string) error {
	if text == "" {
		return nil
	}

	// Generate new context ID if needed (after flush or first use)
	if s.contextID == "" && s.useStreaming {
		s.mu.Lock()
		s.contextID = uuid.New().String()
		log.Printf("[ElevenLabsTTS] Generated new context ID: %s", s.contextID)
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
		s.cumulativeTime = 0
		s.partialWord = ""
		s.partialWordStartTime = 0.0
		s.mu.Unlock()

		log.Printf("[ElevenLabsTTS] Emitting TTSStartedFrame (first text chunk)")
		// Push UPSTREAM so UserAggregator can track bot speaking state
		s.PushFrame(frames.NewTTSStartedFrame(), frames.Upstream)
		// Push DOWNSTREAM so WebSocketOutput can reset llmResponseEnded flag
		s.PushFrame(frames.NewTTSStartedFrame(), frames.Downstream)

		// Create audio context for this synthesis
		if s.useStreaming {
			s.createAudioContext(s.contextID)
		}
	} else {
		s.mu.Unlock()
	}

	// Log first token latency for monitoring parallel processing performance
	if firstToken {
		log.Printf("[ElevenLabsTTS] FIRST TOKEN -> Starting audio generation (parallel LLM+TTS)")
	}

	if s.useStreaming && s.conn != nil {
		// Send text chunk via WebSocket with context_id
		msg := map[string]interface{}{
			"text":                   text,
			"context_id":             s.contextID,
			"try_trigger_generation": true,
		}
		return s.conn.WriteJSON(msg)
	} else {
		// Use HTTP API for non-streaming
		return s.synthesizeHTTP(text)
	}
}

func (s *TTSService) synthesizeHTTP(text string) error {
	// Add output_format parameter to URL
	url := fmt.Sprintf("https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=%s",
		s.voiceID, s.outputFormat)

	requestBody := map[string]interface{}{
		"text":     text,
		"model_id": s.model,
	}

	// Add voice settings
	if s.voiceSettings != nil {
		voiceSettingsMap := map[string]interface{}{}
		if s.voiceSettings.Stability != 0 {
			voiceSettingsMap["stability"] = s.voiceSettings.Stability
		}
		if s.voiceSettings.SimilarityBoost != 0 {
			voiceSettingsMap["similarity_boost"] = s.voiceSettings.SimilarityBoost
		}
		if s.voiceSettings.Style != 0 {
			voiceSettingsMap["style"] = s.voiceSettings.Style
		}
		if s.voiceSettings.UseSpeakerBoost {
			voiceSettingsMap["use_speaker_boost"] = s.voiceSettings.UseSpeakerBoost
		}
		if s.voiceSettings.Speed != 0 {
			voiceSettingsMap["speed"] = s.voiceSettings.Speed
		}
		if len(voiceSettingsMap) > 0 {
			requestBody["voice_settings"] = voiceSettingsMap
		}
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("xi-api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("ElevenLabs API error: %s", string(body))
	}

	// Read audio data
	audioData, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	// Determine sample rate and codec from output format
	sampleRate, codec := s.parseOutputFormat()

	// Create TTS audio frame with codec metadata
	audioFrame := frames.NewTTSAudioFrame(audioData, sampleRate, 1)
	audioFrame.SetMetadata("codec", codec)
	if err := s.PushFrame(audioFrame, frames.Downstream); err != nil {
		return err
	}

	// Emit TTSStoppedFrame after audio is pushed (HTTP mode completes immediately)
	s.mu.Lock()
	s.isSpeaking = false
	s.mu.Unlock()
	log.Printf("[ElevenLabsTTS] Emitting TTSStoppedFrame (HTTP synthesis complete)")
	return s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
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
	log.Printf("[ElevenLabsTTS] Created audio context: %s", contextID)
}

func (s *TTSService) removeAudioContext(contextID string) {
	s.contextMu.Lock()
	defer s.contextMu.Unlock()

	delete(s.audioContexts, contextID)
	log.Printf("[ElevenLabsTTS] Removed audio context: %s", contextID)
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

// calculateWordTimes extracts word timing from alignment info
func (s *TTSService) calculateWordTimes(alignment map[string]interface{}) []WordTimestamp {
	chars, charsOK := alignment["chars"].([]interface{})
	charStartTimesMs, timesOK := alignment["charStartTimesMs"].([]interface{})

	if !charsOK || !timesOK || len(chars) != len(charStartTimesMs) {
		log.Printf("[ElevenLabsTTS] Invalid alignment data")
		return nil
	}

	var timestamps []WordTimestamp
	currentWord := s.partialWord
	wordStartTime := s.partialWordStartTime

	for i := 0; i < len(chars); i++ {
		char, ok := chars[i].(string)
		if !ok {
			continue
		}

		if char == " " {
			// End of word
			if currentWord != "" {
				timestamps = append(timestamps, WordTimestamp{
					Word:      currentWord,
					StartTime: wordStartTime,
				})
				currentWord = ""
				wordStartTime = 0
			}
		} else {
			// Building word
			if currentWord == "" {
				// First character of new word
				if startTimeMs, ok := charStartTimesMs[i].(float64); ok {
					wordStartTime = s.cumulativeTime + (startTimeMs / 1000.0)
				}
			}
			currentWord += char
		}
	}

	// Update partial word state
	s.partialWord = currentWord
	s.partialWordStartTime = wordStartTime

	// Update cumulative time based on last character
	if len(charStartTimesMs) > 0 {
		if charDurationsMs, ok := alignment["charDurationsMs"].([]interface{}); ok && len(charDurationsMs) > 0 {
			if lastStartMs, ok := charStartTimesMs[len(charStartTimesMs)-1].(float64); ok {
				if lastDurationMs, ok := charDurationsMs[len(charDurationsMs)-1].(float64); ok {
					chunkEndTime := (lastStartMs + lastDurationMs) / 1000.0
					s.cumulativeTime += chunkEndTime
				}
			}
		}
	}

	return timestamps
}

func (s *TTSService) receiveAudio() {
	for {
		select {
		case <-s.ctx.Done():
			log.Printf("[ElevenLabsTTS] Context cancelled, stopping audio receiver")
			return
		default:
			if s.conn == nil {
				log.Printf("[ElevenLabsTTS] Connection is nil, stopping receiver")
				return
			}

			messageType, message, err := s.conn.ReadMessage()
			if err != nil {
				// Check if this is a normal closure during shutdown
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					log.Printf("[ElevenLabsTTS] Connection closed normally")
					return
				}
				log.Printf("[ElevenLabsTTS] Error reading message: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			if messageType == websocket.BinaryMessage {
				// Binary audio data (rare, but handle it)
				log.Printf("[ElevenLabsTTS] Received binary audio chunk: %d bytes", len(message))
				sampleRate, codec := s.parseOutputFormat()
				audioFrame := frames.NewTTSAudioFrame(message, sampleRate, 1)
				audioFrame.SetMetadata("codec", codec)
				s.PushFrame(audioFrame, frames.Downstream)
			} else {
				// JSON message (contains base64-encoded audio + metadata)
				var response map[string]interface{}
				if err := json.Unmarshal(message, &response); err != nil {
					log.Printf("[ElevenLabsTTS] Error parsing response: %v", err)
					continue
				}

				// Get context ID from response
				receivedCtxID, hasCtxID := response["contextId"].(string)

				// Check isFinal first - if true, this is just an end marker
				if isFinal, ok := response["isFinal"].(bool); ok && isFinal {
					log.Printf("[ElevenLabsTTS] Received final message for context: %s", receivedCtxID)

					// Get audio context stats before removing
					if hasCtxID {
						s.contextMu.RLock()
						if ctx, exists := s.audioContexts[receivedCtxID]; exists {
							duration := time.Since(ctx.StartTime)
							log.Printf("[ElevenLabsTTS] Context %s completed: %d audio frames, %d bytes, %d words, duration: %v",
								receivedCtxID, len(ctx.AudioFrames), ctx.TotalAudioBytes, len(ctx.WordTimestamps), duration)
						}
						s.contextMu.RUnlock()

						s.removeAudioContext(receivedCtxID)
					}

					s.mu.Lock()
					if s.isSpeaking {
						s.isSpeaking = false
						log.Printf("[ElevenLabsTTS] Synthesis completed (WebSocketOutput will emit TTSStoppedFrame after playback)")
					}
					s.mu.Unlock()
					continue
				}

				// Validate context ID to avoid processing old/stale messages
				if hasCtxID {
					s.mu.Lock()
					currentCtxID := s.contextID
					s.mu.Unlock()

					if receivedCtxID != currentCtxID && !s.audioContextAvailable(receivedCtxID) {
						log.Printf("[ElevenLabsTTS] IGNORING audio from OLD context: %s (current: %s)", receivedCtxID, currentCtxID)
						continue
					}
				}

				// Extract and decode audio if present
				if audioB64, ok := response["audio"].(string); ok && audioB64 != "" {
					// Record TTFB on first audio chunk
					s.mu.Lock()
					if !s.ttfbRecorded && !s.ttfbStart.IsZero() {
						ttfb := time.Since(s.ttfbStart)
						s.ttfbRecorded = true
						log.Printf("[ElevenLabsTTS] TTFB (Time to First Byte): %v", ttfb)
					}
					s.mu.Unlock()

					// Decode base64 audio
					audioData, err := base64.StdEncoding.DecodeString(audioB64)
					if err != nil {
						log.Printf("[ElevenLabsTTS] Error decoding base64 audio: %v", err)
						continue
					}

					sampleRate, codec := s.parseOutputFormat()
					audioFrame := frames.NewTTSAudioFrame(audioData, sampleRate, 1)
					audioFrame.SetMetadata("codec", codec)
					audioFrame.SetMetadata("context_id", receivedCtxID)

					// Add to audio context for tracking
					if hasCtxID {
						s.appendToAudioContext(receivedCtxID, audioFrame)
					}

					s.PushFrame(audioFrame, frames.Downstream)
				}

				// Process alignment data for word timestamps
				if alignment, ok := response["alignment"].(map[string]interface{}); ok {
					timestamps := s.calculateWordTimes(alignment)
					if hasCtxID && len(timestamps) > 0 {
						log.Printf("[ElevenLabsTTS] Received %d word timestamps", len(timestamps))
						s.addWordTimestamps(receivedCtxID, timestamps)
					}
				}
			}
		}
	}
}

// parseOutputFormat extracts sample rate and codec from output format string
func (s *TTSService) parseOutputFormat() (int, string) {
	switch s.outputFormat {
	case "ulaw_8000":
		return 8000, "mulaw"
	case "alaw_8000":
		return 8000, "alaw"
	case "pcm_16000":
		return 16000, "linear16"
	case "pcm_22050":
		return 22050, "linear16"
	case "pcm_24000":
		return 24000, "linear16"
	case "pcm_44100":
		return 44100, "linear16"
	default:
		return 24000, "linear16"
	}
}
