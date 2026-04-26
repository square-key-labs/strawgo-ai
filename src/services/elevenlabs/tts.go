package elevenlabs

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	*services.AudioContextManager
	apiKey string

	// settingsMu protects the runtime-mutable fields so concurrent
	// UpdateSettings (system goroutine) cannot race the synthesis path
	// that reads them while building outgoing context messages.
	settingsMu sync.RWMutex
	voiceID    string
	model      string
	language   string // Language code for multilingual models

	outputFormat       string
	useStreaming       bool
	voiceSettings      *VoiceSettings
	aggregateSentences bool
	conn               *websocket.Conn
	ctx                context.Context
	cancel             context.CancelFunc
	codecDetected      bool // Track if we've auto-detected codec from StartFrame
	log                *logger.Logger

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
	mu         sync.Mutex // Protect concurrent access to isSpeaking and service-specific state
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
		apiKey:              config.APIKey,
		voiceID:             config.VoiceID,
		model:               config.Model,
		outputFormat:        outputFormat,
		useStreaming:        config.UseStreaming,
		voiceSettings:       voiceSettings,
		language:            config.Language,
		aggregateSentences:  aggregateSentences,
		codecDetected:       codecDetected,
		log:                 logger.WithPrefix("ElevenLabsTTS"),
		audioContexts:       make(map[string]*AudioContext),
		AudioContextManager: services.NewAudioContextManager(),
	}
	es.BaseProcessor = processors.NewBaseProcessor("ElevenLabsTTS", es)
	return es
}

func (s *TTSService) SetVoice(voiceID string) {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()
	s.voiceID = voiceID
}

func (s *TTSService) SetModel(model string) {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()
	s.model = model
}

func (s *TTSService) SetVoiceSettings(settings *VoiceSettings) {
	s.voiceSettings = settings
}

func (s *TTSService) SetLanguage(language string) {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()
	s.language = language
}

// UpdateSettings applies a runtime settings update to the ElevenLabs TTS
// service. Recognized keys: "voice" (string voice ID), "model" (string),
// "language" (string). Unknown keys are ignored with a debug log.
// Settings apply to subsequent synthesis turns; ElevenLabs's WebSocket
// URL hard-codes the voice ID, so a voice change does not retune an
// already-open multi-stream session.
func (s *TTSService) UpdateSettings(settings map[string]interface{}) error {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()
	for k, v := range settings {
		switch k {
		case "voice":
			if str, ok := v.(string); ok && str != "" {
				s.voiceID = str
			}
		case "model":
			if str, ok := v.(string); ok && str != "" {
				s.model = str
			}
		case "language":
			if str, ok := v.(string); ok {
				s.language = str
			}
		default:
			s.log.Debug("UpdateSettings: ignoring unknown key %q", k)
		}
	}
	return nil
}

func (s *TTSService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	if s.useStreaming {
		// Generate context ID for multi-stream mode
		s.SetActiveAudioContextID(services.GenerateContextID())

		// Snapshot voice/model/language under settingsMu so a concurrent
		// UpdateSettings cannot tear the URL we are about to build.
		s.settingsMu.RLock()
		voiceID := s.voiceID
		model := s.model
		language := s.language
		s.settingsMu.RUnlock()

		// Build WebSocket URL with multi-stream-input endpoint and output_format
		wsURL := fmt.Sprintf("wss://api.elevenlabs.io/v1/text-to-speech/%s/multi-stream-input?model_id=%s&output_format=%s&auto_mode=true",
			voiceID, model, s.outputFormat)

		// Add language code for multilingual models
		if language != "" && multilingualModels[model] {
			wsURL += fmt.Sprintf("&language_code=%s", language)
			s.log.Info("Using language code: %s", language)
		}

		header := http.Header{}
		header.Set("xi-api-key", s.apiKey)

		var err error
		s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, header)
		if err != nil {
			return fmt.Errorf("failed to connect to ElevenLabs: %w", err)
		}

		// Send initial config with context_id and voice settings
		ctxID := s.GetActiveAudioContextID()
		config := map[string]interface{}{
			"text":       " ",
			"context_id": ctxID,
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

		s.log.Info("Streaming mode connected (context: %s)", ctxID)
	} else {
		s.log.Info("Non-streaming mode initialized")
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

	// Now close the connection. NOTE: pre-existing concurrency hazard --
	// s.conn is touched here without a websocket-write mutex while
	// streaming goroutines may still be in s.conn.WriteJSON. Codex called
	// this out as a real race; tracking it as out-of-scope for PR 4
	// (would require introducing a wsMu like Cartesia has). The 50ms
	// sleep above is the existing best-effort barrier.
	if s.conn != nil {
		// Send close message before closing socket (for ElevenLabs)
		if s.HasActiveAudioContext() {
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
			ctxID := s.GetActiveAudioContextID()
			if s.conn != nil && ctxID != "" {
				keepaliveMsg := map[string]interface{}{
					"text":       "",
					"context_id": ctxID,
				}
				if err := s.conn.WriteJSON(keepaliveMsg); err != nil {
					s.log.Warn("Keepalive error: %v", err)
					return
				}
			}
		}
	}
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Honor TTSUpdateSettingsFrame (forward unchanged so other TTS services
	// in the pipeline can also see it).
	if updateFrame, ok := frame.(*frames.TTSUpdateSettingsFrame); ok {
		if updateFrame.Service == "" || updateFrame.Service == s.Name() {
			if err := s.UpdateSettings(updateFrame.Settings); err != nil {
				s.log.Warn("UpdateSettings failed: %v", err)
			} else {
				s.log.Info("Applied runtime settings: %v", updateFrame.Settings)
			}
		}
		return s.PushFrame(frame, direction)
	}

	// Handle StartFrame - codec detection AND eager initialization
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		// Auto-detect output format from incoming codec (only if user didn't set OutputFormat)
		if !s.codecDetected {
			if meta := startFrame.Metadata(); meta != nil {
				if codec, ok := meta["codec"].(string); ok {
					s.log.Info("Detected incoming codec: %s", codec)
					// Match ElevenLabs output to incoming codec for compatibility
					switch codec {
					case "mulaw":
						s.outputFormat = "ulaw_8000"
						s.log.Info("Auto-configured output format: ulaw_8000")
					case "alaw":
						s.outputFormat = "alaw_8000"
						s.log.Info("Auto-configured output format: alaw_8000")
					case "linear16":
						s.outputFormat = "pcm_16000"
						s.log.Info("Auto-configured output format: pcm_16000")
					}
					s.codecDetected = true
				}
			}
		}

		// Eager initialization for parallel LLM+TTS processing
		if s.useStreaming && s.ctx == nil {
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
		s.log.Info("INTERRUPTION RECEIVED - Stopping TTS synthesis")
		oldContextID := s.GetActiveAudioContextID()
		s.mu.Lock()
		wasSpeaking := s.isSpeaking
		if s.isSpeaking {
			s.isSpeaking = false
		}
		// Clear text buffer and word tracking on interruption
		s.textBuffer.Reset()
		s.partialWord = ""
		s.partialWordStartTime = 0.0
		s.cumulativeTime = 0
		s.ttfbRecorded = false
		s.mu.Unlock()
		// Reset context IDs via AudioContextManager
		s.ResetActiveAudioContext()

		// CRITICAL: Always close the context if it exists, regardless of wasSpeaking
		// This prevents context accumulation on ElevenLabs
		if s.useStreaming && s.conn != nil && oldContextID != "" {
			s.log.Debug("Closing context %s on ElevenLabs (was_speaking=%v)", oldContextID, wasSpeaking)
			closeMsg := map[string]interface{}{
				"context_id":    oldContextID,
				"close_context": true,
			}
			if err := s.conn.WriteJSON(closeMsg); err != nil {
				s.log.Debug("Error closing context: %v", err)
			}

			// Remove old audio context
			s.removeAudioContext(oldContextID)
		}

		if wasSpeaking {
			s.log.Debug("Emitting TTSStoppedFrame upstream to notify aggregators")
			s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
		}

		s.log.Debug("Interruption handled (was_speaking=%v, closed_context=%s)", wasSpeaking, oldContextID)

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
		// Flush any remaining text in buffer
		if s.textBuffer.Len() > 0 {
			remainingText := s.textBuffer.String()
			s.textBuffer.Reset()
			s.log.Debug("Flushing remaining text: %s", remainingText)
			if err := s.synthesizeText(remainingText); err != nil {
				s.log.Warn("Error synthesizing remaining text: %v", err)
			}
		}

		ctxID := s.GetActiveAudioContextID()
		if s.useStreaming && s.conn != nil && ctxID != "" {
			s.log.Info("LLM response ended, sending flush to generate final audio")
			// Send flush message with context_id
			flushMsg := map[string]interface{}{
				"text":       "",
				"context_id": ctxID,
				"flush":      true,
			}
			if err := s.conn.WriteJSON(flushMsg); err != nil {
				s.log.Warn("Error sending flush: %v", err)
			}

			// CRITICAL: Close context after normal completion (not just on interruption)
			// This prevents context accumulation on ElevenLabs
			s.mu.Lock()
			wasSpeaking := s.isSpeaking
			s.isSpeaking = false
			s.cumulativeTime = 0
			s.partialWord = ""
			s.partialWordStartTime = 0.0
			s.ttfbRecorded = false
			s.mu.Unlock()
			s.ResetActiveAudioContext()

			s.log.Info("Closing context %s on normal completion (was_speaking=%v)", ctxID, wasSpeaking)
			closeMsg := map[string]interface{}{
				"context_id":    ctxID,
				"close_context": true,
			}
			if err := s.conn.WriteJSON(closeMsg); err != nil {
				s.log.Debug("Error closing context: %v", err)
			}

			// Remove audio context
			s.removeAudioContext(ctxID)

			if wasSpeaking {
				s.log.Info("Synthesis completed, context %s closed", ctxID)
			}
		} else {
			// Non-streaming mode - reset flags.
			// Guard with wasSpeaking: synthesizeHTTP already emits TTSStoppedFrame on
			// HTTP completion, so we only emit here if speaking is still active (edge
			// case: LLMFullResponseEndFrame arrives before HTTP response returns).
			s.mu.Lock()
			wasSpeaking := s.isSpeaking
			s.isSpeaking = false
			s.mu.Unlock()
			s.ResetActiveAudioContext()

			if wasSpeaking {
				s.log.Info("Emitting TTSStoppedFrame (LLM response ended, non-streaming)")
				s.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
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
		s.cumulativeTime = 0
		s.partialWord = ""
		s.partialWordStartTime = 0.0
		s.mu.Unlock()

		s.log.Info("Emitting TTSStartedFrame (first text chunk) with context ID: %s", ctxID)
		// Push UPSTREAM so UserAggregator can track bot speaking state
		s.PushFrame(frames.NewTTSStartedFrameWithContext(ctxID), frames.Upstream)
		// Push DOWNSTREAM so WebSocketOutput can reset llmResponseEnded flag and set expected context
		s.PushFrame(frames.NewTTSStartedFrameWithContext(ctxID), frames.Downstream)

		// Create audio context for this synthesis
		if s.useStreaming {
			s.createAudioContext(ctxID)
		}
	} else {
		s.mu.Unlock()
	}

	// Log first token latency for monitoring parallel processing performance
	if firstToken {
		s.log.Info("FIRST TOKEN -> Starting audio generation (parallel LLM+TTS)")
	}

	if s.useStreaming && s.conn != nil {
		// Send text chunk via WebSocket with context_id
		msg := map[string]interface{}{
			"text":                   text,
			"context_id":             ctxID,
			"try_trigger_generation": true,
		}
		return s.conn.WriteJSON(msg)
	} else {
		// Use HTTP API for non-streaming
		return s.synthesizeHTTP(text)
	}
}

func (s *TTSService) synthesizeHTTP(text string) error {
	// Snapshot voice/model under settingsMu so a concurrent UpdateSettings
	// cannot tear the URL or body we are about to build.
	s.settingsMu.RLock()
	voiceID := s.voiceID
	model := s.model
	s.settingsMu.RUnlock()

	// Add output_format parameter to URL
	url := fmt.Sprintf("https://api.elevenlabs.io/v1/text-to-speech/%s?output_format=%s",
		voiceID, s.outputFormat)

	requestBody := map[string]interface{}{
		"text":     text,
		"model_id": model,
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
	s.log.Info("Emitting TTSStoppedFrame (HTTP synthesis complete)")
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
	s.log.Info("Created audio context: %s", contextID)
}

func (s *TTSService) removeAudioContext(contextID string) {
	s.contextMu.Lock()
	defer s.contextMu.Unlock()

	delete(s.audioContexts, contextID)
	s.log.Info("Removed audio context: %s", contextID)
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
		s.log.Warn("Invalid alignment data")
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
				s.log.Error("Error reading message: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			if messageType == websocket.BinaryMessage {
				// Binary audio data (rare, but handle it)
				s.log.Debug("Received binary audio chunk: %d bytes", len(message))
				sampleRate, codec := s.parseOutputFormat()
				audioFrame := frames.NewTTSAudioFrame(message, sampleRate, 1)
				audioFrame.SetMetadata("codec", codec)
				s.PushFrame(audioFrame, frames.Downstream)
			} else {
				// JSON message (contains base64-encoded audio + metadata)
				var response map[string]interface{}
				if err := json.Unmarshal(message, &response); err != nil {
					s.log.Error("Error parsing response: %v", err)
					continue
				}

				// Get context ID from response
				receivedCtxID, hasCtxID := response["contextId"].(string)

				// Check isFinal first - if true, this is just an end marker
				if isFinal, ok := response["isFinal"].(bool); ok && isFinal {
					s.log.Info("Received final message for context: %s", receivedCtxID)

					// Get audio context stats before removing
					if hasCtxID {
						s.contextMu.RLock()
						if ctx, exists := s.audioContexts[receivedCtxID]; exists {
							duration := time.Since(ctx.StartTime)
							s.log.Info("Context %s completed: %d audio frames, %d bytes, %d words, duration: %v",
								receivedCtxID, len(ctx.AudioFrames), ctx.TotalAudioBytes, len(ctx.WordTimestamps), duration)
						}
						s.contextMu.RUnlock()

						s.removeAudioContext(receivedCtxID)
					}

					s.mu.Lock()
					if s.isSpeaking {
						s.isSpeaking = false
						s.log.Info("Synthesis completed (WebSocketOutput will emit TTSStoppedFrame after playback)")
					}
					s.mu.Unlock()
					continue
				}

				// Validate context ID to avoid processing old/stale messages
				if hasCtxID {
					currentCtxID := s.GetActiveAudioContextID()

					if receivedCtxID != currentCtxID && !s.audioContextAvailable(receivedCtxID) {
						s.log.Debug("IGNORING audio from OLD context: %s (current: %s)", receivedCtxID, currentCtxID)
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
						s.log.Info("TTFB (Time to First Byte): %v", ttfb)
					}
					s.mu.Unlock()

					// Decode base64 audio
					audioData, err := base64.StdEncoding.DecodeString(audioB64)
					if err != nil {
						s.log.Error("Error decoding base64 audio: %v", err)
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
						s.log.Debug("Received %d word timestamps", len(timestamps))
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
