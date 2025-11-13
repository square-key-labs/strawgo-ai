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
	"time"

	"github.com/google/uuid"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/gorilla/websocket"
)

// TTSService provides text-to-speech using ElevenLabs
type TTSService struct {
	*processors.BaseProcessor
	apiKey        string
	voiceID       string
	model         string
	outputFormat  string
	useStreaming  bool
	conn          *websocket.Conn
	ctx           context.Context
	cancel        context.CancelFunc
	textBuffer    strings.Builder
	codecDetected bool   // Track if we've auto-detected codec from StartFrame
	contextID     string // ElevenLabs context ID for multi-stream mode
}

// TTSConfig holds configuration for ElevenLabs
type TTSConfig struct {
	APIKey       string
	VoiceID      string // e.g., "21m00Tcm4TlvDq8ikWAM" (Rachel)
	Model        string // e.g., "eleven_turbo_v2"
	OutputFormat string // Supported: "ulaw_8000", "alaw_8000", "pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100" (default: "pcm_24000")
	UseStreaming bool   // Use WebSocket streaming for lower latency
}

// NewTTSService creates a new ElevenLabs TTS service
func NewTTSService(config TTSConfig) *TTSService {
	outputFormat := config.OutputFormat
	codecDetected := true // Assume user explicitly set format

	if outputFormat == "" {
		outputFormat = "pcm_24000" // Default PCM at 24kHz
		codecDetected = false // Will auto-detect from StartFrame
	}

	es := &TTSService{
		apiKey:        config.APIKey,
		voiceID:       config.VoiceID,
		model:         config.Model,
		outputFormat:  outputFormat,
		useStreaming:  config.UseStreaming,
		codecDetected: codecDetected, // Only auto-detect if not explicitly set
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

func (s *TTSService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	if s.useStreaming {
		// Generate context ID for multi-stream mode
		s.contextID = uuid.New().String()

		// Build WebSocket URL with multi-stream-input endpoint and output_format
		wsURL := fmt.Sprintf("wss://api.elevenlabs.io/v1/text-to-speech/%s/multi-stream-input?model_id=%s&output_format=%s",
			s.voiceID, s.model, s.outputFormat)

		header := http.Header{}
		header.Set("xi-api-key", s.apiKey)

		var err error
		s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, header)
		if err != nil {
			return fmt.Errorf("failed to connect to ElevenLabs: %w", err)
		}

		// Send initial config with context_id (API key goes in header, not message)
		config := map[string]interface{}{
			"text":       " ",
			"context_id": s.contextID,
			"voice_settings": map[string]interface{}{
				"stability":        0.5,
				"similarity_boost": 0.75,
			},
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
	if s.cancel != nil {
		s.cancel()
	}
	if s.conn != nil {
		// Close the context before closing the socket
		if s.contextID != "" {
			closeMsg := map[string]interface{}{
				"close_socket": true,
			}
			s.conn.WriteJSON(closeMsg)
		}
		s.conn.Close()
	}
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
				log.Printf("[ElevenLabsTTS] Sent keepalive")
			}
		}
	}
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame to detect codec from Asterisk
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
		return s.synthesizeText(textFrame.Text)
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
		return s.synthesizeText(llmFrame.Text)
	}

	// Handle LLM response end to flush TTS
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
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
		}
		return s.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

func (s *TTSService) synthesizeText(text string) error {
	if text == "" {
		return nil
	}

	log.Printf("[ElevenLabsTTS] Synthesizing: %s", text)

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
		"voice_settings": map[string]interface{}{
			"stability":        0.5,
			"similarity_boost": 0.75,
		},
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
	return s.PushFrame(audioFrame, frames.Downstream)
}

func (s *TTSService) receiveAudio() {
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
			messageType, message, err := s.conn.ReadMessage()
			if err != nil {
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
				log.Printf("[ElevenLabsTTS] Pushing TTSAudioFrame downstream (codec: %s, rate: %d)", codec, sampleRate)
				s.PushFrame(audioFrame, frames.Downstream)
			} else {
				// JSON message (contains base64-encoded audio + metadata)
				var response map[string]interface{}
				if err := json.Unmarshal(message, &response); err != nil {
					log.Printf("[ElevenLabsTTS] Error parsing response: %v", err)
					continue
				}

				// Check isFinal first - if true, this is just an end marker
				if isFinal, ok := response["isFinal"].(bool); ok && isFinal {
					log.Printf("[ElevenLabsTTS] Received final message for context")
					continue // Skip processing, this is just metadata
				}

				// Validate context ID to avoid processing old/stale messages
				if receivedCtxID, ok := response["contextId"].(string); ok {
					if receivedCtxID != s.contextID {
						log.Printf("[ElevenLabsTTS] Ignoring message from old/different context: %s (current: %s)", receivedCtxID, s.contextID)
						continue
					}
				}

				// Extract and decode audio if present
				if audioB64, ok := response["audio"].(string); ok && audioB64 != "" {
					// Decode base64 audio
					audioData, err := base64.StdEncoding.DecodeString(audioB64)
					if err != nil {
						log.Printf("[ElevenLabsTTS] Error decoding base64 audio: %v", err)
						continue
					}

					log.Printf("[ElevenLabsTTS] Received audio chunk: %d bytes (decoded from base64)", len(audioData))
					sampleRate, codec := s.parseOutputFormat()
					audioFrame := frames.NewTTSAudioFrame(audioData, sampleRate, 1)
					audioFrame.SetMetadata("codec", codec)
					log.Printf("[ElevenLabsTTS] Pushing TTSAudioFrame downstream (codec: %s, rate: %d)", codec, sampleRate)
					s.PushFrame(audioFrame, frames.Downstream)
				}

				// Optionally log alignment metadata for debugging word timestamps
				if _, ok := response["alignment"]; ok {
					log.Printf("[ElevenLabsTTS] Received alignment data (word timing)")
				}
			}
		}
	}
}

// parseOutputFormat extracts sample rate and codec from output format string
// ElevenLabs supports both telephony codecs (ulaw/alaw) and PCM
func (s *TTSService) parseOutputFormat() (int, string) {
	// Parse ElevenLabs format: CODEC_SAMPLERATE
	// Supported: ulaw_8000, alaw_8000, pcm_16000, pcm_22050, pcm_24000, pcm_44100
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
		// Default to PCM 24kHz
		return 24000, "linear16"
	}
}
