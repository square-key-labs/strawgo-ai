package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/gorilla/websocket"
)

// TTSService provides text-to-speech using ElevenLabs
type TTSService struct {
	*processors.BaseProcessor
	apiKey       string
	voiceID      string
	model        string
	outputFormat string
	useStreaming bool
	conn         *websocket.Conn
	ctx          context.Context
	cancel       context.CancelFunc
	textBuffer   strings.Builder
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
	if outputFormat == "" {
		outputFormat = "pcm_24000" // Default PCM at 24kHz
	}

	es := &TTSService{
		apiKey:       config.APIKey,
		voiceID:      config.VoiceID,
		model:        config.Model,
		outputFormat: outputFormat,
		useStreaming: config.UseStreaming,
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
		// Initialize WebSocket connection for streaming
		wsURL := fmt.Sprintf("wss://api.elevenlabs.io/v1/text-to-speech/%s/stream-input?model_id=%s",
			s.voiceID, s.model)

		header := http.Header{}
		header.Set("xi-api-key", s.apiKey)

		var err error
		s.conn, _, err = websocket.DefaultDialer.Dial(wsURL, header)
		if err != nil {
			return fmt.Errorf("failed to connect to ElevenLabs: %w", err)
		}

		// Send initial config
		config := map[string]interface{}{
			"text": " ",
			"voice_settings": map[string]interface{}{
				"stability":        0.5,
				"similarity_boost": 0.75,
			},
			"xi_api_key": s.apiKey,
		}

		if err := s.conn.WriteJSON(config); err != nil {
			return fmt.Errorf("failed to send config: %w", err)
		}

		// Start receiving audio
		go s.receiveAudio()

		log.Printf("[ElevenLabsTTS] Streaming mode connected")
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
		s.conn.Close()
	}
	return nil
}

func (s *TTSService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Process text frames (LLM output)
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		return s.synthesizeText(textFrame.Text)
	}

	if llmFrame, ok := frame.(*frames.LLMTextFrame); ok {
		return s.synthesizeText(llmFrame.Text)
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
		// Send text chunk via WebSocket
		msg := map[string]interface{}{
			"text":           text,
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
				// Audio data received
				sampleRate, codec := s.parseOutputFormat()
				audioFrame := frames.NewTTSAudioFrame(message, sampleRate, 1)
				audioFrame.SetMetadata("codec", codec)
				s.PushFrame(audioFrame, frames.Downstream)
			} else {
				// JSON message (metadata)
				var response map[string]interface{}
				if err := json.Unmarshal(message, &response); err != nil {
					log.Printf("[ElevenLabsTTS] Error parsing response: %v", err)
					continue
				}
				log.Printf("[ElevenLabsTTS] Received metadata: %v", response)
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
