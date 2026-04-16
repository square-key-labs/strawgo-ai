package azure

import (
	"context"
	"encoding/json"
	"errors"
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

const (
	AzureSTTURLTemplate = "wss://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
	DefaultRegion       = "eastus"
	DefaultLanguage     = "en-US"
	DefaultEncoding     = "audio/x-wav"
	DefaultSampleRate   = 16000
)

// STTService provides speech-to-text using Azure Cognitive Services
type STTService struct {
	*processors.BaseProcessor

	subscriptionKey   string
	region            string
	language          string
	encoding          string
	sampleRate        int
	keepaliveInterval time.Duration
	keepaliveTimeout  time.Duration

	conn        *websocket.Conn
	ctx         context.Context
	cancel      context.CancelFunc
	connMu      sync.Mutex
	goroutineWG sync.WaitGroup
	connDropped atomic.Bool
}

// STTConfig holds configuration for Azure STT
type STTConfig struct {
	SubscriptionKey   string        // Azure subscription key
	Region            string        // e.g., "eastus", "westus"
	Language          string        // e.g., "en-US", "es-ES"
	Encoding          string        // Audio encoding format
	SampleRate        int           // Sample rate in Hz
	KeepaliveInterval time.Duration // Interval for sending keepalive pings (default: 5s)
	KeepaliveTimeout  time.Duration // Timeout for keepalive (default: 30s)
}

// NewSTTService creates a new Azure STT service
func NewSTTService(config STTConfig) *STTService {
	region := config.Region
	if region == "" {
		region = DefaultRegion
	}

	language := config.Language
	if language == "" {
		language = DefaultLanguage
	}

	encoding := config.Encoding
	if encoding == "" {
		encoding = DefaultEncoding
	}

	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultSampleRate
	}

	// Set keepalive defaults
	keepaliveInterval := config.KeepaliveInterval
	if keepaliveInterval == 0 {
		keepaliveInterval = 5 * time.Second
	}
	keepaliveTimeout := config.KeepaliveTimeout
	if keepaliveTimeout == 0 {
		keepaliveTimeout = 30 * time.Second
	}

	service := &STTService{
		subscriptionKey:   config.SubscriptionKey,
		region:            region,
		language:          language,
		encoding:          encoding,
		sampleRate:        sampleRate,
		keepaliveInterval: keepaliveInterval,
		keepaliveTimeout:  keepaliveTimeout,
	}

	service.BaseProcessor = processors.NewBaseProcessor("AzureSTT", service)
	return service
}

// SetLanguage sets the language code
func (s *STTService) SetLanguage(lang string) {
	s.language = lang
}

// SetModel sets the model (not used for Azure STT)
func (s *STTService) SetModel(model string) {
	// Azure STT doesn't have separate models
	// Recognition mode is set via endpoint (conversation, interactive, dictation)
}

func (s *STTService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)

	baseURL := fmt.Sprintf(AzureSTTURLTemplate, s.region)
	u, err := url.Parse(baseURL)
	if err != nil {
		errMsg := fmt.Sprintf("failed to parse URL: %v", err)
		logger.Error("[AzureSTT] %s", errMsg)
		s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
		return errors.New(errMsg)
	}

	q := u.Query()
	q.Set("language", s.language)
	q.Set("format", "detailed")
	u.RawQuery = q.Encode()

	headers := map[string][]string{
		"Ocp-Apim-Subscription-Key": {s.subscriptionKey},
	}

	var dialer websocket.Dialer
	s.conn, _, err = dialer.Dial(u.String(), headers)
	if err != nil {
		errMsg := fmt.Sprintf("failed to connect to Azure: %v", err)
		logger.Error("[AzureSTT] %s", errMsg)
		s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
		return errors.New(errMsg)
	}

	configMsg := map[string]interface{}{
		"context": map[string]interface{}{
			"system": map[string]interface{}{
				"version": "1.0.00000",
			},
			"os": map[string]interface{}{
				"platform": "Linux",
				"name":     "StrawGo",
				"version":  "1.0.0",
			},
		},
	}

	s.connMu.Lock()
	err = s.conn.WriteJSON(configMsg)
	s.connMu.Unlock()

	if err != nil {
		s.conn.Close()
		s.conn = nil
		errMsg := fmt.Sprintf("failed to send configuration: %v", err)
		logger.Error("[AzureSTT] %s", errMsg)
		s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
		return errors.New(errMsg)
	}

	s.connDropped.Store(false)
	conn := s.conn
	s.goroutineWG.Add(2)
	go s.receiveTranscriptions(conn)
	go s.keepaliveTask(conn)

	logger.Debug("[AzureSTT] Connected and initialized (region=%s, language=%s)", s.region, s.language)
	return nil
}

func (s *STTService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	s.connDropped.Store(true)
	s.disconnect()

	logger.Debug("[AzureSTT] Cleaned up")
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

	s.goroutineWG.Wait()
}

func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if _, ok := frame.(*frames.StartFrame); ok {
		// Emit STT metadata for auto-tuning turn detection
		s.PushFrame(frames.NewSTTMetadataFrame("azure", 500*time.Millisecond), frames.Downstream)
		return s.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.EndFrame); ok {
		logger.Debug("[AzureSTT] Received EndFrame, cleaning up")
		if err := s.Cleanup(); err != nil {
			logger.Error("[AzureSTT] Error during cleanup: %v", err)
		}
		return s.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.InterruptionFrame); ok {
		logger.Debug("[AzureSTT] Received InterruptionFrame")
		return s.PushFrame(frame, direction)
	}

	if audioFrame, ok := frame.(*frames.AudioFrame); ok {
		if s.conn == nil {
			logger.Debug("[AzureSTT] Lazy initializing on first AudioFrame")
			if err := s.Initialize(ctx); err != nil {
				logger.Error("[AzureSTT] Failed to initialize: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}

		if s.connDropped.Load() {
			return s.PushFrame(frame, direction)
		}

		s.connMu.Lock()
		conn := s.conn
		if conn == nil {
			s.connMu.Unlock()
			return s.PushFrame(frame, direction)
		}
		err := conn.WriteMessage(websocket.BinaryMessage, audioFrame.Data)
		s.connMu.Unlock()

		if err != nil {
			logger.Error("[AzureSTT] Error sending audio: %v", err)
			s.connDropped.Store(true)
			s.disconnect()
			return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
		}

		return s.PushFrame(frame, direction)
	}

	return s.PushFrame(frame, direction)
}

func (s *STTService) receiveTranscriptions(conn *websocket.Conn) {
	defer s.goroutineWG.Done()

	for {
		select {
		case <-s.ctx.Done():
			logger.Debug("[AzureSTT] Context cancelled, stopping transcription receiver")
			return
		default:
			_, message, err := conn.ReadMessage()
			if err != nil {
				if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
					strings.Contains(err.Error(), "use of closed network connection") {
					logger.Debug("[AzureSTT] Connection closed normally")
					return
				}
				logger.Error("[AzureSTT] Error reading message: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				return
			}

			var response struct {
				RecognitionStatus string `json:"RecognitionStatus"`
				DisplayText       string `json:"DisplayText"`
				Offset            int64  `json:"Offset"`
				Duration          int64  `json:"Duration"`
			}

			if err := json.Unmarshal(message, &response); err != nil {
				errMsg := fmt.Sprintf("error parsing response: %v", err)
				logger.Error("[AzureSTT] %s", errMsg)
				s.PushFrame(frames.NewErrorFrame(errors.New(errMsg)), frames.Upstream)
				continue
			}

			if response.DisplayText != "" && response.RecognitionStatus == "Success" {
				transcriptionFrame := frames.NewTranscriptionFrame(response.DisplayText, true)
				logger.Debug("[AzureSTT] Transcription: %s", response.DisplayText)
				s.PushFrame(transcriptionFrame, frames.Downstream)
			}
		}
	}
}

func (s *STTService) keepaliveTask(conn *websocket.Conn) {
	defer s.goroutineWG.Done()

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

			// Send a WebSocket ping frame (with mutex protection)
			s.connMu.Lock()
			err := conn.WriteMessage(websocket.PingMessage, []byte{})
			s.connMu.Unlock()

			if err != nil {
				logger.Error("[AzureSTT] Error sending keepalive ping: %v", err)
				return
			}
		}
	}
}
