package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
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
	sarvamSTTURL          = "wss://api.sarvam.ai/speech-to-text/ws"
	sarvamSTTTranslateURL = "wss://api.sarvam.ai/speech-to-text-translate/ws"

	defaultModel      = "saaras:v3"
	defaultEncoding   = "wav"
	defaultSampleRate = 16000
	defaultMode       = "transcribe"

	// SarvamTTFSP99 is the estimated P99 time-to-first-segment latency used by
	// turn-detection auto-tuning.
	SarvamTTFSP99 = 500 * time.Millisecond
)

// STTConfig holds construction-time configuration for the Sarvam STT service.
type STTConfig struct {
	// APIKey is the Sarvam subscription key (sent as api-subscription-key header).
	APIKey string

	// Model selects the Sarvam STT model.
	//   "saaras:v3"    — recommended; auto-detects language; supports Mode param.
	//   "saarika:v2.5" — requires Language; does not support Mode or prompt.
	//   "saaras:v2.5"  — translate endpoint; supports Prompt; auto-detects language.
	// Defaults to "saaras:v3".
	Model string

	// Language is the BCP-47 language code, e.g. "en-IN", "hi-IN".
	// Required for saarika:v2.5; optional for saaras:v3 (auto-detect).
	Language string

	// Mode controls the output format (saaras:v3 only).
	// One of: "transcribe" (default), "translate", "verbatim", "translit", "codemix".
	Mode string

	// Prompt is an optional vocabulary/hotword hint for transcription and
	// translation (saaras:v2.5 only — deprecated model; prefer saaras:v3 with
	// Mode="translate"). Inject domain-specific terms so they survive translation.
	Prompt string

	// Encoding is the input audio codec arriving in AudioFrame.Data.
	// One of: "wav" (default), "pcm_s16le", "pcm_l16", "pcm_raw".
	// Callers are responsible for sending audio in this format — the service
	// does not transcode. Sent verbatim as input_audio_codec and prefixed with
	// "audio/" in each audio message envelope.
	Encoding string

	// SampleRate is the input audio sample rate in Hz. 8000 or 16000 (default).
	// Must match the actual audio source.
	SampleRate int

	// VADSignals controls server-side VAD event delivery.
	//   nil  — do not send the param; server uses its default.
	//   true — enable Sarvam VAD; emits UserStartedSpeaking/UserStoppedSpeaking.
	//   false — disable Sarvam VAD; flush_signal is auto-enabled so the pipeline
	//           can flush transcription at speech boundaries via external VAD.
	VADSignals *bool

	// HighVADSensitivity controls server-side VAD sensitivity.
	//   nil   — do not send the param; server uses its default.
	//   true  — high sensitivity (fires on shorter pauses ~0.5s).
	//   false — normal sensitivity (~1s).
	HighVADSensitivity *bool

	// KeepaliveInterval, when non-zero, enables periodic silence frames to
	// prevent the server from closing the WebSocket due to inactivity.
	// Disabled by default — only set if your deployment drops idle connections.
	KeepaliveInterval time.Duration
}

// STTService provides real-time speech-to-text via Sarvam AI's streaming
// WebSocket API.  It implements services.STTService.
//
// Connection is established eagerly on StartFrame so the WebSocket is fully
// open before the first AudioFrame arrives, avoiding the connect-race that
// would otherwise trigger a redundant reconnect on every call.
type STTService struct {
	*processors.BaseProcessor

	apiKey             string
	language           string
	model              string
	mode               string
	prompt             string
	encoding           string
	sampleRate         int
	vadSignals         *bool
	highVADSensitivity *bool
	keepaliveInterval  time.Duration
	useTranslateURL    bool // true when model == "saaras:v2.5"

	// connectMu serializes connect/disconnect so readWG.Add and readWG.Wait are
	// never concurrent (avoids WaitGroup reuse panics).
	connectMu sync.Mutex

	connMu  sync.RWMutex    // guards conn and connCancel pointers
	writeMu sync.Mutex      // gorilla websocket is not concurrent-write-safe
	readWG  sync.WaitGroup  // tracks the single active receiveTranscriptions goroutine

	conn       *websocket.Conn
	connCancel context.CancelFunc // cancels the per-connection context on disconnect

	// connDropped is accessed from both the data goroutine (handleAudio) and
	// keepaliveTask, so it must be atomic.
	connDropped atomic.Bool

	// ctx is the service-lifetime context, set on Initialize and cancelled on Cleanup.
	ctx    context.Context
	cancel context.CancelFunc

	log *logger.Logger
}

// NewSTTService creates a new SarvamSTT service from config.
func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = defaultModel
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = defaultEncoding
	}
	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = defaultSampleRate
	}
	mode := config.Mode
	if mode == "" {
		mode = defaultMode
	}

	s := &STTService{
		apiKey:             config.APIKey,
		language:           config.Language,
		model:              model,
		mode:               mode,
		prompt:             config.Prompt,
		encoding:           encoding,
		sampleRate:         sampleRate,
		vadSignals:         config.VADSignals,
		highVADSensitivity: config.HighVADSensitivity,
		keepaliveInterval:  config.KeepaliveInterval,
		useTranslateURL:    model == "saaras:v2.5",
		log:                logger.WithPrefix("SarvamSTT"),
	}
	s.BaseProcessor = processors.NewBaseProcessor("SarvamSTT", s)
	return s
}

// SetLanguage updates the language code used for subsequent connections.
// If lang is a two-letter ISO code (e.g. "hi") it is normalised to the Sarvam
// BCP-47 form (e.g. "hi-IN").
func (s *STTService) SetLanguage(lang string) {
	if code, ok := languageCodes[lang]; ok {
		s.language = code
	} else {
		s.language = lang
	}
}

// SetModel updates the model used for subsequent connections.
func (s *STTService) SetModel(model string) {
	s.model = model
	s.useTranslateURL = model == "saaras:v2.5"
}

// Initialize sets up the service-lifetime context and dials the Sarvam
// WebSocket.  Called automatically from HandleFrame on StartFrame; safe to
// call explicitly before wiring into a pipeline.
func (s *STTService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.connDropped.Store(false)
	return s.connect()
}

// connect dials the Sarvam WebSocket and starts the receive and keepalive
// goroutines.  It holds connectMu for its entire duration so that readWG.Add
// and readWG.Wait are never concurrent.
func (s *STTService) connect() error {
	s.connectMu.Lock()
	defer s.connectMu.Unlock()

	// Idempotent — skip if already connected.
	s.connMu.RLock()
	already := s.conn != nil
	s.connMu.RUnlock()
	if already {
		return nil
	}

	params := url.Values{}
	params.Set("model", s.model)
	params.Set("sample_rate", fmt.Sprintf("%d", s.sampleRate))

	if s.language != "" {
		params.Set("language-code", s.language)
	}
	// mode is only meaningful for saaras:v3
	if s.mode != "" && s.model == "saaras:v3" {
		params.Set("mode", s.mode)
	}
	// prompt is only meaningful for saaras:v2.5 (deprecated; prefer saaras:v3)
	if s.prompt != "" && s.model == "saaras:v2.5" {
		params.Set("prompt", s.prompt)
	}
	// Send vad_signals only when explicitly set — nil means "let server decide".
	if s.vadSignals != nil {
		if *s.vadSignals {
			params.Set("vad_signals", "true")
		} else {
			params.Set("vad_signals", "false")
		}
	}
	if s.highVADSensitivity != nil {
		if *s.highVADSensitivity {
			params.Set("high_vad_sensitivity", "true")
		} else {
			params.Set("high_vad_sensitivity", "false")
		}
	}
	// flush_signal is auto-enabled when not using Sarvam VAD so the pipeline
	// can flush transcription at speech boundaries (matches pipecat behaviour).
	if s.vadSignals == nil || !*s.vadSignals {
		params.Set("flush_signal", "true")
	}
	params.Set("input_audio_codec", s.encoding)

	base := sarvamSTTURL
	if s.useTranslateURL {
		base = sarvamSTTTranslateURL
	}
	wsURL := fmt.Sprintf("%s?%s", base, params.Encode())

	s.log.Info("Connecting language=%s codec=%s model=%s", s.language, s.encoding, s.model)

	header := http.Header{
		"api-subscription-key": []string{s.apiKey},
	}

	conn, _, err := websocket.DefaultDialer.DialContext(s.ctx, wsURL, header)
	if err != nil {
		return fmt.Errorf("sarvam dial: %w", err)
	}

	// Each connection gets its own cancellable context so keepaliveTask exits
	// when this specific connection is torn down, preventing goroutine accumulation
	// across reconnects.
	connCtx, connCancel := context.WithCancel(s.ctx)

	s.connMu.Lock()
	s.conn = conn
	s.connCancel = connCancel
	s.connMu.Unlock()

	// readWG is incremented inside connectMu so it can never race with the
	// readWG.Wait() call inside disconnect() (which also holds connectMu).
	s.readWG.Add(1)
	go s.receiveTranscriptions(conn)
	if s.keepaliveInterval > 0 {
		go s.keepaliveTask(conn, connCtx)
	}

	s.log.Info("Connected")
	return nil
}

// disconnect tears down the active WebSocket and waits for the receive goroutine
// to exit.  It holds connectMu so that concurrent connect() calls cannot call
// readWG.Add while readWG.Wait is in progress.
func (s *STTService) disconnect() {
	s.connectMu.Lock()
	defer s.connectMu.Unlock()

	s.connMu.Lock()
	conn := s.conn
	cancel := s.connCancel
	s.conn = nil
	s.connCancel = nil
	s.connMu.Unlock()

	// Cancel the per-connection context first so keepaliveTask exits promptly.
	if cancel != nil {
		cancel()
	}

	if conn != nil {
		_ = conn.WriteControl(
			websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
			time.Now().Add(100*time.Millisecond),
		)
		conn.Close()
	}

	// Wait for receiveTranscriptions to finish.  This is safe because:
	//  • readWG.Add is called inside connectMu (in connect()).
	//  • We hold connectMu here, so no concurrent Add can race with Wait.
	s.readWG.Wait()
}

// Cleanup closes the WebSocket and waits for goroutines to finish.
func (s *STTService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	s.disconnect()
	return nil
}

// HandleFrame dispatches pipeline frames.
//
//   - StartFrame  → eager Initialize (connect before audio arrives)
//   - AudioFrame  → base64-encode and send as JSON to Sarvam
//   - EndFrame    → Cleanup
//   - all others  → pass through
func (s *STTService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.StartFrame:
		// Guard against duplicate StartFrames — do not re-initialize if already up.
		s.connMu.RLock()
		initialized := s.conn != nil
		s.connMu.RUnlock()

		if !initialized {
			s.log.Info("StartFrame — connecting eagerly")
			if err := s.Initialize(ctx); err != nil {
				s.log.Error("Initialize failed: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		}
		// Emit TTFS metadata so turn-detection can auto-tune its silence threshold.
		s.PushFrame(frames.NewSTTMetadataFrame("sarvam", SarvamTTFSP99), frames.Downstream)
		return s.PushFrame(f, direction)

	case *frames.EndFrame:
		s.log.Info("EndFrame — cleaning up")
		if err := s.Cleanup(); err != nil {
			s.log.Warn("Cleanup error: %v", err)
		}
		return s.PushFrame(f, direction)

	case *frames.AudioFrame:
		return s.handleAudio(f, direction)

	case *frames.InterruptionFrame:
		if err := s.Flush(); err != nil {
			s.log.Warn("Flush on interruption failed: %v", err)
		}
		return s.PushFrame(f, direction)

	case *frames.UserStoppedSpeakingFrame:
		// Only flush via external VAD when Sarvam VAD is not active — if Sarvam
		// VAD is on, the server manages its own segment boundaries.
		if s.vadSignals == nil || !*s.vadSignals {
			if err := s.Flush(); err != nil {
				s.log.Warn("Flush on user stopped speaking failed: %v", err)
			}
		}
		return s.PushFrame(f, direction)
	}

	return s.PushFrame(frame, direction)
}

// Flush sends a flush signal to Sarvam, forcing finalization of any buffered
// audio into a transcription segment. No-op when disconnected. Requires
// flush_signal=true in the connect URL (auto-enabled when VAD signals are off).
func (s *STTService) Flush() error {
	s.connMu.RLock()
	conn := s.conn
	s.connMu.RUnlock()
	if conn == nil {
		return nil
	}
	s.writeMu.Lock()
	err := conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"flush"}`))
	s.writeMu.Unlock()
	return err
}

// audioDataMsg is the nested audio object within audioMsg.
type audioDataMsg struct {
	Data       string `json:"data"`
	SampleRate int    `json:"sample_rate"`
	Encoding   string `json:"encoding"`
}

// audioMsg is the JSON payload for each audio chunk sent to Sarvam.
// Sarvam expects: {"audio": {"data": "<b64>", "encoding": "...", "sample_rate": N}}
type audioMsg struct {
	Audio audioDataMsg `json:"audio"`
}

func (s *STTService) handleAudio(frame *frames.AudioFrame, direction frames.FrameDirection) error {
	// While a reconnect is in progress, drop frames silently to avoid ~50/sec
	// log messages from the write-failure path.
	if s.connDropped.Load() {
		return s.PushFrame(frame, direction)
	}

	s.connMu.RLock()
	conn := s.conn
	s.connMu.RUnlock()

	if conn == nil {
		// Shouldn't happen after eager init, but guard defensively.
		s.log.Warn("conn nil on AudioFrame — dropping")
		return s.PushFrame(frame, direction)
	}

	data, err := s.marshalAudio(frame.Data)
	if err != nil {
		return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
	}

	s.writeMu.Lock()
	writeErr := conn.WriteMessage(websocket.TextMessage, data)
	s.writeMu.Unlock()

	if writeErr != nil {
		// ErrCloseSent means gorilla already acknowledged a server-initiated close
		// frame. The connection is in a terminal state — reconnecting immediately
		// will hit the same server rejection (rate limit, 1003, etc.) and loop.
		if errors.Is(writeErr, websocket.ErrCloseSent) {
			s.log.Error("Server closed connection, not reconnecting: %v", writeErr)
			s.connDropped.Store(true)
			s.disconnect()
			return s.PushFrame(frames.NewErrorFrame(fmt.Errorf("sarvam STT: server closed connection: %w", writeErr)), frames.Upstream)
		}

		s.connDropped.Store(true)
		s.log.Warn("Write failed, disconnecting: %v", writeErr)
		s.disconnect()
		return s.PushFrame(frames.NewErrorFrame(writeErr), frames.Upstream)
	}

	// Always pass AudioFrame downstream for audio-based interruption detection.
	return s.PushFrame(frame, direction)
}

func (s *STTService) marshalAudio(raw []byte) ([]byte, error) {
	msg := audioMsg{
		Audio: audioDataMsg{
			Data:       base64.StdEncoding.EncodeToString(raw),
			Encoding:   "audio/wav", // fixed — Sarvam always expects "audio/wav" here; input_audio_codec URL param controls server-side decoding
			SampleRate: s.sampleRate,
		},
	}
	return json.Marshal(msg)
}

// --- server → client message types ---

type sarvamMsg struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

type sarvamEventData struct {
	SignalType string `json:"signal_type"` // "START_SPEECH" | "END_SPEECH"
}

type sarvamTranscriptData struct {
	Transcript   string `json:"transcript"`
	LanguageCode string `json:"language_code"`
}

// receiveTranscriptions reads messages from myConn until the connection closes.
// myConn is captured as a parameter so a stale goroutine from a previous
// connection cannot interfere with a freshly established one.
func (s *STTService) receiveTranscriptions(myConn *websocket.Conn) {
	defer s.readWG.Done()

	for {
		_, raw, err := myConn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
				strings.Contains(err.Error(), "use of closed network connection") {
				s.log.Debug("Connection closed normally")
				return
			}

			// Only push an error frame if this goroutine's conn is still the active
			// one — prevents a stale goroutine from injecting errors after a
			// successful reconnect.
			s.connMu.RLock()
			stillActive := s.conn == myConn
			s.connMu.RUnlock()

			if !stillActive {
				// Stale goroutine — don't push error frame, but log the close
				// code so we can diagnose what the server sent on the first close.
				s.log.Debug("Stale goroutine saw close (conn already replaced): %v", err)
				return
			}
			s.log.Error("Read error: %v", err)
			s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			return
		}

		var msg sarvamMsg
		if err := json.Unmarshal(raw, &msg); err != nil {
			s.log.Warn("Failed to parse server message: %v", err)
			continue
		}

		switch msg.Type {
		case "events":
			// Ignore server VAD events unless the client explicitly enabled them.
			if s.vadSignals == nil || !*s.vadSignals {
				continue
			}
			var event sarvamEventData
			if err := json.Unmarshal(msg.Data, &event); err != nil {
				s.log.Warn("Failed to parse event data: %v", err)
				continue
			}
			switch event.SignalType {
			case "START_SPEECH":
				s.log.Debug("VAD: user started speaking")
				s.PushFrame(frames.NewUserStartedSpeakingFrame(), frames.Upstream)
			case "END_SPEECH":
				s.log.Debug("VAD: user stopped speaking")
				s.PushFrame(frames.NewUserStoppedSpeakingFrame(), frames.Upstream)
			}

		case "data":
			var td sarvamTranscriptData
			if err := json.Unmarshal(msg.Data, &td); err != nil {
				s.log.Warn("Failed to parse transcript data: %v", err)
				continue
			}
			t := strings.TrimSpace(td.Transcript)
			if t == "" {
				continue
			}
			s.log.Debug("Transcript (lang=%s): %s", td.LanguageCode, t)
			tf := frames.NewTranscriptionFrame(t, true /* Sarvam emits final segments */)
			if td.LanguageCode != "" {
				tf.Language = td.LanguageCode
			}
			s.PushFrame(tf, frames.Downstream)
		}
	}
}

// keepaliveTask sends silent audio frames at KeepaliveInterval to prevent the
// server from closing the WebSocket due to inactivity. Only started when
// KeepaliveInterval > 0. Exits when the per-connection context is cancelled.
func (s *STTService) keepaliveTask(conn *websocket.Conn, connCtx context.Context) {
	ticker := time.NewTicker(s.keepaliveInterval)
	defer ticker.Stop()

	// 100 ms of silence per ping — small enough to be harmless, large enough
	// that the server sees activity.
	silenceSamples := s.sampleRate / 10 // 100 ms
	silence := make([]byte, silenceSamples*2) // 16-bit PCM, mono

	payload, err := s.marshalAudio(silence)
	if err != nil {
		s.log.Error("Failed to build keepalive payload: %v", err)
		return
	}

	for {
		select {
		case <-connCtx.Done():
			return
		case <-ticker.C:
			if s.connDropped.Load() {
				continue
			}

			s.writeMu.Lock()
			err := conn.WriteMessage(websocket.TextMessage, payload)
			s.writeMu.Unlock()

			if err != nil {
				s.log.Warn("Keepalive write failed: %v", err)
				return
			}
		}
	}
}

// languageCodes maps two-letter ISO 639-1 codes to Sarvam BCP-47 codes.
var languageCodes = map[string]string{
	"en": "en-IN",
	"hi": "hi-IN",
	"bn": "bn-IN",
	"gu": "gu-IN",
	"kn": "kn-IN",
	"ml": "ml-IN",
	"mr": "mr-IN",
	"ta": "ta-IN",
	"te": "te-IN",
	"pa": "pa-IN",
	"or": "od-IN",
	"as": "as-IN",
}
