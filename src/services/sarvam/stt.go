package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	sarvamSTTURL          = "wss://api.sarvam.ai/speech-to-text-streaming/transcribe/ws"
	sarvamSTTTranslateURL = "wss://api.sarvam.ai/speech-to-text-translate-streaming/transcribe/ws"

	defaultModel      = "saaras:v3"
	defaultEncoding   = "pcm_s16le"
	defaultSampleRate = 16000
	defaultMode       = "transcribe"

	// SarvamTTFSP99 is the estimated P99 time-to-first-segment latency used by
	// turn-detection auto-tuning.
	SarvamTTFSP99 = 500 * time.Millisecond

	keepalivePeriod = 5 * time.Second
)

// STTConfig holds construction-time configuration for the Sarvam STT service.
type STTConfig struct {
	// APIKey is the Sarvam subscription key (sent as api-subscription-key header).
	APIKey string

	// Model selects the Sarvam STT model.
	//   "saaras:v3"    — recommended; auto-detects language; supports Mode param.
	//   "saarika:v2.5" — requires Language; does not support Mode or prompt.
	//   "saaras:v2.5"  — translate endpoint; supports prompt; auto-detects language.
	// Defaults to "saaras:v3".
	Model string

	// Language is the BCP-47 language code, e.g. "en-IN", "hi-IN".
	// Required for saarika:v2.5; optional for saaras:v3 (auto-detect).
	Language string

	// Mode controls the output format (saaras:v3 only).
	// One of: "transcribe" (default), "translate", "verbatim", "translit", "codemix".
	Mode string

	// Encoding is the input audio codec arriving in AudioFrame.Data.
	// One of: "pcm_s16le" (default), "wav", "alaw"/"PCMA", "mulaw"/"ulaw"/"PCMU".
	//
	// Telephony codecs ("alaw", "mulaw") are decoded to pcm_s16le internally before
	// transmission — Sarvam's streaming API only accepts raw PCM and WAV.  When a
	// telephony codec is specified and SampleRate is 0, SampleRate defaults to 8000.
	Encoding string

	// SampleRate is the input audio sample rate in Hz.  8000 or 16000 (default).
	// Must match the actual audio source — sent in both the connect URL and each
	// audio message.
	SampleRate int

	// VADSignals, when true, enables server-side VAD events.  The service emits
	// UserStartedSpeakingFrame / UserStoppedSpeakingFrame upstream on each event.
	VADSignals bool

	// HighVADSensitivity enables Sarvam's high-sensitivity VAD mode, which fires
	// on shorter pauses (useful with high_vad_sensitivity=0.5s vs 1s).
	HighVADSensitivity bool

	// FlushOnSilence enables the server-side flush_signal feature so transcription
	// segments are finalized cleanly at speech boundaries.
	FlushOnSilence bool
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
	encoding           string
	sampleRate         int
	vadSignals         bool
	highVADSensitivity bool
	flushOnSilence     bool
	useTranslateURL    bool // true when model == "saaras:v2.5"
	needsG711Decode    bool // true when input is alaw/mulaw — convert to pcm_s16le before send

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
	needsG711Decode := isTelephonyCodec(encoding)
	sampleRate := config.SampleRate
	if sampleRate == 0 {
		if needsG711Decode {
			sampleRate = 8000 // telephony is always 8 kHz
		} else {
			sampleRate = defaultSampleRate
		}
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
		encoding:           encoding,
		sampleRate:         sampleRate,
		vadSignals:         config.VADSignals,
		highVADSensitivity: config.HighVADSensitivity,
		flushOnSilence:     config.FlushOnSilence,
		useTranslateURL:    model == "saaras:v2.5",
		needsG711Decode:    needsG711Decode,
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
		params.Set("language_code", s.language)
	}
	// mode is only meaningful for saaras:v3
	if s.mode != "" && s.model == "saaras:v3" {
		params.Set("mode", s.mode)
	}
	if s.vadSignals {
		params.Set("vad_signals", "true")
	}
	if s.highVADSensitivity {
		params.Set("high_vad_sensitivity", "true")
	}
	if s.flushOnSilence {
		params.Set("flush_signal", "true")
	}

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
	go s.keepaliveTask(conn, connCtx)

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

func (s *STTService) reconnect() error {
	s.log.Warn("Reconnecting")
	s.disconnect()
	return s.connect()
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
	}

	return s.PushFrame(frame, direction)
}

// audioMsg is the JSON payload for each audio chunk sent to Sarvam.
type audioMsg struct {
	Audio      string `json:"audio"`
	Encoding   string `json:"encoding"`
	SampleRate int    `json:"sample_rate"`
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
		s.connDropped.Store(true)
		s.log.Warn("Write failed, reconnecting (frames silently dropped until ready): %v", writeErr)

		if reconnErr := s.reconnect(); reconnErr != nil {
			s.log.Error("Reconnect failed: %v", reconnErr)
			return s.PushFrame(frames.NewErrorFrame(reconnErr), frames.Upstream)
		}
		s.connDropped.Store(false)
		s.log.Info("Reconnected")

		// Retry the triggering frame once on the new connection.
		s.connMu.RLock()
		conn = s.conn
		s.connMu.RUnlock()

		s.writeMu.Lock()
		retryErr := conn.WriteMessage(websocket.TextMessage, data)
		s.writeMu.Unlock()

		if retryErr != nil {
			s.connDropped.Store(true)
			s.log.Error("Write still failed after reconnect: %v", retryErr)
			return s.PushFrame(frames.NewErrorFrame(retryErr), frames.Upstream)
		}
	}

	// Always pass AudioFrame downstream for audio-based interruption detection.
	return s.PushFrame(frame, direction)
}

func (s *STTService) marshalAudio(raw []byte) ([]byte, error) {
	pcmBytes := raw
	wireEncoding := "audio/" + s.encoding

	if s.needsG711Decode {
		// Sarvam's streaming API only accepts raw PCM and WAV — alaw/mulaw must be
		// decoded to pcm_s16le before transmission.  Conversion uses the in-tree
		// G.711 tables in src/audio/converter.go; no external dependency required.
		var pcm []int16
		switch normalizeG711Codec(s.encoding) {
		case "alaw":
			pcm = audio.AlawToPCM(raw)
		default: // mulaw / ulaw / PCMU
			pcm = audio.MulawToPCM(raw)
		}
		pcmBytes = audio.PCMToBytes(pcm)
		wireEncoding = "audio/pcm_s16le"
	}

	msg := audioMsg{
		Audio:      base64.StdEncoding.EncodeToString(pcmBytes),
		Encoding:   wireEncoding,
		SampleRate: s.sampleRate,
	}
	return json.Marshal(msg)
}

// makeSilence builds a correctly-sized and correctly-valued silence payload for
// the given encoding.
//   - pcm_s16le: 2 bytes/sample, all zeros
//   - alaw:      1 byte/sample, 0xD5 (ITU silence value)
//   - mulaw:     1 byte/sample, 0xFF (ITU silence value)
func makeSilence(samples int, encoding string, isG711 bool) []byte {
	if !isG711 {
		return make([]byte, samples*2) // 16-bit PCM — zero = silence
	}
	var silenceByte byte
	if normalizeG711Codec(encoding) == "alaw" {
		silenceByte = 0xD5 // A-law silence
	} else {
		silenceByte = 0xFF // μ-law silence
	}
	buf := make([]byte, samples) // G.711 — 1 byte/sample
	for i := range buf {
		buf[i] = silenceByte
	}
	return buf
}

// isTelephonyCodec reports whether the codec name is a G.711 telephony codec
// (alaw/PCMA or mulaw/ulaw/PCMU).
func isTelephonyCodec(codec string) bool {
	switch strings.ToLower(codec) {
	case "alaw", "pcma", "mulaw", "ulaw", "pcmu":
		return true
	}
	return false
}

// normalizeG711Codec maps all alaw/mulaw variants to a canonical two-value set.
func normalizeG711Codec(codec string) string {
	switch strings.ToLower(codec) {
	case "alaw", "pcma":
		return "alaw"
	default:
		return "mulaw"
	}
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

			if stillActive {
				s.log.Error("Read error: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
			return
		}

		var msg sarvamMsg
		if err := json.Unmarshal(raw, &msg); err != nil {
			s.log.Warn("Failed to parse server message: %v", err)
			continue
		}

		switch msg.Type {
		case "events":
			if !s.vadSignals {
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

// keepaliveTask sends silent PCM frames at keepalivePeriod intervals to prevent
// the Sarvam server from closing the WebSocket due to inactivity.
//
// It receives a connection-scoped context (connCtx) that is cancelled by
// disconnect() when this specific connection is torn down.  This ensures each
// keepaliveTask goroutine exits with its own connection rather than
// accumulating across reconnects.
func (s *STTService) keepaliveTask(conn *websocket.Conn, connCtx context.Context) {
	ticker := time.NewTicker(keepalivePeriod)
	defer ticker.Stop()

	// 100 ms of silence per ping — small enough to be harmless, large enough
	// that the server sees activity.
	silenceSamples := s.sampleRate / 10 // 100 ms
	silence := makeSilence(silenceSamples, s.encoding, s.needsG711Decode)

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
