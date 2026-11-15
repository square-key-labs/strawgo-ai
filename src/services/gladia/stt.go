package gladia

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

const (
	DefaultURL       = "https://api.gladia.io/v2/live"
	DefaultModel     = "solaria-1"
	DefaultReplay    = 10 * time.Second
	keepaliveIdle    = 20 * time.Second
	chunkPrefix      = `{"type":"audio_chunk","data":{"chunk":"`
	chunkSuffix      = `"}}`
	initTimeout      = 10 * time.Second
	errorBodyLimit   = 4096
	defaultMaxNoEndp = 5
)

var (
	stopRecording = []byte(`{"type":"stop_recording"}`)
	emptyChunk    = []byte(chunkPrefix + chunkSuffix)
)

type STTConfig struct {
	APIKey                            string
	URL                               string
	Region                            string
	Model                             string
	Language                          string
	Encoding                          string
	BitDepth                          int
	Channels                          int
	SampleRate                        int
	CustomMetadata                    map[string]any
	Endpointing                       *float64
	MaximumDurationWithoutEndpointing *int
	LanguageConfig                    *LanguageConfig
	PreProcessing                     *PreProcessingConfig
	RealtimeProcessing                *RealtimeProcessingConfig
	MessagesConfig                    *MessagesConfig
	EnableVAD                         bool
	DisableInterruptions              bool
	ReplayBuffer                      time.Duration
	HTTPClient                        *http.Client
}

type STTService struct {
	*stt.Base
	cfg    STTConfig
	client *http.Client

	session atomic.Pointer[reconnect.Session]
	sink    stt.Sink
	ctx     context.Context

	mu          sync.Mutex
	sessionURL  string
	sessionID   string
	opened      bool
	speaking    bool
	ring        []byte
	head        int
	size        int
	acked       int64
	replayBytes int
}

func NewSTTService(config STTConfig) *STTService {
	if config.URL == "" {
		config.URL = DefaultURL
	}
	if config.Model == "" {
		config.Model = DefaultModel
	}
	if config.Encoding == "" {
		config.Encoding = "wav/pcm"
	}
	if config.BitDepth <= 0 {
		config.BitDepth = 16
	}
	if config.Channels <= 0 {
		config.Channels = 1
	}
	if config.MaximumDurationWithoutEndpointing == nil {
		v := defaultMaxNoEndp
		config.MaximumDurationWithoutEndpointing = &v
	}
	if config.ReplayBuffer == 0 {
		config.ReplayBuffer = DefaultReplay
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: initTimeout}
	}
	s := &STTService{cfg: config, client: client}
	s.Base = stt.NewStreaming("GladiaSTT", s, stt.Options{
		Provider:   "gladia",
		Settings:   frames.STTSettings{Model: config.Model, Language: config.Language},
		SampleRate: config.SampleRate,
		TTFSP99:    1490 * time.Millisecond,
	})
	return s
}

func (s *STTService) initPayload() map[string]any {
	settings := s.Settings()
	meta := map[string]any{}
	for k, v := range s.cfg.CustomMetadata {
		meta[k] = v
	}
	meta["strawgo"] = "1"
	p := map[string]any{
		"encoding":        s.cfg.Encoding,
		"bit_depth":       s.cfg.BitDepth,
		"sample_rate":     s.SampleRate(),
		"channels":        s.cfg.Channels,
		"model":           settings.Model,
		"custom_metadata": meta,
	}
	if s.cfg.Endpointing != nil {
		p["endpointing"] = *s.cfg.Endpointing
	}
	p["maximum_duration_without_endpointing"] = *s.cfg.MaximumDurationWithoutEndpointing
	lc := s.cfg.LanguageConfig
	if lc == nil && settings.Language != "" {
		lang := strings.ToLower(settings.Language)
		if i := strings.IndexAny(lang, "-_"); i > 0 {
			lang = lang[:i]
		}
		lc = &LanguageConfig{Languages: []string{lang}}
	}
	if lc != nil {
		p["language_config"] = lc
	}
	if s.cfg.PreProcessing != nil {
		p["pre_processing"] = s.cfg.PreProcessing
	}
	if s.cfg.RealtimeProcessing != nil {
		p["realtime_processing"] = s.cfg.RealtimeProcessing
	}
	if s.cfg.MessagesConfig != nil {
		p["messages_config"] = s.cfg.MessagesConfig
	}
	return p
}

func (s *STTService) initSession() (string, string, error) {
	body, err := json.Marshal(s.initPayload())
	if err != nil {
		return "", "", err
	}
	endpoint := s.cfg.URL
	if s.cfg.Region != "" {
		endpoint += "?" + url.Values{"region": {s.cfg.Region}}.Encode()
	}
	req, err := http.NewRequestWithContext(s.ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", "", err
	}
	req.Header.Set("X-Gladia-Key", s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		return "", "", fmt.Errorf("gladia init HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var out struct {
		ID  string `json:"id"`
		URL string `json:"url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", "", fmt.Errorf("decode gladia init response: %w", err)
	}
	if out.URL == "" {
		return "", "", errors.New("gladia init response has no session url")
	}
	return out.URL, out.ID, nil
}

func (s *STTService) endpoint() (string, http.Header) {
	s.mu.Lock()
	sessionURL, opened := s.sessionURL, s.opened
	s.opened = false
	s.mu.Unlock()
	if sessionURL != "" && opened {
		return sessionURL, nil
	}
	u, id, err := s.initSession()
	if err != nil {
		s.sink.Error(err, false)
		return "", nil
	}
	s.mu.Lock()
	s.sessionURL, s.sessionID = u, id
	s.mu.Unlock()
	return u, nil
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	s.sink = sink
	s.ctx = ctx
	s.replayBytes = int(s.cfg.ReplayBuffer.Seconds() * float64(s.SampleRate()*2*s.cfg.Channels))
	if s.cfg.EnableVAD {
		meta := frames.NewServiceMetadataFrame(s.Name())
		strategies := turns.UserTurnStrategies{
			StartStrategies: []user_start.UserTurnStartStrategy{proposedStart{interrupt: !s.cfg.DisableInterruptions}},
			StopStrategies:  []user_stop.UserTurnStopStrategy{proposedStop{}},
		}
		meta.UserTurnStrategies = &strategies
		sink.Push(meta, frames.Downstream)
	}
	session := reconnect.New(reconnect.Config{
		Name:          "GladiaSTT",
		Endpoint:      s.endpoint,
		OnOpen:        s.onOpen,
		OnMessage:     func(_ int, data []byte) { s.onMessage(data) },
		OnError:       sink.Error,
		CloseMessage:  &reconnect.Message{Type: websocket.TextMessage, Data: stopRecording},
		KeepaliveIdle: keepaliveIdle,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: emptyChunk}, true
		},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) onOpen(w reconnect.Writer) error {
	s.mu.Lock()
	s.opened = true
	s.acked = 0
	var replay []byte
	if s.size > 0 {
		replay = make([]byte, s.size)
		s.copyRingLocked(replay)
	}
	s.mu.Unlock()
	if replay == nil {
		return nil
	}
	return w.WriteMessage(websocket.TextMessage, chunk(replay))
}

func chunk(pcm []byte) []byte {
	buf := make([]byte, 0, len(chunkPrefix)+base64.StdEncoding.EncodedLen(len(pcm))+len(chunkSuffix))
	buf = append(buf, chunkPrefix...)
	buf = base64.StdEncoding.AppendEncode(buf, pcm)
	return append(buf, chunkSuffix...)
}

func (s *STTService) copyRingLocked(dst []byte) {
	n := copy(dst, s.ring[s.head:min(len(s.ring), s.head+s.size)])
	if n < s.size {
		copy(dst[n:], s.ring[:s.size-n])
	}
}

func (s *STTService) remember(audio []byte) {
	if s.replayBytes <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ring == nil {
		s.ring = make([]byte, s.replayBytes)
	}
	if len(audio) >= len(s.ring) {
		copy(s.ring, audio[len(audio)-len(s.ring):])
		s.head, s.size = 0, len(s.ring)
		return
	}
	if over := s.size + len(audio) - len(s.ring); over > 0 {
		s.head = (s.head + over) % len(s.ring)
		s.size -= over
	}
	tail := (s.head + s.size) % len(s.ring)
	n := copy(s.ring[tail:], audio)
	copy(s.ring, audio[n:])
	s.size += len(audio)
}

func (s *STTService) acknowledge(end int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if end <= s.acked {
		return
	}
	drop := int(min(end-s.acked, int64(s.size)))
	s.acked = end
	if drop > 0 {
		s.head = (s.head + drop) % len(s.ring)
		s.size -= drop
	}
}

func (s *STTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil {
		return
	}
	s.remember(audio)
	if session.Connected() {
		session.SendDroppable(websocket.TextMessage, chunk(audio))
	}
}

func (s *STTService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

type message struct {
	Type         string `json:"type"`
	Acknowledged bool   `json:"acknowledged"`
	Error        *struct {
		Message string `json:"message"`
	} `json:"error"`
	Data json.RawMessage `json:"data"`
}

type transcriptData struct {
	IsFinal   bool `json:"is_final"`
	Utterance struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	} `json:"utterance"`
}

type ackData struct {
	ByteRange []int64 `json:"byte_range"`
}

type translationData struct {
	OriginalLanguage    string `json:"original_language"`
	TranslatedUtterance struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	} `json:"translated_utterance"`
}

func errText(e *struct {
	Message string `json:"message"`
}) string {
	if e == nil || e.Message == "" {
		return "unknown error"
	}
	return e.Message
}

func (s *STTService) onMessage(raw []byte) {
	var msg message
	if err := json.Unmarshal(raw, &msg); err != nil {
		return
	}
	switch msg.Type {
	case "audio_chunk":
		if !msg.Acknowledged {
			s.sink.Error(errors.New("gladia rejected an audio chunk: "+errText(msg.Error)), false)
			return
		}
		var d ackData
		if json.Unmarshal(msg.Data, &d) == nil && len(d.ByteRange) == 2 {
			s.acknowledge(d.ByteRange[1])
		}
	case "transcript":
		var d transcriptData
		if json.Unmarshal(msg.Data, &d) != nil {
			return
		}
		s.sink.Transcript(stt.Transcript{Text: d.Utterance.Text, Final: d.IsFinal, Language: d.Utterance.Language})
	case "translation":
		if msg.Error != nil || len(msg.Data) == 0 || string(msg.Data) == "null" {
			s.sink.Error(errors.New("gladia translation addon error: "+errText(msg.Error)), false)
			return
		}
		var d translationData
		if json.Unmarshal(msg.Data, &d) != nil {
			return
		}
		if d.TranslatedUtterance.Language != d.OriginalLanguage && d.TranslatedUtterance.Text != "" {
			s.sink.Push(frames.NewTranslationFrame(d.TranslatedUtterance.Text, "", d.TranslatedUtterance.Language), frames.Downstream)
		}
	case "speech_start":
		s.speech(true)
	case "speech_end":
		s.speech(false)
	}
}

func (s *STTService) speech(started bool) {
	if !s.cfg.EnableVAD {
		return
	}
	s.mu.Lock()
	changed := s.speaking != started
	s.speaking = started
	s.mu.Unlock()
	if !changed {
		return
	}
	if started {
		s.sink.Push(frames.NewProposedUserStartedSpeakingFrame(), frames.Downstream)
		s.sink.Push(frames.NewProposedUserStartedSpeakingFrame(), frames.Upstream)
		return
	}
	s.sink.Push(frames.NewProposedUserStoppedSpeakingFrame(), frames.Downstream)
	s.sink.Push(frames.NewProposedUserStoppedSpeakingFrame(), frames.Upstream)
}

type proposedStart struct{ interrupt bool }

func (proposedStart) ShouldStart(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStartedSpeakingFrame)
	return ok
}

func (p proposedStart) EnableInterruptions() bool { return p.interrupt }

func (proposedStart) Reset() {}

type proposedStop struct{}

func (proposedStop) ShouldStop(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStoppedSpeakingFrame)
	return ok
}

func (proposedStop) Reset() {}
