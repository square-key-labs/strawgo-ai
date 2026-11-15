package serializers

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const GenesysProtocolVersion = "2"

type GenesysChannel string

const (
	GenesysChannelExternal GenesysChannel = "external"
	GenesysChannelInternal GenesysChannel = "internal"
	GenesysChannelBoth     GenesysChannel = "both"
)

type GenesysConfig struct {
	SampleRate         int
	Channel            GenesysChannel
	SupportedLanguages []string
	SelectedLanguage   string
	StartPaused        bool
	OnOpen             func(msg map[string]any)
	OnClose            func(msg map[string]any)
	OnPaused           func(msg map[string]any)
	OnResumed          func(msg map[string]any)
	OnUpdate           func(msg map[string]any)
	OnError            func(msg map[string]any)
	OnDTMF             func(msg map[string]any)
}

type GenesysAudioHookSerializer struct {
	cfg  GenesysConfig
	rate int

	open   atomic.Bool
	paused atomic.Bool

	mu             sync.Mutex
	channel        GenesysChannel
	sessionID      string
	conversationID string
	clientSeq      int64
	serverSeq      int64
	position       time.Duration
	participant    map[string]any
	inputVars      map[string]any
	outputVars     map[string]any

	wire wireAudio
	mono []byte
}

type genesysMessage struct {
	Version    string         `json:"version"`
	Type       string         `json:"type"`
	Seq        int64          `json:"seq"`
	ClientSeq  int64          `json:"clientseq"`
	ID         string         `json:"id"`
	Position   string         `json:"position,omitempty"`
	Parameters map[string]any `json:"parameters"`
}

func NewGenesysAudioHookSerializer(cfg GenesysConfig) *GenesysAudioHookSerializer {
	if cfg.SampleRate <= 0 {
		cfg.SampleRate = 8000
	}
	if cfg.Channel == "" {
		cfg.Channel = GenesysChannelExternal
	}
	return &GenesysAudioHookSerializer{
		cfg:       cfg,
		rate:      cfg.SampleRate,
		channel:   cfg.Channel,
		sessionID: newUUID(),
		wire:      newWireAudio(audio.CodecMulaw, audio.CodecMulaw, cfg.SampleRate),
	}
}

func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func (s *GenesysAudioHookSerializer) WireSampleRates() (int, int) { return s.rate, s.rate }

func (s *GenesysAudioHookSerializer) NativeAudioFormat() (string, int) {
	return audio.CodecMulaw, s.rate
}

func (s *GenesysAudioHookSerializer) SetNativeAudioInput(enabled bool) { s.wire.native = enabled }

func (s *GenesysAudioHookSerializer) SessionID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sessionID
}

func (s *GenesysAudioHookSerializer) ConversationID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conversationID
}

func (s *GenesysAudioHookSerializer) Participant() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.participant
}

func (s *GenesysAudioHookSerializer) InputVariables() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inputVars
}

func (s *GenesysAudioHookSerializer) OutputVariables() map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.outputVars
}

func (s *GenesysAudioHookSerializer) SetOutputVariables(vars map[string]any) {
	s.mu.Lock()
	s.outputVars = vars
	s.mu.Unlock()
}

func (s *GenesysAudioHookSerializer) IsOpen() bool { return s.open.Load() }

func (s *GenesysAudioHookSerializer) IsPaused() bool { return s.paused.Load() }

func (s *GenesysAudioHookSerializer) CallInfo() CallInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	info := CallInfo{Provider: "genesys", StreamID: s.sessionID, CallID: s.conversationID, Encoding: audio.CodecMulaw, SampleRate: s.rate}
	if ani, ok := s.participant["ani"].(string); ok {
		info.From = ani
	}
	if dnis, ok := s.participant["dnis"].(string); ok {
		info.To = dnis
	}
	return info
}

func (s *GenesysAudioHookSerializer) message(typ string, params map[string]any, withPosition bool) genesysMessage {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.serverSeq++
	m := genesysMessage{Version: GenesysProtocolVersion, Type: typ, Seq: s.serverSeq, ClientSeq: s.clientSeq, ID: s.sessionID, Parameters: params}
	if withPosition {
		m.Position = "PT" + strconv.FormatFloat(s.position.Seconds(), 'f', 3, 64) + "S"
	}
	if m.Parameters == nil {
		m.Parameters = map[string]any{}
	}
	return m
}

func (s *GenesysAudioHookSerializer) openedResponse() genesysMessage {
	s.mu.Lock()
	channel := s.channel
	s.mu.Unlock()
	channels := []string{string(channel)}
	if channel == GenesysChannelBoth {
		channels = []string{"external", "internal"}
	}
	params := map[string]any{
		"startPaused": s.cfg.StartPaused,
		"media":       []map[string]any{{"type": "audio", "format": "PCMU", "channels": channels, "rate": s.rate}},
	}
	if len(s.cfg.SupportedLanguages) > 0 {
		params["supportedLanguages"] = s.cfg.SupportedLanguages
	}
	if s.cfg.SelectedLanguage != "" {
		params["selectedLanguage"] = s.cfg.SelectedLanguage
	}
	m := s.message("opened", params, false)
	s.open.Store(true)
	s.paused.Store(s.cfg.StartPaused)
	return m
}

func (s *GenesysAudioHookSerializer) closedResponse() genesysMessage {
	var params map[string]any
	if vars := s.OutputVariables(); len(vars) > 0 {
		params = map[string]any{"outputVariables": vars}
	}
	m := s.message("closed", params, true)
	s.open.Store(false)
	return m
}

func (s *GenesysAudioHookSerializer) PauseRequest() any { return s.message("pause", nil, true) }

func (s *GenesysAudioHookSerializer) ResumeRequest() any { return s.message("resume", nil, true) }

func (s *GenesysAudioHookSerializer) BargeInEvent() any {
	return s.message("event", map[string]any{"entities": []map[string]any{{"type": "barge_in", "data": map[string]any{}}}}, true)
}

func (s *GenesysAudioHookSerializer) DisconnectMessage(reason, action string, outputVars map[string]any, info string) any {
	out := map[string]any{"action": action}
	for k, v := range outputVars {
		out[k] = v
	}
	params := map[string]any{"reason": reason, "outputVariables": out}
	if info != "" {
		params["info"] = info
	}
	return s.message("disconnect", params, true)
}

func (s *GenesysAudioHookSerializer) ErrorMessage(code int, msg string, retryable bool) any {
	return s.message("error", map[string]any{"code": code, "message": msg, "retryable": retryable}, true)
}

func (s *GenesysAudioHookSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	if !s.open.Load() || s.paused.Load() {
		return dst, nil
	}
	enc, err := s.wire.encodeOwned(pcm, sampleRate)
	if err != nil {
		return dst, err
	}
	return append(dst, Binary(enc)), nil
}

func (s *GenesysAudioHookSerializer) SerializeNativeAudio(dst []Message, data []byte) ([]Message, error) {
	if !s.open.Load() || s.paused.Load() {
		return dst, nil
	}
	return append(dst, Binary(data)), nil
}

func (s *GenesysAudioHookSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	switch f := frame.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		return appendJSON(dst, s.DisconnectMessage("completed", "transfer", s.OutputVariables(), ""))
	case *frames.InterruptionFrame:
		s.wire.reset()
		return appendJSON(dst, s.BargeInEvent())
	case *frames.OutputTransportMessageFrame:
		return s.passThrough(dst, f.Message)
	case *frames.OutputTransportMessageUrgentFrame:
		return s.passThrough(dst, f.Message)
	}
	if s.open.Load() && !s.paused.Load() {
		if out, ok, err := serializeAudioFrame(s, &s.wire.rs, dst, frame, s.rate); ok {
			return out, err
		}
	}
	return dst, nil
}

func (s *GenesysAudioHookSerializer) passThrough(dst []Message, msg any) ([]Message, error) {
	switch m := msg.(type) {
	case genesysMessage:
		return appendJSON(dst, m)
	case map[string]any:
		if _, ok := m["version"]; ok && !isRTVIMessage(m) {
			return appendJSON(dst, m)
		}
	}
	return dst, nil
}

func appendJSON(dst []Message, v any) ([]Message, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return dst, err
	}
	return append(dst, Message{Type: TextMessage, Data: data}), nil
}

func (s *GenesysAudioHookSerializer) Deserialize(msg Message) (frames.Frame, error) {
	if msg.Type == BinaryMessage {
		return s.deserializeAudio(msg.Data)
	}
	var m map[string]any
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("genesys: %w", err)
	}
	typ, _ := m["type"].(string)
	params, _ := m["parameters"].(map[string]any)
	s.mu.Lock()
	if seq, ok := m["seq"].(float64); ok {
		s.clientSeq = int64(seq)
	}
	if pos, ok := m["position"].(string); ok {
		s.position = parseISODuration(pos)
	}
	s.mu.Unlock()
	switch typ {
	case "open":
		s.handleOpen(m, params)
		call(s.cfg.OnOpen, m)
		return frames.NewOutputTransportMessageUrgentFrame(s.openedResponse()), nil
	case "close":
		s.open.Store(false)
		call(s.cfg.OnClose, m)
		return frames.NewOutputTransportMessageUrgentFrame(s.closedResponse()), nil
	case "ping":
		return frames.NewOutputTransportMessageUrgentFrame(s.message("pong", nil, true)), nil
	case "paused":
		s.paused.Store(true)
		call(s.cfg.OnPaused, m)
	case "resumed":
		s.paused.Store(false)
		call(s.cfg.OnResumed, m)
	case "update":
		if p, ok := params["participant"].(map[string]any); ok {
			s.mu.Lock()
			s.participant = p
			s.mu.Unlock()
		}
		call(s.cfg.OnUpdate, m)
	case "error":
		logger.Error("[Genesys] AudioHook error: %v - %v", params["code"], params["message"])
		call(s.cfg.OnError, m)
	case "dtmf":
		digit, _ := params["digit"].(string)
		call(s.cfg.OnDTMF, m)
		if digit == "" {
			return nil, nil
		}
		return dtmfFrame(digit), nil
	}
	return nil, nil
}

func call(fn func(map[string]any), m map[string]any) {
	if fn != nil {
		fn(m)
	}
}

func (s *GenesysAudioHookSerializer) handleOpen(m, params map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if id, ok := m["id"].(string); ok && id != "" {
		s.sessionID = id
	}
	s.conversationID, _ = params["conversationId"].(string)
	s.participant, _ = params["participant"].(map[string]any)
	s.inputVars, _ = params["inputVariables"].(map[string]any)
	media, _ := params["media"].([]any)
	if len(media) == 0 {
		return
	}
	first, _ := media[0].(map[string]any)
	channels, _ := first["channels"].([]any)
	var ext, in bool
	for _, c := range channels {
		switch c {
		case "external":
			ext = true
		case "internal":
			in = true
		}
	}
	switch {
	case ext && in:
		s.channel = GenesysChannelBoth
	case ext:
		s.channel = GenesysChannelExternal
	case in:
		s.channel = GenesysChannelInternal
	}
}

func (s *GenesysAudioHookSerializer) deserializeAudio(data []byte) (frames.Frame, error) {
	if !s.open.Load() || s.paused.Load() || len(data) == 0 {
		return nil, nil
	}
	s.mu.Lock()
	stereo := s.channel == GenesysChannelBoth
	s.mu.Unlock()
	if stereo {
		n := (len(data) + 1) / 2
		for i := 0; i < n; i++ {
			data[i] = data[2*i]
		}
		data = data[:n]
	}
	return s.wire.decode(data)
}

func parseISODuration(v string) time.Duration {
	if !strings.HasPrefix(v, "PT") || !strings.HasSuffix(v, "S") {
		return 0
	}
	secs, err := strconv.ParseFloat(v[2:len(v)-1], 64)
	if err != nil {
		return 0
	}
	return time.Duration(secs * float64(time.Second))
}
