package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

const (
	DefaultSTTModel          = "gpt-transcribe"
	DefaultSTTURL            = "https://api.openai.com/v1/audio/transcriptions"
	DefaultRealtimeSTTModel  = "gpt-realtime-whisper"
	DefaultRealtimeSTTURL    = "wss://api.openai.com/v1/realtime"
	RealtimeSTTSampleRate    = 24000
	STTExtraPrompt           = "prompt"
	STTExtraTemperature      = "temperature"
	STTExtraKeywords         = "keywords"
	RealtimeExtraNoise       = "noise_reduction"
	RealtimeExtraDelay       = "delay"
	realtimeAppendPrefix     = `{"type":"input_audio_buffer.append","audio":"`
	realtimeAppendSuffix     = `"}`
	realtimeConfirmTimeout   = 10 * time.Second
	openAIErrorBodyLimit     = 4096
	openAISTTDefaultLanguage = "en"
)

var realtimeCommit = []byte(`{"type":"input_audio_buffer.commit"}`)

type STTConfig struct {
	APIKey      string
	Model       string
	Language    string
	Prompt      string
	Temperature *float64
	Keywords    []string
	SampleRate  int
	BaseURL     string
	HTTPClient  *http.Client
}

type STTService struct {
	*stt.Base
	apiKey string
	url    string
	client *http.Client
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultSTTModel
	}
	language := config.Language
	if language == "" {
		language = openAISTTDefaultLanguage
	}
	url := config.BaseURL
	if url == "" {
		url = DefaultSTTURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	extra := map[string]any{}
	if config.Prompt != "" {
		extra[STTExtraPrompt] = config.Prompt
	}
	if config.Temperature != nil {
		extra[STTExtraTemperature] = *config.Temperature
	}
	if len(config.Keywords) > 0 {
		extra[STTExtraKeywords] = config.Keywords
	}
	s := &STTService{apiKey: config.APIKey, url: url, client: client}
	s.Base = stt.NewSegmented("OpenAISTT", s, stt.Options{
		Provider:   "openai",
		Settings:   frames.STTSettings{Model: model, Language: language, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    2010 * time.Millisecond,
		MaxSegment: 30 * time.Second,
	})
	return s
}

func baseLanguage(lang string) string {
	if i := strings.IndexAny(lang, "-_"); i > 0 {
		lang = lang[:i]
	}
	return strings.ToLower(lang)
}

func (s *STTService) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	settings := s.Settings()
	var body bytes.Buffer
	body.Grow(len(wav) + 1024)
	w := multipart.NewWriter(&body)
	part, err := w.CreateFormFile("file", "audio.wav")
	if err != nil {
		return stt.Transcript{}, err
	}
	if _, err := part.Write(wav); err != nil {
		return stt.Transcript{}, err
	}
	fields := [][2]string{{"model", settings.Model}}
	lang := baseLanguage(settings.Language)
	if lang != "" {
		fields = append(fields, [2]string{"language", lang})
	}
	if strings.Contains(settings.Model, "diarize") {
		fields = append(fields, [2]string{"response_format", "diarized_json"}, [2]string{"chunking_strategy", "auto"})
	}
	if v, ok := settings.Extra[STTExtraPrompt].(string); ok && v != "" {
		fields = append(fields, [2]string{"prompt", v})
	}
	if v, ok := settings.Extra[STTExtraTemperature].(float64); ok {
		fields = append(fields, [2]string{"temperature", strconv.FormatFloat(v, 'f', -1, 64)})
	}
	if kws, ok := settings.Extra[STTExtraKeywords].([]string); ok {
		for _, k := range kws {
			fields = append(fields, [2]string{"keywords[]", k})
		}
	}
	for _, f := range fields {
		if err := w.WriteField(f[0], f[1]); err != nil {
			return stt.Transcript{}, err
		}
	}
	if err := w.Close(); err != nil {
		return stt.Transcript{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, &body)
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, openAIErrorBodyLimit))
		return stt.Transcript{}, fmt.Errorf("openai stt HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var result struct {
		Text     string `json:"text"`
		Language string `json:"language"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return stt.Transcript{}, fmt.Errorf("decode openai stt response: %w", err)
	}
	if result.Language == "" {
		result.Language = lang
	}
	return stt.Transcript{Text: strings.TrimSpace(result.Text), Language: result.Language}, nil
}

type RealtimeSTTConfig struct {
	APIKey               string
	Model                string
	Language             string
	Prompt               string
	NoiseReduction       string
	Delay                string
	TurnDetection        map[string]any
	DisableInterruptions bool
	SampleRate           int
	BaseURL              string
}

type RealtimeSTTService struct {
	*stt.Base
	apiKey          string
	baseURL         string
	turnDetection   map[string]any
	shouldInterrupt bool

	session atomic.Pointer[reconnect.Session]
	sink    stt.Sink

	resampler *resample.Resampler
	scratch   []byte

	mu           sync.Mutex
	ready        bool
	confirmTimer *time.Timer
	closed       bool
}

func NewRealtimeSTTService(config RealtimeSTTConfig) *RealtimeSTTService {
	model := config.Model
	if model == "" {
		model = DefaultRealtimeSTTModel
	}
	language := config.Language
	if language == "" {
		language = openAISTTDefaultLanguage
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultRealtimeSTTURL
	}
	extra := map[string]any{}
	if config.Prompt != "" {
		extra[STTExtraPrompt] = config.Prompt
	}
	if config.NoiseReduction != "" {
		extra[RealtimeExtraNoise] = config.NoiseReduction
	}
	if config.Delay != "" {
		extra[RealtimeExtraDelay] = config.Delay
	}
	s := &RealtimeSTTService{
		apiKey:          config.APIKey,
		baseURL:         baseURL,
		turnDetection:   config.TurnDetection,
		shouldInterrupt: !config.DisableInterruptions,
	}
	s.Base = stt.NewStreaming("OpenAIRealtimeSTT", s, stt.Options{
		Provider:   "openai-realtime",
		Settings:   frames.STTSettings{Model: model, Language: language, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    1660 * time.Millisecond,
	})
	return s
}

func (s *RealtimeSTTService) serverVAD() bool {
	return s.turnDetection != nil
}

func (s *RealtimeSTTService) endpoint() (string, http.Header) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.apiKey)
	return s.baseURL + "?intent=transcription", header
}

func (s *RealtimeSTTService) Connect(ctx context.Context, sink stt.Sink) error {
	s.sink = sink
	if rate := s.SampleRate(); rate != RealtimeSTTSampleRate {
		r, err := resample.New(rate, RealtimeSTTSampleRate)
		if err != nil {
			return err
		}
		s.resampler = r
	}
	if s.serverVAD() {
		meta := frames.NewServiceMetadataFrame(s.Name())
		strategies := proposedTurnStrategies(s.shouldInterrupt)
		meta.UserTurnStrategies = &strategies
		sink.Push(meta, frames.Downstream)
	}
	session := reconnect.New(reconnect.Config{
		Name:      "OpenAIRealtimeSTT",
		Endpoint:  s.endpoint,
		OnOpen:    s.onOpen,
		OnMessage: func(_ int, data []byte) { s.onMessage(data) },
		OnError:   sink.Error,
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *RealtimeSTTService) onOpen(reconnect.Writer) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ready = false
	if s.confirmTimer == nil {
		s.confirmTimer = time.AfterFunc(realtimeConfirmTimeout, s.confirmTimeout)
	} else {
		s.confirmTimer.Reset(realtimeConfirmTimeout)
	}
	return nil
}

func (s *RealtimeSTTService) confirmTimeout() {
	s.mu.Lock()
	expired := !s.ready && !s.closed
	s.mu.Unlock()
	if !expired {
		return
	}
	s.sink.Error(errors.New("openai realtime stt: session was not configured in time"), false)
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
}

func (s *RealtimeSTTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil || len(audio) == 0 {
		return
	}
	pcm := audio
	if s.resampler != nil {
		s.scratch = s.resampler.ProcessBytes(s.scratch[:0], audio)
		pcm = s.scratch
		if len(pcm) == 0 {
			return
		}
	}
	buf := make([]byte, 0, len(realtimeAppendPrefix)+base64.StdEncoding.EncodedLen(len(pcm))+len(realtimeAppendSuffix))
	buf = append(buf, realtimeAppendPrefix...)
	buf = base64.StdEncoding.AppendEncode(buf, pcm)
	buf = append(buf, realtimeAppendSuffix...)
	session.SendDroppable(websocket.TextMessage, buf)
}

func (s *RealtimeSTTService) Finalize() {
	if s.serverVAD() {
		return
	}
	if session := s.session.Load(); session != nil {
		session.Send(websocket.TextMessage, realtimeCommit)
	}
}

func (s *RealtimeSTTService) Disconnect(graceful bool) {
	s.mu.Lock()
	s.closed = true
	if s.confirmTimer != nil {
		s.confirmTimer.Stop()
	}
	s.mu.Unlock()
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *RealtimeSTTService) ApplySettings(frames.STTSettings) error {
	s.mu.Lock()
	ready := s.ready
	s.mu.Unlock()
	if !ready {
		return nil
	}
	session := s.session.Load()
	if session == nil {
		return nil
	}
	return session.SendJSON(s.sessionUpdate())
}

func (s *RealtimeSTTService) sessionUpdate() map[string]any {
	settings := s.Settings()
	transcription := map[string]any{"model": settings.Model}
	if lang := baseLanguage(settings.Language); lang != "" {
		transcription["language"] = lang
	}
	if v, ok := settings.Extra[STTExtraPrompt].(string); ok && v != "" && settings.Model != DefaultRealtimeSTTModel {
		transcription["prompt"] = v
	}
	if v, ok := settings.Extra[RealtimeExtraDelay].(string); ok && v != "" {
		transcription["delay"] = v
	}
	input := map[string]any{
		"format":        map[string]any{"type": "audio/pcm", "rate": RealtimeSTTSampleRate},
		"transcription": transcription,
	}
	if s.turnDetection == nil {
		input["turn_detection"] = nil
	} else {
		input["turn_detection"] = s.turnDetection
	}
	if v, ok := settings.Extra[RealtimeExtraNoise].(string); ok && v != "" {
		input["noise_reduction"] = map[string]any{"type": v}
	}
	return map[string]any{
		"type":    "session.update",
		"session": map[string]any{"type": "transcription", "audio": map[string]any{"input": input}},
	}
}

type realtimeEvent struct {
	Type       string `json:"type"`
	Delta      string `json:"delta"`
	Transcript string `json:"transcript"`
	Error      struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (s *RealtimeSTTService) onMessage(data []byte) {
	var evt realtimeEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		return
	}
	switch evt.Type {
	case "session.created":
		if session := s.session.Load(); session != nil {
			if err := session.SendJSON(s.sessionUpdate()); err != nil {
				s.sink.Error(err, false)
			}
		}
	case "session.updated":
		s.mu.Lock()
		s.ready = true
		if s.confirmTimer != nil {
			s.confirmTimer.Stop()
		}
		s.mu.Unlock()
	case "conversation.item.input_audio_transcription.delta":
		if evt.Delta != "" {
			s.sink.Transcript(stt.Transcript{Text: evt.Delta, Language: baseLanguage(s.Settings().Language)})
		}
	case "conversation.item.input_audio_transcription.completed":
		if evt.Transcript != "" {
			s.sink.Transcript(stt.Transcript{Text: evt.Transcript, Final: true, Finalized: true, Language: baseLanguage(s.Settings().Language)})
		}
	case "conversation.item.input_audio_transcription.failed":
		s.sink.Error(errors.New("openai realtime stt: "+sttOrDefault(evt.Error.Message, "transcription failed")), false)
	case "input_audio_buffer.speech_started":
		s.sink.Push(frames.NewProposedUserStartedSpeakingFrame(), frames.Downstream)
		s.sink.Push(frames.NewProposedUserStartedSpeakingFrame(), frames.Upstream)
	case "input_audio_buffer.speech_stopped":
		s.sink.Push(frames.NewProposedUserStoppedSpeakingFrame(), frames.Downstream)
		s.sink.Push(frames.NewProposedUserStoppedSpeakingFrame(), frames.Upstream)
	case "error":
		s.sink.Error(fmt.Errorf("openai realtime stt error [%s]: %s", evt.Error.Code, sttOrDefault(evt.Error.Message, "unknown error")), false)
		if session := s.session.Load(); session != nil {
			session.Reconnect()
		}
	}
}

func sttOrDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func proposedTurnStrategies(enableInterruptions bool) turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{proposedTurnStart{interrupt: enableInterruptions}},
		StopStrategies:  []user_stop.UserTurnStopStrategy{proposedTurnStop{}},
	}
}

type proposedTurnStart struct{ interrupt bool }

func (proposedTurnStart) ShouldStart(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStartedSpeakingFrame)
	return ok
}

func (s proposedTurnStart) EnableInterruptions() bool { return s.interrupt }

func (proposedTurnStart) Reset() {}

type proposedTurnStop struct{}

func (proposedTurnStop) ShouldStop(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStoppedSpeakingFrame)
	return ok
}

func (proposedTurnStop) Reset() {}
