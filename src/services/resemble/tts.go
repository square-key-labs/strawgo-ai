package resemble

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL       = "wss://websocket.cluster.resemble.ai/stream"
	DefaultPrecision = "PCM_16"
	fallbackRate     = 48000
)

var supportedRates = map[int]bool{8000: true, 16000: true, 22050: true, 32000: true, 44100: true, 48000: true}

type TTSConfig struct {
	APIKey                    string
	VoiceID                   string
	SampleRate                int
	ApplyCustomPronunciations bool
	URL                       string
}

type request struct {
	contextID string
	started   bool
	offset    float64
}

type contextState struct {
	pending      int
	flushed      bool
	samples      int64
	rate         int
	rs           *resample.Resampler
	carry        []byte
	partial      string
	partialStart float64
}

type TTSService struct {
	*tts.Base
	apiKey  string
	url     string
	session atomic.Pointer[reconnect.Session]
	sink    atomic.Pointer[tts.Sink]

	mu       sync.Mutex
	nextID   int64
	requests map[int64]*request
	contexts map[string]*contextState
}

func NewTTSService(config TTSConfig) *TTSService {
	u := config.URL
	if u == "" {
		u = DefaultURL
	}
	s := &TTSService{
		apiKey:   config.APIKey,
		url:      u,
		requests: make(map[int64]*request),
		contexts: make(map[string]*contextState),
	}
	settings := frames.TTSSettings{Voice: config.VoiceID}
	if config.ApplyCustomPronunciations {
		settings.Extra = map[string]any{"apply_custom_pronunciations": true}
	}
	s.Base = tts.NewStreaming("ResembleTTS", s, tts.Options{
		Settings:       settings,
		SampleRate:     config.SampleRate,
		WordTimestamps: true,
	})
	return s
}

func (s *TTSService) endpoint() (string, http.Header) {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.apiKey)
	return s.url, header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	s.sink.Store(&sink)
	session := reconnect.New(reconnect.Config{
		Name:         "ResembleTTS",
		Endpoint:     s.endpoint,
		OnDisconnect: func(error) { s.reset(sink) },
		OnMessage:    func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:      sink.Error,
		PingInterval: 20 * time.Second,
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *TTSService) reset(sink tts.Sink) {
	s.mu.Lock()
	ids := make([]string, 0, len(s.contexts))
	for id := range s.contexts {
		ids = append(ids, id)
	}
	clear(s.contexts)
	clear(s.requests)
	s.mu.Unlock()
	for _, id := range ids {
		sink.Done(id)
	}
}

func wireRate(rate int) int {
	if supportedRates[rate] {
		return rate
	}
	return fallbackRate
}

type message struct {
	VoiceUUID                 string `json:"voice_uuid"`
	Data                      string `json:"data"`
	BinaryResponse            bool   `json:"binary_response"`
	RequestID                 int64  `json:"request_id"`
	OutputFormat              string `json:"output_format"`
	SampleRate                int    `json:"sample_rate"`
	Precision                 string `json:"precision"`
	NoAudioHeader             bool   `json:"no_audio_header"`
	ApplyCustomPronunciations bool   `json:"apply_custom_pronunciations"`
}

func (s *TTSService) Speak(contextID, text string) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("resemble not connected")
	}
	settings := s.Settings()
	out := s.SampleRate()
	rate := wireRate(out)
	s.mu.Lock()
	c := s.contexts[contextID]
	if c == nil {
		c = &contextState{rate: rate}
		if rate != out {
			c.rs, _ = resample.New(rate, out)
		}
		s.contexts[contextID] = c
	}
	c.pending++
	id := s.nextID
	s.nextID++
	s.requests[id] = &request{contextID: contextID}
	s.mu.Unlock()
	custom, _ := settings.Extra["apply_custom_pronunciations"].(bool)
	return session.SendJSON(message{
		VoiceUUID:                 settings.Voice,
		Data:                      text,
		RequestID:                 id,
		OutputFormat:              "wav",
		SampleRate:                c.rate,
		Precision:                 DefaultPrecision,
		NoAudioHeader:             true,
		ApplyCustomPronunciations: custom,
	})
}

func (s *TTSService) Flush(contextID string) error {
	s.mu.Lock()
	c := s.contexts[contextID]
	finished := c == nil || c.pending == 0
	if c != nil {
		c.flushed = true
	}
	var tail []byte
	if finished && c != nil {
		tail = drain(c)
		delete(s.contexts, contextID)
	}
	s.mu.Unlock()
	if finished {
		if sink := s.sink.Load(); sink != nil {
			(*sink).Audio(contextID, tail)
			(*sink).Done(contextID)
		}
	}
	return nil
}

func (s *TTSService) Cancel(contextID string) error {
	s.mu.Lock()
	delete(s.contexts, contextID)
	for id, r := range s.requests {
		if r.contextID == contextID {
			delete(s.requests, id)
		}
	}
	s.mu.Unlock()
	return nil
}

func drain(c *contextState) []byte {
	if c.rs == nil {
		return nil
	}
	var out []byte
	for _, v := range c.rs.Flush(nil) {
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}
	return out
}

type response struct {
	Type            string `json:"type"`
	RequestID       *int64 `json:"request_id"`
	AudioContent    string `json:"audio_content"`
	AudioTimestamps *struct {
		GraphChars []string    `json:"graph_chars"`
		GraphTimes [][]float64 `json:"graph_times"`
	} `json:"audio_timestamps"`
	ErrorName  string `json:"error_name"`
	Message    string `json:"message"`
	StatusCode int    `json:"status_code"`
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	switch resp.Type {
	case "audio":
		s.onAudio(sink, resp)
	case "audio_end":
		s.finishRequest(sink, resp.RequestID)
	case "error":
		sink.Error(fmt.Errorf("resemble: %s (status %d): %s", resp.ErrorName, resp.StatusCode, resp.Message), false)
		s.finishRequest(sink, resp.RequestID)
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			if session := s.session.Load(); session != nil {
				session.Reconnect()
			}
		}
	}
}

func (s *TTSService) onAudio(sink tts.Sink, resp response) {
	if resp.RequestID == nil {
		return
	}
	var audio []byte
	if resp.AudioContent != "" {
		var err error
		if audio, err = base64.StdEncoding.DecodeString(resp.AudioContent); err != nil {
			sink.Error(err, false)
			return
		}
	}
	s.mu.Lock()
	r := s.requests[*resp.RequestID]
	var c *contextState
	if r != nil {
		c = s.contexts[r.contextID]
	}
	if c == nil {
		s.mu.Unlock()
		return
	}
	if !r.started {
		r.started = true
		r.offset = float64(c.samples) / float64(c.rate)
	}
	if len(c.carry) > 0 {
		audio = append(c.carry, audio...)
		c.carry = nil
	}
	if len(audio)%2 == 1 {
		c.carry = []byte{audio[len(audio)-1]}
		audio = audio[:len(audio)-1]
	}
	c.samples += int64(len(audio) / 2)
	if c.rs != nil && len(audio) > 0 {
		audio = c.rs.ProcessBytes(nil, audio)
	}
	var words []tts.Word
	if ts := resp.AudioTimestamps; ts != nil {
		words = wordTimes(c, r.offset, ts.GraphChars, ts.GraphTimes)
	}
	id := r.contextID
	s.mu.Unlock()
	sink.Audio(id, audio)
	if len(words) > 0 {
		sink.Words(id, words)
	}
}

func wordTimes(c *contextState, offset float64, chars []string, times [][]float64) []tts.Word {
	var words []tts.Word
	for i, ch := range chars {
		if i >= len(times) || len(times[i]) < 1 {
			break
		}
		if ch == " " || ch == "" {
			if c.partial != "" {
				words = append(words, tts.Word{Text: c.partial, Start: seconds(c.partialStart)})
				c.partial = ""
			}
			continue
		}
		if c.partial == "" {
			c.partialStart = offset + times[i][0]
		}
		c.partial += ch
	}
	return words
}

func seconds(v float64) time.Duration {
	return time.Duration(v * float64(time.Second))
}

func (s *TTSService) finishRequest(sink tts.Sink, requestID *int64) {
	if requestID == nil {
		return
	}
	s.mu.Lock()
	r := s.requests[*requestID]
	delete(s.requests, *requestID)
	var c *contextState
	if r != nil {
		c = s.contexts[r.contextID]
	}
	if c == nil {
		s.mu.Unlock()
		return
	}
	c.pending--
	var words []tts.Word
	if c.partial != "" {
		words = []tts.Word{{Text: c.partial, Start: seconds(c.partialStart)}}
		c.partial = ""
	}
	done := c.flushed && c.pending <= 0
	var tail []byte
	if done {
		tail = drain(c)
		delete(s.contexts, r.contextID)
	}
	s.mu.Unlock()
	if len(words) > 0 {
		sink.Words(r.contextID, words)
	}
	if done {
		sink.Audio(r.contextID, tail)
		sink.Done(r.contextID)
	}
}
