package speechify

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL         = "https://api.speechify.ai"
	DefaultModel       = "simba-3.5"
	DefaultVoice       = "geffen_32"
	APIVersion         = "2026-09-30"
	defaultRate        = 24000
	queueSize          = 256
	extraLoudness      = "loudness_normalization"
	extraNormalize     = "text_normalization"
	extraSafety        = "safety_identifier"
	extraPronunciation = "pronunciations"
)

var (
	supportedRates = map[int]bool{8000: true, 16000: true, 22050: true, 24000: true, 44100: true, 48000: true}
	errQueueFull   = errors.New("speechify request queue full")
)

var languages = map[string]string{"de": "de-DE", "en": "en-US", "es": "es-ES", "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR"}

func Language(code string) string {
	if strings.ContainsAny(code, "-_") {
		return code
	}
	if mapped, ok := languages[strings.ToLower(code)]; ok {
		return mapped
	}
	return code
}

type TTSConfig struct {
	APIKey     string
	VoiceID    string
	Model      string
	Language   string
	Params     map[string]any
	SampleRate int
	URL        string
	HTTPClient *http.Client
}

type job struct {
	id    string
	text  string
	flush bool
	epoch uint64
}

type TTSService struct {
	*tts.Base
	apiKey string
	url    string
	client *http.Client
	jobs   chan job

	mu      sync.Mutex
	epoch   uint64
	current context.CancelFunc
	stop    context.CancelFunc
	done    chan struct{}
}

func NewTTSService(config TTSConfig) *TTSService {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.VoiceID, Language: config.Language}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if settings.Voice == "" {
		settings.Voice = DefaultVoice
	}
	if len(config.Params) > 0 {
		settings.Extra = make(map[string]any, len(config.Params))
		for k, v := range config.Params {
			settings.Extra[k] = v
		}
	}
	base := strings.TrimRight(config.URL, "/")
	if base == "" {
		base = DefaultURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{apiKey: config.APIKey, url: base + "/v1/audio/stream/with-timestamps", client: client, jobs: make(chan job, queueSize)}
	s.Base = tts.NewStreaming("SpeechifyTTS", s, tts.Options{
		Settings:       settings,
		SampleRate:     config.SampleRate,
		WordTimestamps: true,
	})
	return s
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	runCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	s.mu.Lock()
	s.stop, s.done = cancel, done
	s.mu.Unlock()
	go s.run(runCtx, sink, done)
	return nil
}

func (s *TTSService) Disconnect(bool) {
	s.mu.Lock()
	stop, done := s.stop, s.done
	s.mu.Unlock()
	if stop == nil {
		return
	}
	stop()
	<-done
}

func (s *TTSService) enqueue(j job) error {
	s.mu.Lock()
	j.epoch = s.epoch
	s.mu.Unlock()
	select {
	case s.jobs <- j:
		return nil
	default:
		return errQueueFull
	}
}

func (s *TTSService) Speak(contextID, text string) error {
	return s.enqueue(job{id: contextID, text: text})
}

func (s *TTSService) Flush(contextID string) error {
	return s.enqueue(job{id: contextID, flush: true})
}

func (s *TTSService) Cancel(string) error {
	s.mu.Lock()
	s.epoch++
	if s.current != nil {
		s.current()
	}
	s.mu.Unlock()
	return nil
}

func (s *TTSService) run(ctx context.Context, sink tts.Sink, done chan struct{}) {
	defer close(done)
	offsets := make(map[string]float64)
	var seen uint64
	for {
		var j job
		select {
		case <-ctx.Done():
			return
		case j = <-s.jobs:
		}
		s.mu.Lock()
		if s.epoch != seen {
			seen = s.epoch
			clear(offsets)
		}
		if j.epoch != s.epoch {
			s.mu.Unlock()
			continue
		}
		if j.flush {
			s.mu.Unlock()
			delete(offsets, j.id)
			sink.Done(j.id)
			continue
		}
		reqCtx, cancel := context.WithCancel(ctx)
		s.current = cancel
		s.mu.Unlock()
		end, err := s.synthesize(reqCtx, sink, j.id, j.text, offsets[j.id])
		if err == nil || end > 0 {
			offsets[j.id] += end
		}
		if err != nil && reqCtx.Err() == nil {
			sink.Error(err, false)
		}
		s.mu.Lock()
		s.current = nil
		s.mu.Unlock()
		cancel()
	}
}

func (s *TTSService) payload(text string, settings frames.TTSSettings, format string) map[string]any {
	p := map[string]any{"input": text, "voice_id": settings.Voice, "model": settings.Model, "output_format": format}
	if settings.Language != "" {
		p["language"] = Language(settings.Language)
	}
	if v, ok := settings.Extra[extraSafety]; ok {
		p["safety_identifier"] = v
	}
	if v, ok := settings.Extra[extraPronunciation]; ok {
		p["pronunciations"] = v
	}
	options := map[string]any{}
	if v, ok := settings.Extra[extraLoudness]; ok {
		options["loudness_normalization"] = v
	}
	if v, ok := settings.Extra[extraNormalize]; ok {
		options["text_normalization"] = v
	}
	if len(options) > 0 {
		p["options"] = options
	}
	return p
}

type speechMark struct {
	Type      string  `json:"type"`
	Start     *int    `json:"start"`
	End       *int    `json:"end"`
	StartTime float64 `json:"start_time"`
	EndTime   float64 `json:"end_time"`
	Value     string  `json:"value"`
}

type event struct {
	Audio           string       `json:"audio"`
	SpeechMarks     []speechMark `json:"speech_marks"`
	AudioDurationMs float64      `json:"audio_duration_ms"`
	Error           *struct {
		Code    any    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func (s *TTSService) synthesize(ctx context.Context, sink tts.Sink, id, text string, offset float64) (float64, error) {
	outRate := s.SampleRate()
	rate := outRate
	if !supportedRates[rate] {
		rate = defaultRate
	}
	body, err := json.Marshal(s.payload(text, s.Settings(), "pcm_"+strconv.Itoa(rate)))
	if err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Speechify-Version", APIVersion)
	req.Header.Set("Speechify-Caller", "strawgo")
	resp, err := s.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return 0, fmt.Errorf("speechify HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var rs *resample.Resampler
	if rate != outRate {
		if rs, err = resample.New(rate, outRate); err != nil {
			return 0, err
		}
	}
	marks := marker{text: []rune(text), offset: offset}
	duration := 0.0
	var name string
	var data []byte
	r := bufio.NewReaderSize(resp.Body, 64*1024)
	for {
		line, rerr := r.ReadBytes('\n')
		line = bytes.TrimRight(line, "\r\n")
		if len(line) == 0 && len(data) > 0 {
			if derr := s.dispatch(sink, id, name, data, rs, &marks, &duration); derr != nil {
				return marks.end, derr
			}
			name, data = "", data[:0]
		} else if v, ok := bytes.CutPrefix(line, []byte("event:")); ok {
			name = string(bytes.TrimSpace(v))
		} else if v, ok := bytes.CutPrefix(line, []byte("data:")); ok {
			if len(data) > 0 {
				data = append(data, '\n')
			}
			data = append(data, bytes.TrimPrefix(v, []byte(" "))...)
		}
		if rerr != nil {
			if len(data) > 0 {
				if derr := s.dispatch(sink, id, name, data, rs, &marks, &duration); derr != nil {
					return marks.end, derr
				}
			}
			if w := marks.flush(); len(w) > 0 {
				sink.Words(id, w)
			}
			if rs != nil {
				if tail := rs.Flush(nil); len(tail) > 0 {
					out := make([]byte, 0, 2*len(tail))
					for _, v := range tail {
						out = append(out, byte(v), byte(uint16(v)>>8))
					}
					sink.Audio(id, out)
				}
			}
			if errors.Is(rerr, io.EOF) {
				return max(duration, marks.end), nil
			}
			return max(duration, marks.end), rerr
		}
	}
}

func (s *TTSService) dispatch(sink tts.Sink, id, name string, data []byte, rs *resample.Resampler, marks *marker, duration *float64) error {
	var ev event
	if err := json.Unmarshal(data, &ev); err != nil {
		return nil
	}
	switch name {
	case "speech.chunk":
		if ev.Audio != "" {
			audio, err := base64.StdEncoding.DecodeString(ev.Audio)
			if err != nil {
				return err
			}
			if rs != nil {
				audio = rs.ProcessBytes(nil, audio)
			}
			sink.Audio(id, audio)
		}
		if w := marks.add(ev.SpeechMarks); len(w) > 0 {
			sink.Words(id, w)
		}
	case "speech.done":
		*duration = max(*duration, ev.AudioDurationMs/1000)
	case "speech.error":
		if ev.Error != nil {
			return fmt.Errorf("speechify (%v): %s", ev.Error.Code, ev.Error.Message)
		}
		return fmt.Errorf("speechify: %s", data)
	}
	return nil
}

type marker struct {
	text      []rune
	offset    float64
	end       float64
	pending   bool
	ps, pe    int
	pendStart float64
}

func seconds(v float64) time.Duration {
	return time.Duration(v * float64(time.Second))
}

func (m *marker) add(marks []speechMark) []tts.Word {
	var words []tts.Word
	for _, mk := range marks {
		if mk.Type != "" && mk.Type != "word" {
			continue
		}
		m.end = max(m.end, mk.EndTime/1000)
		start := m.offset + mk.StartTime/1000
		if mk.Start == nil || mk.End == nil || *mk.Start < 0 || *mk.Start >= *mk.End || *mk.End > len(m.text) {
			words = append(words, m.flush()...)
			if v := strings.TrimSpace(mk.Value); v != "" {
				words = append(words, tts.Word{Text: v, Start: seconds(start)})
			}
			continue
		}
		s, e := *mk.Start, *mk.End
		if m.pending && m.pe == s {
			m.pe = e
		} else {
			words = append(words, m.flush()...)
			m.pending, m.ps, m.pe, m.pendStart = true, s, e, start
		}
		if e >= len(m.text) || unicode.IsSpace(m.text[e]) {
			words = append(words, m.flush()...)
		}
	}
	return words
}

func (m *marker) flush() []tts.Word {
	if !m.pending {
		return nil
	}
	m.pending = false
	w := strings.TrimSpace(string(m.text[m.ps:m.pe]))
	if w == "" {
		return nil
	}
	return []tts.Word{{Text: w, Start: seconds(m.pendStart)}}
}
