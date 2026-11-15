package elevenlabs

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultHTTPTTSModel       = "eleven_flash_v2_5"
	httpTTSFallbackRate       = 24000
	httpTTSMaxQueuedRequests  = 256
	httpTTSNormalizationField = "apply_text_normalization"
)

var (
	httpTTSRates          = map[int]bool{8000: true, 16000: true, 22050: true, 24000: true, 32000: true, 44100: true, 48000: true}
	httpTTSNoPreviousText = map[string]bool{"eleven_v3": true, "eleven_v3_conversational": true}
	errHTTPTTSQueueFull   = errors.New("elevenlabs request queue full")
	errHTTPTTSNotStarted  = errors.New("elevenlabs not started")
)

type HTTPTTSConfig struct {
	APIKey                   string
	VoiceID                  string
	Model                    string
	Language                 string
	SampleRate               int
	VoiceSettings            *VoiceSettings
	OptimizeStreamingLatency *int
	EnableLogging            *bool
	ApplyTextNormalization   string
	BaseURL                  string
	HTTPClient               *http.Client
}

type httpTTSJob struct {
	id    string
	text  string
	flush bool
}

type httpTTSState struct {
	offset   float64
	previous string
	rs       *resample.Resampler
	carry    []byte
	tail     []int16
}

type HTTPTTSService struct {
	*tts.Base
	apiKey        string
	voiceSettings *VoiceSettings
	latency       *int
	logging       *bool
	baseURL       string
	client        *http.Client

	mu      sync.Mutex
	jobs    []httpTTSJob
	cur     string
	stopCur context.CancelFunc
	states  map[string]*httpTTSState
	wake    chan struct{}
	cancel  context.CancelFunc
	done    chan struct{}
}

func NewHTTPTTSService(config HTTPTTSConfig) *HTTPTTSService {
	model := config.Model
	if model == "" {
		model = DefaultHTTPTTSModel
	}
	baseURL := strings.TrimRight(config.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultHTTPURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	settings := frames.TTSSettings{Model: model, Voice: config.VoiceID, Language: config.Language}
	if config.ApplyTextNormalization != "" {
		settings.Extra = map[string]any{httpTTSNormalizationField: config.ApplyTextNormalization}
	}
	s := &HTTPTTSService{
		apiKey:        config.APIKey,
		voiceSettings: config.VoiceSettings,
		latency:       config.OptimizeStreamingLatency,
		logging:       config.EnableLogging,
		baseURL:       baseURL,
		client:        client,
		states:        make(map[string]*httpTTSState),
		wake:          make(chan struct{}, 1),
	}
	s.Base = tts.NewStreaming("ElevenLabsHTTPTTS", s, tts.Options{
		Settings:       settings,
		SampleRate:     config.SampleRate,
		WordTimestamps: true,
	})
	return s
}

func (s *HTTPTTSService) Connect(ctx context.Context, sink tts.Sink) error {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	s.mu.Lock()
	s.cancel, s.done = cancel, done
	s.mu.Unlock()
	go s.loop(ctx, sink, done)
	return nil
}

func (s *HTTPTTSService) Disconnect(bool) {
	s.mu.Lock()
	cancel, done := s.cancel, s.done
	s.cancel = nil
	s.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
}

func (s *HTTPTTSService) enqueue(j httpTTSJob) error {
	s.mu.Lock()
	switch {
	case s.cancel == nil:
		s.mu.Unlock()
		return errHTTPTTSNotStarted
	case len(s.jobs) >= httpTTSMaxQueuedRequests:
		s.mu.Unlock()
		return errHTTPTTSQueueFull
	}
	s.jobs = append(s.jobs, j)
	s.mu.Unlock()
	select {
	case s.wake <- struct{}{}:
	default:
	}
	return nil
}

func (s *HTTPTTSService) Speak(contextID, text string) error {
	return s.enqueue(httpTTSJob{id: contextID, text: text})
}

func (s *HTTPTTSService) Flush(contextID string) error {
	return s.enqueue(httpTTSJob{id: contextID, flush: true})
}

func (s *HTTPTTSService) Cancel(contextID string) error {
	s.mu.Lock()
	kept := s.jobs[:0]
	for _, j := range s.jobs {
		if j.id != contextID {
			kept = append(kept, j)
		}
	}
	clear(s.jobs[len(kept):])
	s.jobs = kept
	delete(s.states, contextID)
	if s.cur == contextID && s.stopCur != nil {
		s.stopCur()
	}
	s.mu.Unlock()
	return nil
}

func (s *HTTPTTSService) next(ctx context.Context) (httpTTSJob, *httpTTSState, context.Context, bool) {
	for {
		s.mu.Lock()
		if len(s.jobs) > 0 {
			j := s.jobs[0]
			s.jobs[0] = httpTTSJob{}
			s.jobs = s.jobs[1:]
			state := s.states[j.id]
			if j.flush {
				delete(s.states, j.id)
				s.mu.Unlock()
				return j, state, nil, true
			}
			if state == nil {
				state = &httpTTSState{}
				s.states[j.id] = state
			}
			jctx, cancel := context.WithCancel(ctx)
			s.cur, s.stopCur = j.id, cancel
			s.mu.Unlock()
			return j, state, jctx, true
		}
		s.mu.Unlock()
		select {
		case <-s.wake:
		case <-ctx.Done():
			return httpTTSJob{}, nil, nil, false
		}
	}
}

func (s *HTTPTTSService) loop(ctx context.Context, sink tts.Sink, done chan struct{}) {
	defer close(done)
	for {
		j, state, jctx, ok := s.next(ctx)
		if !ok {
			return
		}
		if j.flush {
			if state != nil && state.rs != nil {
				state.tail = state.rs.Flush(state.tail[:0])
				if len(state.tail) > 0 {
					out := make([]byte, 0, 2*len(state.tail))
					for _, v := range state.tail {
						out = binary.LittleEndian.AppendUint16(out, uint16(v))
					}
					sink.Audio(j.id, out)
				}
			}
			sink.Done(j.id)
			continue
		}
		err := s.synthesize(jctx, sink, j, state)
		s.mu.Lock()
		s.stopCur()
		s.cur, s.stopCur = "", nil
		s.mu.Unlock()
		if err != nil && jctx.Err() == nil {
			sink.Error(err, false)
		}
	}
}

func (s *HTTPTTSService) voiceSettingsBody(settings frames.TTSSettings) map[string]any {
	m := map[string]any{}
	if vs := s.voiceSettings; vs != nil {
		if vs.Stability != 0 {
			m["stability"] = vs.Stability
		}
		if vs.SimilarityBoost != 0 {
			m["similarity_boost"] = vs.SimilarityBoost
		}
		if vs.Style != 0 {
			m["style"] = vs.Style
		}
		if vs.UseSpeakerBoost {
			m["use_speaker_boost"] = true
		}
		if vs.Speed != 0 {
			m["speed"] = vs.Speed
		}
	}
	if settings.Speed != nil {
		m["speed"] = *settings.Speed
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

type httpTTSAlignment struct {
	Characters []string  `json:"characters"`
	Starts     []float64 `json:"character_start_times_seconds"`
	Ends       []float64 `json:"character_end_times_seconds"`
}

type httpTTSLine struct {
	AudioBase64         string            `json:"audio_base64"`
	Alignment           *httpTTSAlignment `json:"alignment"`
	NormalizedAlignment *httpTTSAlignment `json:"normalized_alignment"`
}

func (s *HTTPTTSService) synthesize(ctx context.Context, sink tts.Sink, j httpTTSJob, state *httpTTSState) error {
	settings := s.Settings()
	outRate := s.SampleRate()
	reqRate := outRate
	if !httpTTSRates[reqRate] {
		reqRate = httpTTSFallbackRate
		if state.rs == nil {
			rs, err := resample.New(reqRate, outRate)
			if err != nil {
				return err
			}
			state.rs = rs
		}
	}
	body := map[string]any{"text": j.text, "model_id": settings.Model}
	if state.previous != "" && !httpTTSNoPreviousText[settings.Model] {
		body["previous_text"] = state.previous
	}
	if vs := s.voiceSettingsBody(settings); vs != nil {
		body["voice_settings"] = vs
	}
	if v, ok := settings.Extra[httpTTSNormalizationField]; ok {
		body[httpTTSNormalizationField] = v
	}
	if settings.Language != "" && multilingualModels[settings.Model] {
		body["language_code"] = settings.Language
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	q := url.Values{}
	q.Set("output_format", "pcm_"+strconv.Itoa(reqRate))
	if s.latency != nil {
		q.Set("optimize_streaming_latency", strconv.Itoa(*s.latency))
	}
	if s.logging != nil {
		q.Set("enable_logging", strconv.FormatBool(*s.logging))
	}
	endpoint := fmt.Sprintf("%s/v1/text-to-speech/%s/stream/with-timestamps?%s", s.baseURL, url.PathEscape(settings.Voice), q.Encode())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("xi-api-key", s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("elevenlabs HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	r := bufio.NewReaderSize(resp.Body, 64*1024)
	var w httpTTSWords
	w.offset = state.offset
	duration := 0.0
	started := false
	for {
		line, readErr := r.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			var msg httpTTSLine
			if json.Unmarshal(line, &msg) == nil {
				if msg.AudioBase64 != "" {
					audio, err := base64.StdEncoding.DecodeString(msg.AudioBase64)
					if err != nil {
						return err
					}
					if audio = state.convert(audio); len(audio) > 0 {
						sink.Audio(j.id, audio)
					}
				}
				a := msg.Alignment
				if a == nil {
					a = msg.NormalizedAlignment
				}
				if a != nil && len(a.Characters) == len(a.Starts) {
					if !started {
						a = trimLeadingSpaces(a)
					}
					started = true
					if n := len(a.Ends); n > 0 {
						duration = max(duration, a.Ends[n-1])
					}
					if words := w.add(a); len(words) > 0 {
						sink.Words(j.id, words)
					}
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return readErr
		}
	}
	if w.partial != "" {
		sink.Words(j.id, []tts.Word{{Text: w.partial, Start: httpTTSSeconds(w.partialStart)}})
	}
	state.offset += duration
	if state.previous == "" {
		state.previous = j.text
	} else {
		state.previous += " " + j.text
	}
	return nil
}

func (st *httpTTSState) convert(audio []byte) []byte {
	if st.rs == nil {
		return audio
	}
	if len(st.carry) > 0 {
		audio = append(st.carry, audio...)
		st.carry = nil
	}
	var odd []byte
	if len(audio)%2 == 1 {
		odd = []byte{audio[len(audio)-1]}
		audio = audio[:len(audio)-1]
	}
	out := st.rs.ProcessBytes(nil, audio)
	st.carry = odd
	return out
}

func trimLeadingSpaces(a *httpTTSAlignment) *httpTTSAlignment {
	n := 0
	for n < len(a.Characters) && a.Characters[n] == " " {
		n++
	}
	if n == 0 {
		return a
	}
	out := &httpTTSAlignment{Characters: a.Characters[n:], Starts: a.Starts[n:]}
	if len(a.Ends) >= n {
		out.Ends = a.Ends[n:]
	}
	return out
}

type httpTTSWords struct {
	offset       float64
	partial      string
	partialStart float64
}

func (w *httpTTSWords) add(a *httpTTSAlignment) []tts.Word {
	var words []tts.Word
	for i, ch := range a.Characters {
		if ch == " " {
			if w.partial != "" {
				words = append(words, tts.Word{Text: w.partial, Start: httpTTSSeconds(w.partialStart)})
				w.partial = ""
			}
			continue
		}
		if w.partial == "" {
			w.partialStart = w.offset + a.Starts[i]
		}
		w.partial += ch
	}
	return words
}

func httpTTSSeconds(v float64) time.Duration {
	return time.Duration(v * float64(time.Second))
}
