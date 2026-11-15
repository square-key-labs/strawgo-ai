package inworld

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const maxQueuedRequests = 256

var errQueueFull = errors.New("inworld request queue full")

type job struct {
	id    string
	text  string
	flush bool
}

type httpState struct {
	offset   float64
	sawWords bool
}

type HTTPTTSService struct {
	*tts.Base
	apiKey   string
	url      string
	encoding string
	client   *http.Client

	mu      sync.Mutex
	jobs    []job
	cur     string
	stopCur context.CancelFunc
	states  map[string]*httpState
	wake    chan struct{}
	cancel  context.CancelFunc
	done    chan struct{}
}

func NewHTTPTTSService(config TTSConfig) *HTTPTTSService {
	u := config.URL
	if u == "" {
		u = DefaultHTTPURL
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = DefaultEncoding
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &HTTPTTSService{
		apiKey:   config.APIKey,
		url:      u,
		encoding: encoding,
		client:   client,
		states:   make(map[string]*httpState),
		wake:     make(chan struct{}, 1),
	}
	s.Base = tts.NewStreaming("InworldHTTPTTS", s, tts.Options{
		Settings:       defaultSettings(config),
		SampleRate:     config.SampleRate,
		WordTimestamps: true,
	})
	return s
}

func (s *HTTPTTSService) Connect(ctx context.Context, sink tts.Sink) error {
	ctx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.cancel = cancel
	s.done = make(chan struct{})
	done := s.done
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

func (s *HTTPTTSService) enqueue(j job) error {
	s.mu.Lock()
	if s.cancel == nil {
		s.mu.Unlock()
		return errors.New("inworld not started")
	}
	if len(s.jobs) >= maxQueuedRequests {
		s.mu.Unlock()
		return errQueueFull
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
	return s.enqueue(job{id: contextID, text: text})
}

func (s *HTTPTTSService) Flush(contextID string) error {
	return s.enqueue(job{id: contextID, flush: true})
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

func (s *HTTPTTSService) next(ctx context.Context) (job, *httpState, context.Context, bool) {
	for {
		s.mu.Lock()
		if len(s.jobs) > 0 {
			j := s.jobs[0]
			s.jobs[0] = job{}
			s.jobs = s.jobs[1:]
			if len(s.jobs) == 0 {
				s.jobs = s.jobs[:0:0]
			}
			state := s.states[j.id]
			if j.flush {
				delete(s.states, j.id)
				s.mu.Unlock()
				return j, state, nil, true
			}
			if state == nil {
				state = &httpState{}
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
			return job{}, nil, nil, false
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

type streamLine struct {
	Result *struct {
		AudioContent  string         `json:"audioContent"`
		TimestampInfo *timestampInfo `json:"timestampInfo"`
	} `json:"result"`
	Error json.RawMessage `json:"error"`
}

func (s *HTTPTTSService) synthesize(ctx context.Context, sink tts.Sink, j job, state *httpState) error {
	settings := s.Settings()
	body := map[string]any{
		"text":                       j.text,
		"voiceId":                    settings.Voice,
		"modelId":                    settings.Model,
		"audioConfig":                audioConfig(settings, s.encoding, s.SampleRate()),
		"timestampType":              "WORD",
		"timestampTransportStrategy": "ASYNC",
	}
	if settings.Language != "" {
		body["language"] = Language(settings.Language)
	}
	for k, v := range settings.Extra {
		body[k] = v
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Basic "+s.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Request-Id", uuid.NewString())
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("inworld HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	r := bufio.NewReaderSize(resp.Body, 64*1024)
	duration := 0.0
	for {
		line, readErr := r.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			var msg streamLine
			if err := json.Unmarshal(line, &msg); err == nil {
				if len(msg.Error) > 0 && string(msg.Error) != "null" {
					return fmt.Errorf("inworld: %s", msg.Error)
				}
				if res := msg.Result; res != nil {
					if res.AudioContent != "" {
						audio, err := decodeAudio(res.AudioContent)
						if err != nil {
							return err
						}
						sink.Audio(j.id, audio)
					}
					if res.TimestampInfo != nil {
						words, end := wordTimes(res.TimestampInfo.WordAlignment, state.offset)
						duration = max(duration, end)
						if len(words) > 0 {
							state.sawWords = true
							sink.Words(j.id, words)
						}
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
	if !state.sawWords {
		if text := strings.TrimSpace(j.text); text != "" {
			sink.Words(j.id, []tts.Word{{Text: text, Start: time.Duration(state.offset * float64(time.Second))}})
		}
	}
	state.sawWords = false
	state.offset += duration
	return nil
}
