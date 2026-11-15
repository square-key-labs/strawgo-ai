package xtts

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	NativeSampleRate = 24000
	DefaultLanguage  = "en"
)

type TTSConfig struct {
	BaseURL    string
	Voice      string
	Language   string
	SampleRate int
	HTTPClient *http.Client
}

type speaker struct {
	SpeakerEmbedding json.RawMessage `json:"speaker_embedding"`
	GPTCondLatent    json.RawMessage `json:"gpt_cond_latent"`
}

type TTSService struct {
	*tts.Base
	baseURL string
	client  *http.Client

	mu       sync.Mutex
	speakers map[string]speaker
}

func NewTTSService(config TTSConfig) *TTSService {
	language := config.Language
	if language == "" {
		language = DefaultLanguage
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{baseURL: strings.TrimRight(config.BaseURL, "/"), client: client}
	s.Base = tts.NewHTTP("XTTS", s, tts.Options{
		Settings:   frames.TTSSettings{Voice: config.Voice, Language: language},
		SampleRate: config.SampleRate,
	})
	return s
}

func Language(code string) string {
	lower := strings.ToLower(code)
	if strings.HasPrefix(lower, "zh") {
		return "zh-cn"
	}
	if i := strings.IndexAny(lower, "-_"); i > 0 {
		return lower[:i]
	}
	return lower
}

func (s *TTSService) studioSpeakers(ctx context.Context) (map[string]speaker, error) {
	s.mu.Lock()
	cached := s.speakers
	s.mu.Unlock()
	if cached != nil {
		return cached, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.baseURL+"/studio_speakers", nil)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("xtts studio speakers HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var speakers map[string]speaker
	if err := json.NewDecoder(resp.Body).Decode(&speakers); err != nil {
		return nil, fmt.Errorf("xtts studio speakers: %w", err)
	}
	s.mu.Lock()
	s.speakers = speakers
	s.mu.Unlock()
	return speakers, nil
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	speakers, err := s.studioSpeakers(ctx)
	if err != nil {
		return nil, err
	}
	sp, ok := speakers[req.Settings.Voice]
	if !ok {
		return nil, fmt.Errorf("xtts voice %q is not one of the studio speakers", req.Settings.Voice)
	}
	if len(sp.SpeakerEmbedding) == 0 || len(sp.GPTCondLatent) == 0 {
		return nil, errors.New("xtts studio speaker has no embeddings")
	}
	payload, err := json.Marshal(map[string]any{
		"text":              strings.NewReplacer(".", "", "*", "").Replace(req.Text),
		"language":          Language(req.Settings.Language),
		"speaker_embedding": sp.SpeakerEmbedding,
		"gpt_cond_latent":   sp.GPTCondLatent,
		"add_wav_header":    false,
		"stream_chunk_size": 20,
	})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/tts_stream", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("xtts HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return tts.NewReadCloser(newTTSResampler(resp.Body, NativeSampleRate, req.SampleRate), resp.Body), nil
}

type ttsResampler struct {
	src   io.Reader
	rs    *resample.Resampler
	in    []byte
	carry int
	out   []byte
	tail  []int16
	off   int
	err   error
}

func newTTSResampler(src io.Reader, inRate, outRate int) io.Reader {
	if inRate <= 0 || outRate <= 0 || inRate == outRate {
		return src
	}
	rs, err := resample.New(inRate, outRate)
	if err != nil {
		return src
	}
	return &ttsResampler{src: src, rs: rs, in: make([]byte, 8192)}
}

func (r *ttsResampler) Read(p []byte) (int, error) {
	for r.off == len(r.out) {
		if r.err != nil {
			return 0, r.err
		}
		r.out, r.off = r.out[:0], 0
		n, err := r.src.Read(r.in[r.carry:])
		n += r.carry
		even := n &^ 1
		r.out = r.rs.ProcessBytes(r.out, r.in[:even])
		r.carry = n - even
		if r.carry == 1 {
			r.in[0] = r.in[even]
		}
		if err != nil {
			r.err = err
			if errors.Is(err, io.EOF) {
				r.tail = r.rs.Flush(r.tail[:0])
				for _, v := range r.tail {
					r.out = binary.LittleEndian.AppendUint16(r.out, uint16(v))
				}
			}
		}
	}
	n := copy(p, r.out[r.off:])
	r.off += n
	return n, nil
}
