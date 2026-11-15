package openai

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

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultTTSModel      = "gpt-4o-mini-tts"
	DefaultTTSVoice      = "alloy"
	TTSNativeSampleRate  = 24000
	ttsInstructionsField = "instructions"
)

var ValidTTSVoices = map[string]bool{
	"alloy": true, "ash": true, "ballad": true, "cedar": true, "coral": true, "echo": true, "fable": true,
	"marin": true, "nova": true, "onyx": true, "sage": true, "shimmer": true, "verse": true,
}

type TTSConfig struct {
	APIKey       string
	BaseURL      string
	Model        string
	Voice        string
	Instructions string
	Speed        float64
	SampleRate   int
	HTTPClient   *http.Client
}

type TTSService struct {
	*tts.Base
	apiKey  string
	baseURL string
	client  *http.Client
}

func NewTTSService(config TTSConfig) *TTSService {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.Voice}
	if settings.Model == "" {
		settings.Model = DefaultTTSModel
	}
	if settings.Voice == "" {
		settings.Voice = DefaultTTSVoice
	}
	if config.Speed > 0 {
		speed := config.Speed
		settings.Speed = &speed
	}
	if config.Instructions != "" {
		settings.Extra = map[string]any{ttsInstructionsField: config.Instructions}
	}
	baseURL := strings.TrimRight(config.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{apiKey: config.APIKey, baseURL: baseURL, client: client}
	s.Base = tts.NewHTTP("OpenAITTS", s, tts.Options{Settings: settings, SampleRate: config.SampleRate})
	return s
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	if s.baseURL == DefaultBaseURL && !ValidTTSVoices[req.Settings.Voice] {
		return nil, fmt.Errorf("openai TTS voice %q is not supported", req.Settings.Voice)
	}
	body := map[string]any{
		"input":           req.Text,
		"model":           req.Settings.Model,
		"voice":           req.Settings.Voice,
		"response_format": "pcm",
	}
	if v, ok := req.Settings.Extra[ttsInstructionsField].(string); ok && v != "" {
		body[ttsInstructionsField] = v
	}
	if req.Settings.Speed != nil && *req.Settings.Speed > 0 {
		body["speed"] = *req.Settings.Speed
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/audio/speech", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("openai TTS HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return tts.NewReadCloser(newTTSResampler(resp.Body, TTSNativeSampleRate, req.SampleRate), resp.Body), nil
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
