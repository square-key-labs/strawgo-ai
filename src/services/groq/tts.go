package groq

import (
	"bufio"
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
	DefaultTTSModel = "canopylabs/orpheus-v1-english"
	DefaultTTSVoice = "autumn"
)

type TTSConfig struct {
	APIKey     string
	BaseURL    string
	Model      string
	Voice      string
	Speed      float64
	SampleRate int
	HTTPClient *http.Client
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
	speed := config.Speed
	if speed <= 0 {
		speed = 1
	}
	settings.Speed = &speed
	baseURL := strings.TrimRight(config.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{apiKey: config.APIKey, baseURL: baseURL, client: client}
	s.Base = tts.NewHTTP("GroqTTS", s, tts.Options{Settings: settings, SampleRate: config.SampleRate})
	return s
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	body := map[string]any{
		"model":           req.Settings.Model,
		"voice":           req.Settings.Voice,
		"response_format": "wav",
		"input":           req.Text,
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
		return nil, fmt.Errorf("groq TTS HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	pcm, rate, err := ttsReadWAV(resp.Body)
	if err != nil {
		resp.Body.Close()
		return nil, err
	}
	return tts.NewReadCloser(newTTSResampler(pcm, rate, req.SampleRate), resp.Body), nil
}

func ttsReadWAV(r io.Reader) (io.Reader, int, error) {
	br := bufio.NewReaderSize(r, 4096)
	head := make([]byte, 12)
	if _, err := io.ReadFull(br, head); err != nil {
		return nil, 0, fmt.Errorf("groq TTS wav header: %w", err)
	}
	if string(head[0:4]) != "RIFF" || string(head[8:12]) != "WAVE" {
		return nil, 0, errors.New("groq TTS response is not WAV")
	}
	rate := 0
	chunk := make([]byte, 8)
	for {
		if _, err := io.ReadFull(br, chunk); err != nil {
			return nil, 0, fmt.Errorf("groq TTS wav header: %w", err)
		}
		size := int(binary.LittleEndian.Uint32(chunk[4:8]))
		switch string(chunk[0:4]) {
		case "data":
			if rate == 0 {
				return nil, 0, errors.New("groq TTS wav has no fmt chunk")
			}
			return br, rate, nil
		case "fmt ":
			fmtChunk := make([]byte, size+size%2)
			if _, err := io.ReadFull(br, fmtChunk); err != nil || size < 16 {
				return nil, 0, errors.New("groq TTS wav fmt chunk invalid")
			}
			channels := binary.LittleEndian.Uint16(fmtChunk[2:4])
			bits := binary.LittleEndian.Uint16(fmtChunk[14:16])
			if binary.LittleEndian.Uint16(fmtChunk[0:2]) != 1 || channels != 1 || bits != 16 {
				return nil, 0, fmt.Errorf("groq TTS wav format unsupported: %d channels, %d bits", channels, bits)
			}
			rate = int(binary.LittleEndian.Uint32(fmtChunk[4:8]))
		default:
			if _, err := br.Discard(size + size%2); err != nil {
				return nil, 0, err
			}
		}
	}
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
