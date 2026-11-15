package piper

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

var synthesisFields = []string{"speaker_id", "length_scale", "noise_scale", "noise_w_scale"}

type TTSConfig struct {
	BaseURL    string
	Voice      string
	Speed      float64
	Params     map[string]any
	SampleRate int
	HTTPClient *http.Client
}

type TTSService struct {
	*tts.Base
	url    string
	client *http.Client
}

func NewTTSService(config TTSConfig) *TTSService {
	settings := frames.TTSSettings{Voice: config.Voice}
	if config.Speed > 0 {
		speed := config.Speed
		settings.Speed = &speed
	}
	if len(config.Params) > 0 {
		settings.Extra = make(map[string]any, len(config.Params))
		for k, v := range config.Params {
			settings.Extra[k] = v
		}
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{url: strings.TrimRight(config.BaseURL, "/"), client: client}
	s.Base = tts.NewHTTP("PiperTTS", s, tts.Options{Settings: settings, SampleRate: config.SampleRate})
	return s
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	body := map[string]any{"text": req.Text}
	if req.Settings.Voice != "" {
		body["voice"] = req.Settings.Voice
	}
	for _, name := range synthesisFields {
		if v, ok := req.Settings.Extra[name]; ok && v != nil {
			body[name] = v
		}
	}
	if _, ok := body["length_scale"]; !ok && req.Settings.Speed != nil && *req.Settings.Speed > 0 {
		body["length_scale"] = 1 / *req.Settings.Speed
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url, bytes.NewReader(payload))
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
		return nil, fmt.Errorf("piper HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	audio, rate, err := readWAV(resp.Body)
	if err != nil {
		resp.Body.Close()
		return nil, err
	}
	return tts.NewReadCloser(newTTSResampler(audio, rate, req.SampleRate), resp.Body), nil
}

func readWAV(r io.Reader) (io.Reader, int, error) {
	br := bufio.NewReaderSize(r, 4096)
	head := make([]byte, 12)
	if _, err := io.ReadFull(br, head); err != nil {
		return nil, 0, fmt.Errorf("piper: read WAV header: %w", err)
	}
	if string(head[0:4]) != "RIFF" || string(head[8:12]) != "WAVE" {
		return nil, 0, errors.New("piper: response is not WAV")
	}
	rate := 0
	chunk := make([]byte, 8)
	for {
		if _, err := io.ReadFull(br, chunk); err != nil {
			return nil, 0, fmt.Errorf("piper: read WAV chunk: %w", err)
		}
		id := string(chunk[0:4])
		size := int(binary.LittleEndian.Uint32(chunk[4:8]))
		switch id {
		case "data":
			if rate == 0 {
				return nil, 0, errors.New("piper: WAV data before fmt chunk")
			}
			return br, rate, nil
		case "fmt ":
			if size < 16 {
				return nil, 0, errors.New("piper: short WAV fmt chunk")
			}
			f := make([]byte, size+size%2)
			if _, err := io.ReadFull(br, f); err != nil {
				return nil, 0, err
			}
			format, channels, bits := binary.LittleEndian.Uint16(f[0:2]), binary.LittleEndian.Uint16(f[2:4]), binary.LittleEndian.Uint16(f[14:16])
			if format != 1 || channels != 1 || bits != 16 {
				return nil, 0, fmt.Errorf("piper: unsupported WAV format %d, %d channels, %d bits", format, channels, bits)
			}
			rate = int(binary.LittleEndian.Uint32(f[4:8]))
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
