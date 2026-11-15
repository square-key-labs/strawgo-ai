package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DeepgramHTTPTTSURL  = "https://api.deepgram.com/v1/speak"
	DefaultHTTPTTSVoice = "aura-2-helena-en"
)

type HTTPTTSConfig struct {
	APIKey     string
	Voice      string
	Encoding   string
	SampleRate int
	Speed      float64
	MIPOptOut  *bool
	URL        string
	HTTPClient *http.Client
}

type HTTPTTSService struct {
	*tts.Base
	apiKey    string
	encoding  string
	mipOptOut *bool
	url       string
	client    *http.Client
}

func NewHTTPTTSService(config HTTPTTSConfig) *HTTPTTSService {
	voice := config.Voice
	if voice == "" {
		voice = DefaultHTTPTTSVoice
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = DefaultTTSEncoding
	}
	sampleRate := config.SampleRate
	bytesPerSample := 2
	if encoding == "mulaw" || encoding == "alaw" {
		bytesPerSample = 1
		if sampleRate == 0 {
			sampleRate = 8000
		}
	}
	u := config.URL
	if u == "" {
		u = DeepgramHTTPTTSURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	settings := frames.TTSSettings{Voice: voice}
	if config.Speed > 0 {
		speed := config.Speed
		settings.Speed = &speed
	}
	s := &HTTPTTSService{apiKey: config.APIKey, encoding: encoding, mipOptOut: config.MIPOptOut, url: u, client: client}
	s.Base = tts.NewHTTP("DeepgramHTTPTTS", s, tts.Options{
		Settings:       settings,
		SampleRate:     sampleRate,
		Codec:          encodingToCodec(encoding),
		BytesPerSample: bytesPerSample,
	})
	return s
}

func (s *HTTPTTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	model := req.Settings.Voice
	if model == "" {
		model = req.Settings.Model
	}
	q := url.Values{}
	q.Set("model", model)
	q.Set("encoding", s.encoding)
	q.Set("sample_rate", strconv.Itoa(req.SampleRate))
	q.Set("container", "none")
	if req.Settings.Speed != nil && *req.Settings.Speed > 0 {
		q.Set("speed", strconv.FormatFloat(*req.Settings.Speed, 'f', -1, 64))
	}
	if s.mipOptOut != nil {
		q.Set("mip_opt_out", strconv.FormatBool(*s.mipOptOut))
	}
	payload, err := json.Marshal(struct {
		Text string `json:"text"`
	}{Text: req.Text})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url+"?"+q.Encode(), bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Token "+s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("deepgram TTS HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return resp.Body, nil
}
