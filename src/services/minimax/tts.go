package minimax

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL         = "https://api.minimax.io/v1/t2a_v2"
	DefaultModel       = "speech-2.8-turbo"
	DefaultVoice       = "Calm_Woman"
	fallbackRate       = 24000
	extraVolume        = "volume"
	extraPitch         = "pitch"
	extraEmotion       = "emotion"
	extraNormalize     = "text_normalization_mode"
	extraLatexRead     = "latex_read"
	extraLanguageBoost = "language_boost"
	extraPronunciation = "pronunciation_dict"
)

var supportedRates = map[int]bool{8000: true, 16000: true, 22050: true, 24000: true, 32000: true, 44100: true}

var languages = map[string]string{
	"af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "ca": "Catalan", "cs": "Czech", "da": "Danish",
	"de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish",
	"fil": "Filipino", "fr": "French", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
	"id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nb": "Norwegian",
	"nn": "Nynorsk", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
	"sk": "Slovak", "sl": "Slovenian", "sv": "Swedish", "ta": "Tamil", "th": "Thai", "tr": "Turkish",
	"uk": "Ukrainian", "vi": "Vietnamese", "yue": "Chinese,Yue", "zh": "Chinese",
}

func Language(code string) string {
	base := strings.ToLower(code)
	if i := strings.IndexAny(base, "-_"); i > 0 {
		base = base[:i]
	}
	if mapped, ok := languages[base]; ok {
		return mapped
	}
	return code
}

type TTSConfig struct {
	APIKey     string
	GroupID    string
	Model      string
	VoiceID    string
	Language   string
	Speed      float64
	Params     map[string]any
	SampleRate int
	URL        string
	HTTPClient *http.Client
}

type TTSService struct {
	*tts.Base
	apiKey   string
	endpoint string
	client   *http.Client
}

func NewTTSService(config TTSConfig) *TTSService {
	settings := frames.TTSSettings{Model: config.Model, Voice: config.VoiceID, Language: config.Language}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if settings.Voice == "" {
		settings.Voice = DefaultVoice
	}
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
	base := config.URL
	if base == "" {
		base = DefaultURL
	}
	endpoint := base
	if config.GroupID != "" {
		sep := "?"
		if strings.Contains(base, "?") {
			sep = "&"
		}
		endpoint = base + sep + "GroupId=" + url.QueryEscape(config.GroupID)
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{apiKey: config.APIKey, endpoint: endpoint, client: client}
	s.Base = tts.NewHTTP("MiniMaxTTS", s, tts.Options{Settings: settings, SampleRate: config.SampleRate})
	return s
}

func buildPayload(req tts.Request, rate int) map[string]any {
	st := req.Settings
	speed := 1.0
	if st.Speed != nil && *st.Speed > 0 {
		speed = *st.Speed
	}
	voice := map[string]any{"voice_id": st.Voice, "speed": speed, "vol": 1.0, "pitch": 0}
	if v, ok := st.Extra[extraVolume]; ok {
		voice["vol"] = v
	}
	if v, ok := st.Extra[extraPitch]; ok {
		voice["pitch"] = v
	}
	if v, ok := st.Extra[extraEmotion]; ok {
		voice["emotion"] = v
	}
	if v, ok := st.Extra[extraLatexRead]; ok {
		voice["latex_read"] = v
	}
	payload := map[string]any{
		"stream":        true,
		"model":         st.Model,
		"text":          req.Text,
		"voice_setting": voice,
		"audio_setting": map[string]any{"bitrate": 128000, "format": "pcm", "channel": 1, "sample_rate": rate},
	}
	if v, ok := st.Extra[extraLanguageBoost]; ok {
		payload["language_boost"] = v
	} else if st.Language != "" {
		payload["language_boost"] = Language(st.Language)
	}
	if v, ok := st.Extra[extraNormalize]; ok {
		payload["text_normalization_mode"] = v
	}
	if v, ok := st.Extra[extraPronunciation]; ok {
		payload["pronunciation_dict"] = map[string]any{"tone": v}
	}
	return payload
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	rate := req.SampleRate
	if !supportedRates[rate] {
		rate = fallbackRate
	}
	payload, err := json.Marshal(buildPayload(req, rate))
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/plain, */*")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("minimax HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	audio := newTTSResampler(&sseAudio{r: bufio.NewReaderSize(resp.Body, 64*1024)}, rate, req.SampleRate)
	return tts.NewReadCloser(audio, resp.Body), nil
}

type sseEvent struct {
	Data *struct {
		Audio string `json:"audio"`
	} `json:"data"`
	ExtraInfo json.RawMessage `json:"extra_info"`
	BaseResp  *struct {
		StatusCode int    `json:"status_code"`
		StatusMsg  string `json:"status_msg"`
	} `json:"base_resp"`
}

type sseAudio struct {
	r   *bufio.Reader
	buf []byte
	off int
}

func (a *sseAudio) Read(p []byte) (int, error) {
	for a.off == len(a.buf) {
		if err := a.next(); err != nil {
			return 0, err
		}
	}
	n := copy(p, a.buf[a.off:])
	a.off += n
	return n, nil
}

func (a *sseAudio) next() error {
	line, err := a.r.ReadBytes('\n')
	if len(line) == 0 && err != nil {
		return err
	}
	line = bytes.TrimSpace(line)
	data, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok {
		if len(line) > 0 && line[0] == '{' {
			data = line
		} else {
			return nil
		}
	}
	var ev sseEvent
	if jerr := json.Unmarshal(bytes.TrimSpace(data), &ev); jerr != nil {
		return nil
	}
	if ev.BaseResp != nil && ev.BaseResp.StatusCode != 0 {
		return fmt.Errorf("minimax: %d %s", ev.BaseResp.StatusCode, ev.BaseResp.StatusMsg)
	}
	if len(ev.ExtraInfo) > 0 || ev.Data == nil || ev.Data.Audio == "" {
		return nil
	}
	n := hex.DecodedLen(len(ev.Data.Audio))
	if cap(a.buf) < n {
		a.buf = make([]byte, n)
	}
	a.buf = a.buf[:n]
	if _, herr := hex.Decode(a.buf, []byte(ev.Data.Audio)); herr != nil {
		a.buf, a.off = a.buf[:0], 0
		return errors.New("minimax: invalid hex audio")
	}
	a.off = 0
	return nil
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
