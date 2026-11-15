package google

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
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	GoogleTTSURL        = "https://texttospeech.googleapis.com/v1/text:synthesize"
	DefaultLanguageCode = "en-US"
	DefaultVoiceName    = "en-US-Neural2-C"
	DefaultSampleRate   = 16000
)

type VoiceGender string

const (
	GenderUnspecified VoiceGender = "SSML_VOICE_GENDER_UNSPECIFIED"
	GenderMale        VoiceGender = "MALE"
	GenderFemale      VoiceGender = "FEMALE"
	GenderNeutral     VoiceGender = "NEUTRAL"
)

type AudioEncoding string

const (
	EncodingLinear16 AudioEncoding = "LINEAR16"
	EncodingMP3      AudioEncoding = "MP3"
	EncodingOggOpus  AudioEncoding = "OGG_OPUS"
	EncodingMulaw    AudioEncoding = "MULAW"
	EncodingAlaw     AudioEncoding = "ALAW"
)

type TTSConfig struct {
	APIKey       string
	LanguageCode string
	VoiceName    string
	Gender       VoiceGender
	Encoding     AudioEncoding
	SampleRate   int
	Endpoint     string
	HTTPClient   *http.Client
}

type TTSService struct {
	*tts.Base
	apiKey     string
	gender     VoiceGender
	encoding   AudioEncoding
	endpoint   string
	httpClient *http.Client
}

func NewTTSService(config TTSConfig) *TTSService {
	languageCode := config.LanguageCode
	if languageCode == "" {
		languageCode = DefaultLanguageCode
	}
	voiceName := config.VoiceName
	if voiceName == "" {
		voiceName = DefaultVoiceName
	}
	gender := config.Gender
	if gender == "" {
		gender = GenderFemale
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = EncodingLinear16
	}
	sampleRate := config.SampleRate
	if sampleRate == 0 {
		sampleRate = DefaultSampleRate
	}
	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = GoogleTTSURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{
		apiKey:     config.APIKey,
		gender:     gender,
		encoding:   encoding,
		endpoint:   endpoint,
		httpClient: client,
	}
	bytesPerSample := 2
	if encoding == EncodingMulaw || encoding == EncodingAlaw {
		bytesPerSample = 1
	}
	s.Base = tts.NewHTTP("GoogleTTS", s, tts.Options{
		Settings:       frames.TTSSettings{Voice: voiceName, Language: languageCode},
		SampleRate:     sampleRate,
		Codec:          getCodec(encoding),
		BytesPerSample: bytesPerSample,
	})
	return s
}

func getCodec(encoding AudioEncoding) string {
	switch encoding {
	case EncodingMP3:
		return "mp3"
	case EncodingOggOpus:
		return "opus"
	case EncodingMulaw:
		return "mulaw"
	case EncodingAlaw:
		return "alaw"
	default:
		return "linear16"
	}
}

func (s *TTSService) buildRequest(req tts.Request) map[string]any {
	audioConfig := map[string]any{
		"audioEncoding":   string(s.encoding),
		"sampleRateHertz": req.SampleRate,
	}
	if req.Settings.Speed != nil {
		audioConfig["speakingRate"] = *req.Settings.Speed
	}
	return map[string]any{
		"input": map[string]string{"text": req.Text},
		"voice": map[string]any{
			"languageCode": req.Settings.Language,
			"name":         req.Settings.Voice,
			"ssmlGender":   string(s.gender),
		},
		"audioConfig": audioConfig,
	}
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	payload, err := json.Marshal(s.buildRequest(req))
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		httpReq.Header.Set("X-Goog-Api-Key", s.apiKey)
	}
	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("google TTS HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var audio io.Reader = base64.NewDecoder(base64.StdEncoding, newFieldReader(resp.Body, "audioContent"))
	if s.encoding == EncodingLinear16 {
		audio = tts.SkipWAVHeader(audio)
	}
	return tts.NewReadCloser(audio, resp.Body), nil
}

type fieldReader struct {
	r     *bufio.Reader
	key   []byte
	found bool
	done  bool
}

func newFieldReader(r io.Reader, key string) *fieldReader {
	return &fieldReader{r: bufio.NewReaderSize(r, 32*1024), key: []byte(`"` + key + `"`)}
}

func (f *fieldReader) seek() error {
	matched := 0
	for matched < len(f.key) {
		c, err := f.r.ReadByte()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return errors.New("google TTS response has no audioContent")
			}
			return err
		}
		switch {
		case c == f.key[matched]:
			matched++
		case c == f.key[0]:
			matched = 1
		default:
			matched = 0
		}
	}
	for {
		c, err := f.r.ReadByte()
		if err != nil {
			return err
		}
		if c == '"' {
			return nil
		}
		if c != ':' && c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return fmt.Errorf("unexpected %q after audioContent key", c)
		}
	}
}

func (f *fieldReader) Read(p []byte) (int, error) {
	if !f.found {
		if err := f.seek(); err != nil {
			return 0, err
		}
		f.found = true
	}
	if f.done {
		return 0, io.EOF
	}
	n := 0
	for n < len(p) {
		c, err := f.r.ReadByte()
		if err != nil {
			if errors.Is(err, io.EOF) {
				err = io.ErrUnexpectedEOF
			}
			return n, err
		}
		if c == '"' {
			f.done = true
			break
		}
		if c == '\\' {
			esc, err := f.r.ReadByte()
			if err != nil {
				return n, err
			}
			if esc != '/' {
				continue
			}
			c = '/'
		}
		p[n] = c
		n++
		if f.r.Buffered() == 0 && n > 0 {
			break
		}
	}
	if n == 0 && f.done {
		return 0, io.EOF
	}
	return n, nil
}
