package azure

import (
	"bytes"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	AzureTTSURLTemplate = "https://%s.tts.speech.microsoft.com/cognitiveservices/v1"
	DefaultVoice        = "en-US-JennyNeural"
	DefaultOutputFormat = "raw-16khz-16bit-mono-pcm"
)

type TTSConfig struct {
	SubscriptionKey string
	Region          string
	Voice           string
	Language        string
	OutputFormat    string
	Endpoint        string
	HTTPClient      *http.Client
}

type TTSService struct {
	*tts.Base
	subscriptionKey string
	endpoint        string
	outputFormat    string
	httpClient      *http.Client
}

func NewTTSService(config TTSConfig) *TTSService {
	region := config.Region
	if region == "" {
		region = DefaultRegion
	}
	voice := config.Voice
	if voice == "" {
		voice = DefaultVoice
	}
	language := config.Language
	if language == "" {
		language = DefaultLanguage
	}
	outputFormat := config.OutputFormat
	if outputFormat == "" {
		outputFormat = DefaultOutputFormat
	}
	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = fmt.Sprintf(AzureTTSURLTemplate, region)
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &TTSService{
		subscriptionKey: config.SubscriptionKey,
		endpoint:        endpoint,
		outputFormat:    outputFormat,
		httpClient:      client,
	}
	sampleRate, _ := parseOutputFormat(outputFormat)
	codec := getCodec(outputFormat)
	bytesPerSample := 2
	if codec == "mulaw" || codec == "alaw" {
		bytesPerSample = 1
	}
	s.Base = tts.NewHTTP("AzureTTS", s, tts.Options{
		Settings:       frames.TTSSettings{Voice: voice, Language: language},
		SampleRate:     sampleRate,
		Codec:          codec,
		BytesPerSample: bytesPerSample,
	})
	return s
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	ssml := buildSSML(req.Text, req.Settings)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, s.endpoint, strings.NewReader(ssml))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Ocp-Apim-Subscription-Key", s.subscriptionKey)
	httpReq.Header.Set("Content-Type", "application/ssml+xml")
	httpReq.Header.Set("X-Microsoft-OutputFormat", s.outputFormat)
	httpReq.Header.Set("User-Agent", "strawgo")
	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("azure TTS HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	if strings.HasPrefix(s.outputFormat, "riff-") {
		return tts.NewReadCloser(tts.SkipWAVHeader(resp.Body), resp.Body), nil
	}
	return resp.Body, nil
}

func escapeXML(s string) string {
	var buf bytes.Buffer
	_ = xml.EscapeText(&buf, []byte(s))
	return buf.String()
}

func buildSSML(text string, settings frames.TTSSettings) string {
	language := settings.Language
	if language == "" {
		language = DefaultLanguage
	}
	voice := settings.Voice
	if voice == "" {
		voice = DefaultVoice
	}
	body := escapeXML(text)
	if settings.Speed != nil {
		body = fmt.Sprintf("<prosody rate='%s'>%s</prosody>", strconv.FormatFloat(*settings.Speed, 'f', 2, 64), body)
	}
	return fmt.Sprintf("<speak version='1.0' xml:lang='%s' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts'><voice name='%s'>%s</voice></speak>",
		escapeXML(language), escapeXML(voice), body)
}

var rateRe = regexp.MustCompile(`(\d+)(khz|hz)`)

func parseOutputFormat(format string) (sampleRate int, channels int) {
	sampleRate, channels = 16000, 1
	m := rateRe.FindStringSubmatch(format)
	if m == nil {
		return sampleRate, channels
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		return sampleRate, channels
	}
	if m[2] == "khz" {
		n *= 1000
	}
	return n, channels
}

func getCodec(format string) string {
	switch {
	case strings.HasSuffix(format, "mulaw"):
		return "mulaw"
	case strings.HasSuffix(format, "alaw"):
		return "alaw"
	case strings.HasSuffix(format, "mp3"):
		return "mp3"
	case strings.HasSuffix(format, "opus"):
		return "opus"
	default:
		return "linear16"
	}
}
