package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	DefaultSTTModel             = "latest_long"
	DefaultSTTLocation          = "global"
	STTExtraLanguageCodes       = "language_codes"
	STTExtraAutoPunctuation     = "enable_automatic_punctuation"
	STTExtraSpokenPunctuation   = "enable_spoken_punctuation"
	STTExtraSpokenEmojis        = "enable_spoken_emojis"
	STTExtraProfanityFilter     = "profanity_filter"
	STTExtraWordTimeOffsets     = "enable_word_time_offsets"
	STTExtraWordConfidence      = "enable_word_confidence"
	STTExtraAdaptation          = "adaptation"
	STTExtraDenoiserConfig      = "denoiser_config"
	sttCloudPlatformScope       = "https://www.googleapis.com/auth/cloud-platform"
	sttWAVHeaderSize            = 44
	sttDefaultGoogleLanguage    = "en-US"
	sttErrorBodyLimit           = 4096
	sttDefaultRecognizeDeadline = 30 * time.Second
)

var sttFeatureKeys = []struct{ extra, field string }{
	{STTExtraAutoPunctuation, "enableAutomaticPunctuation"},
	{STTExtraSpokenPunctuation, "enableSpokenPunctuation"},
	{STTExtraSpokenEmojis, "enableSpokenEmojis"},
	{STTExtraProfanityFilter, "profanityFilter"},
	{STTExtraWordTimeOffsets, "enableWordTimeOffsets"},
	{STTExtraWordConfidence, "enableWordConfidence"},
}

type STTConfig struct {
	APIKey          string
	CredentialsJSON []byte
	ProjectID       string
	Location        string
	Model           string
	Languages       []string
	SampleRate      int
	Adaptation      map[string]any
	DenoiserConfig  map[string]any
	Features        map[string]bool
	BaseURL         string
	HTTPClient      *http.Client
}

type STTService struct {
	*stt.Base
	apiKey   string
	creds    *auth.Credentials
	project  string
	location string
	baseURL  string
	client   *http.Client
}

func NewSTTService(config STTConfig) (*STTService, error) {
	location := config.Location
	if location == "" {
		location = DefaultSTTLocation
	}
	model := config.Model
	if model == "" {
		model = DefaultSTTModel
	}
	s := &STTService{apiKey: config.APIKey, project: config.ProjectID, location: location, baseURL: config.BaseURL, client: config.HTTPClient}
	if s.client == nil {
		s.client = &http.Client{Timeout: sttDefaultRecognizeDeadline}
	}
	if s.apiKey == "" {
		creds, err := credentials.DetectDefault(&credentials.DetectOptions{
			Scopes:          []string{sttCloudPlatformScope},
			CredentialsJSON: config.CredentialsJSON,
		})
		if err != nil {
			return nil, fmt.Errorf("google stt: load credentials: %w", err)
		}
		s.creds = creds
		if s.project == "" {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			project, err := creds.ProjectID(ctx)
			cancel()
			if err != nil {
				return nil, fmt.Errorf("google stt: project id: %w", err)
			}
			s.project = project
		}
	}
	if s.project == "" {
		return nil, errors.New("google stt: ProjectID is required")
	}
	if s.baseURL == "" {
		host := "speech.googleapis.com"
		if location != "global" {
			host = location + "-speech.googleapis.com"
		}
		s.baseURL = "https://" + host
	}
	s.baseURL = strings.TrimRight(s.baseURL, "/")
	extra := map[string]any{STTExtraAutoPunctuation: true}
	for k, v := range config.Features {
		extra[k] = v
	}
	if len(config.Languages) > 0 {
		extra[STTExtraLanguageCodes] = config.Languages
	}
	if config.Adaptation != nil {
		extra[STTExtraAdaptation] = config.Adaptation
	}
	if config.DenoiserConfig != nil {
		extra[STTExtraDenoiserConfig] = config.DenoiserConfig
	}
	s.Base = stt.NewSegmented("GoogleSTT", s, stt.Options{
		Provider:   "google",
		Settings:   frames.STTSettings{Model: model, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    1570 * time.Millisecond,
		MaxSegment: 55 * time.Second,
	})
	return s, nil
}

func sttLanguageCodes(settings frames.STTSettings) []string {
	var codes []string
	switch v := settings.Extra[STTExtraLanguageCodes].(type) {
	case []string:
		codes = v
	case []any:
		for _, x := range v {
			if str, ok := x.(string); ok {
				codes = append(codes, str)
			}
		}
	}
	if len(codes) == 0 && settings.Language != "" {
		codes = []string{settings.Language}
	}
	if len(codes) == 0 {
		codes = []string{sttDefaultGoogleLanguage}
	}
	return codes
}

func (s *STTService) request(wav []byte) ([]byte, []string, error) {
	settings := s.Settings()
	pcm := wav
	if len(pcm) >= sttWAVHeaderSize && string(pcm[:4]) == "RIFF" {
		pcm = pcm[sttWAVHeaderSize:]
	}
	features := map[string]any{}
	for _, f := range sttFeatureKeys {
		if v, ok := settings.Extra[f.extra].(bool); ok {
			features[f.field] = v
		}
	}
	languages := sttLanguageCodes(settings)
	cfg := map[string]any{
		"explicitDecodingConfig": map[string]any{
			"encoding":          "LINEAR16",
			"sampleRateHertz":   s.SampleRate(),
			"audioChannelCount": 1,
		},
		"languageCodes": languages,
		"model":         settings.Model,
		"features":      features,
	}
	if v, ok := settings.Extra[STTExtraAdaptation]; ok && v != nil && !strings.EqualFold(settings.Model, "telephony") {
		cfg["adaptation"] = v
	}
	if v, ok := settings.Extra[STTExtraDenoiserConfig]; ok && v != nil {
		cfg["denoiserConfig"] = v
	}
	body, err := json.Marshal(map[string]any{
		"config":  cfg,
		"content": base64.StdEncoding.EncodeToString(pcm),
	})
	return body, languages, err
}

func (s *STTService) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	body, languages, err := s.request(wav)
	if err != nil {
		return stt.Transcript{}, err
	}
	url := fmt.Sprintf("%s/v2/projects/%s/locations/%s/recognizers/_:recognize", s.baseURL, s.project, s.location)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("x-goog-api-key", s.apiKey)
	} else {
		tok, err := s.creds.Token(ctx)
		if err != nil {
			return stt.Transcript{}, fmt.Errorf("google stt: token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+tok.Value)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, sttErrorBodyLimit))
		return stt.Transcript{}, fmt.Errorf("google stt HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var result struct {
		Results []struct {
			Alternatives []struct {
				Transcript string `json:"transcript"`
			} `json:"alternatives"`
			LanguageCode string `json:"languageCode"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return stt.Transcript{}, fmt.Errorf("decode google stt response: %w", err)
	}
	var text strings.Builder
	language := languages[0]
	for _, r := range result.Results {
		if len(r.Alternatives) == 0 || r.Alternatives[0].Transcript == "" {
			continue
		}
		if text.Len() > 0 {
			text.WriteByte(' ')
		}
		text.WriteString(strings.TrimSpace(r.Alternatives[0].Transcript))
		if r.LanguageCode != "" {
			language = r.LanguageCode
		}
	}
	return stt.Transcript{Text: text.String(), Language: language}, nil
}
