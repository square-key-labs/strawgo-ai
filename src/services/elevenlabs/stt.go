package elevenlabs

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	DefaultSTTModel         = "scribe_v2"
	DefaultRealtimeSTTModel = "scribe_v2_realtime"
	DefaultRealtimeSTTHost  = "api.elevenlabs.io"
	CommitManual            = "manual"
	CommitVAD               = "vad"
	sttKeepaliveIdle        = 5 * time.Second
)

const (
	STTExtraTagAudioEvents       = "tag_audio_events"
	STTExtraKeyterms             = "keyterms"
	STTExtraNoVerbatim           = "no_verbatim"
	STTExtraDiarize              = "diarize"
	STTExtraVADSilenceThreshold  = "vad_silence_threshold_secs"
	STTExtraVADThreshold         = "vad_threshold"
	STTExtraMinSpeechDurationMs  = "min_speech_duration_ms"
	STTExtraMinSilenceDurationMs = "min_silence_duration_ms"
	STTExtraSecondaryLanguages   = "secondary_languages"
	STTExtraFilterBackground     = "filter_background_audio"
)

var sttPCMRates = []int{8000, 16000, 22050, 24000, 44100, 48000}

type STTConfig struct {
	APIKey         string
	Model          string
	Language       string
	TagAudioEvents *bool
	Keyterms       []string
	NoVerbatim     *bool
	Diarize        *bool
	SampleRate     int
	BaseURL        string
	HTTPClient     *http.Client
}

type STTService struct {
	*stt.Base
	apiKey  string
	baseURL string
	client  *http.Client
}

func NewSTTService(config STTConfig) *STTService {
	model := config.Model
	if model == "" {
		model = DefaultSTTModel
	}
	language := config.Language
	if language == "" {
		language = "eng"
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultHTTPURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	extra := map[string]any{}
	if config.TagAudioEvents != nil {
		extra[STTExtraTagAudioEvents] = *config.TagAudioEvents
	}
	if len(config.Keyterms) > 0 {
		extra[STTExtraKeyterms] = config.Keyterms
	}
	if config.NoVerbatim != nil {
		extra[STTExtraNoVerbatim] = *config.NoVerbatim
	}
	if config.Diarize != nil {
		extra[STTExtraDiarize] = *config.Diarize
	}
	s := &STTService{apiKey: config.APIKey, baseURL: strings.TrimRight(baseURL, "/"), client: client}
	s.Base = stt.NewSegmented("ElevenLabsSTT", s, stt.Options{
		Provider:   "elevenlabs",
		Settings:   frames.STTSettings{Model: model, Language: language, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    2010 * time.Millisecond,
	})
	return s
}

func (s *STTService) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	settings := s.Settings()
	var body bytes.Buffer
	body.Grow(len(wav) + 1024)
	w := multipart.NewWriter(&body)
	h := textproto.MIMEHeader{}
	h.Set("Content-Disposition", `form-data; name="file"; filename="audio.wav"`)
	h.Set("Content-Type", "audio/x-wav")
	part, err := w.CreatePart(h)
	if err != nil {
		return stt.Transcript{}, err
	}
	if _, err := part.Write(wav); err != nil {
		return stt.Transcript{}, err
	}
	fields := [][2]string{{"model_id", settings.Model}}
	if lang := sttLanguage(settings.Language); lang != "" {
		fields = append(fields, [2]string{"language_code", lang})
	}
	for _, key := range []string{STTExtraTagAudioEvents, STTExtraNoVerbatim, STTExtraDiarize} {
		if v, ok := settings.Extra[key].(bool); ok {
			fields = append(fields, [2]string{key, strconv.FormatBool(v)})
		}
	}
	for _, k := range sttStrings(settings.Extra[STTExtraKeyterms]) {
		fields = append(fields, [2]string{"keyterms", k})
	}
	for _, f := range fields {
		if err := w.WriteField(f[0], f[1]); err != nil {
			return stt.Transcript{}, err
		}
	}
	if err := w.Close(); err != nil {
		return stt.Transcript{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+"/v1/speech-to-text", &body)
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("xi-api-key", s.apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return stt.Transcript{}, fmt.Errorf("elevenlabs stt HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	var result struct {
		Text         string `json:"text"`
		LanguageCode string `json:"language_code"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return stt.Transcript{}, fmt.Errorf("decode elevenlabs stt response: %w", err)
	}
	if result.LanguageCode == "" {
		result.LanguageCode = "eng"
	}
	return stt.Transcript{Text: strings.TrimSpace(result.Text), Language: result.LanguageCode}, nil
}

type RealtimeSTTConfig struct {
	APIKey                   string
	Model                    string
	Language                 string
	CommitStrategy           string
	SampleRate               int
	Host                     string
	Keyterms                 []string
	NoVerbatim               *bool
	VADSilenceThresholdSecs  *float64
	VADThreshold             *float64
	MinSpeechDurationMs      *int
	MinSilenceDurationMs     *int
	SecondaryLanguages       []string
	FilterBackgroundAudio    *bool
	IncludeTimestamps        bool
	EnableLogging            bool
	IncludeLanguageDetection bool
}

type RealtimeSTTService struct {
	*stt.Base
	apiKey     string
	host       string
	commit     string
	timestamps bool
	logging    bool
	langDetect bool

	session   atomic.Pointer[reconnect.Session]
	sendRate  int
	resampler *resample.Resampler
	scratch   []byte
	prefix    []byte
	suffix    []byte
	commitMsg []byte
}

func NewRealtimeSTTService(config RealtimeSTTConfig) *RealtimeSTTService {
	model := config.Model
	if model == "" {
		model = DefaultRealtimeSTTModel
	}
	host := config.Host
	if host == "" {
		host = DefaultRealtimeSTTHost
	}
	commit := config.CommitStrategy
	if commit != CommitVAD {
		commit = CommitManual
	}
	extra := map[string]any{}
	if len(config.Keyterms) > 0 {
		extra[STTExtraKeyterms] = config.Keyterms
	}
	if config.NoVerbatim != nil {
		extra[STTExtraNoVerbatim] = *config.NoVerbatim
	}
	if config.VADSilenceThresholdSecs != nil {
		extra[STTExtraVADSilenceThreshold] = *config.VADSilenceThresholdSecs
	}
	if config.VADThreshold != nil {
		extra[STTExtraVADThreshold] = *config.VADThreshold
	}
	if config.MinSpeechDurationMs != nil {
		extra[STTExtraMinSpeechDurationMs] = *config.MinSpeechDurationMs
	}
	if config.MinSilenceDurationMs != nil {
		extra[STTExtraMinSilenceDurationMs] = *config.MinSilenceDurationMs
	}
	if len(config.SecondaryLanguages) > 0 {
		extra[STTExtraSecondaryLanguages] = config.SecondaryLanguages
	}
	if config.FilterBackgroundAudio != nil {
		extra[STTExtraFilterBackground] = *config.FilterBackgroundAudio
	}
	s := &RealtimeSTTService{
		apiKey:     config.APIKey,
		host:       sttWSBase(host),
		commit:     commit,
		timestamps: config.IncludeTimestamps,
		logging:    config.EnableLogging,
		langDetect: config.IncludeLanguageDetection,
	}
	s.Base = stt.NewStreaming("ElevenLabsRealtimeSTT", s, stt.Options{
		Provider:   "elevenlabs-realtime",
		Settings:   frames.STTSettings{Model: model, Language: config.Language, Extra: extra},
		SampleRate: config.SampleRate,
		TTFSP99:    410 * time.Millisecond,
	})
	return s
}

func (s *RealtimeSTTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	q := url.Values{}
	q.Set("model_id", settings.Model)
	if lang := sttLanguage(settings.Language); lang != "" {
		q.Set("language_code", lang)
	}
	for _, l := range sttStrings(settings.Extra[STTExtraSecondaryLanguages]) {
		q.Add("secondary_languages", sttLanguage(l))
	}
	q.Set("audio_format", "pcm_"+strconv.Itoa(s.sendRate))
	q.Set("commit_strategy", s.commit)
	for _, k := range sttStrings(settings.Extra[STTExtraKeyterms]) {
		q.Add("keyterms", k)
	}
	if v, ok := settings.Extra[STTExtraNoVerbatim].(bool); ok {
		q.Set(STTExtraNoVerbatim, strconv.FormatBool(v))
	}
	if s.timestamps {
		q.Set("include_timestamps", "true")
	}
	if s.logging {
		q.Set("enable_logging", "true")
	}
	if s.langDetect {
		q.Set("include_language_detection", "true")
	}
	if s.commit == CommitVAD {
		for _, key := range []string{STTExtraVADSilenceThreshold, STTExtraVADThreshold, STTExtraMinSpeechDurationMs, STTExtraMinSilenceDurationMs} {
			if v, ok := sttNumber(settings.Extra[key]); ok {
				q.Set(key, v)
			}
		}
	}
	if v, ok := settings.Extra[STTExtraFilterBackground].(bool); ok {
		q.Set(STTExtraFilterBackground, strconv.FormatBool(v))
	}
	header := http.Header{}
	header.Set("xi-api-key", s.apiKey)
	return s.host + "/v1/speech-to-text/realtime?" + q.Encode(), header
}

func (s *RealtimeSTTService) Connect(ctx context.Context, sink stt.Sink) error {
	rate := s.SampleRate()
	s.sendRate = 16000
	for _, r := range sttPCMRates {
		if r == rate {
			s.sendRate = rate
		}
	}
	if s.sendRate != rate {
		r, err := resample.New(rate, s.sendRate)
		if err != nil {
			return err
		}
		s.resampler = r
	}
	s.prefix = []byte(`{"message_type":"input_audio_chunk","audio_base_64":"`)
	s.suffix = []byte(`","commit":false,"sample_rate":` + strconv.Itoa(s.sendRate) + `}`)
	s.commitMsg = []byte(`{"message_type":"input_audio_chunk","audio_base_64":"","commit":true,"sample_rate":` + strconv.Itoa(s.sendRate) + `}`)
	silence := s.chunk(make([]byte, s.sendRate/10*2))
	session := reconnect.New(reconnect.Config{
		Name:          "ElevenLabsRealtimeSTT",
		Endpoint:      s.endpoint,
		OnMessage:     func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:       sink.Error,
		KeepaliveIdle: sttKeepaliveIdle,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: silence}, true
		},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *RealtimeSTTService) chunk(pcm []byte) []byte {
	buf := make([]byte, 0, len(s.prefix)+base64.StdEncoding.EncodedLen(len(pcm))+len(s.suffix))
	buf = append(buf, s.prefix...)
	buf = base64.StdEncoding.AppendEncode(buf, pcm)
	return append(buf, s.suffix...)
}

func (s *RealtimeSTTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil {
		return
	}
	pcm := audio
	if s.resampler != nil {
		s.scratch = s.resampler.ProcessBytes(s.scratch[:0], audio)
		pcm = s.scratch
	}
	if len(pcm) == 0 {
		return
	}
	session.SendDroppable(websocket.TextMessage, s.chunk(pcm))
}

func (s *RealtimeSTTService) Finalize() {
	if s.commit != CommitManual {
		return
	}
	if session := s.session.Load(); session != nil && session.Connected() {
		session.Send(websocket.TextMessage, s.commitMsg)
	}
}

func (s *RealtimeSTTService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *RealtimeSTTService) ApplySettings(frames.STTSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type realtimeSTTMessage struct {
	MessageType  string `json:"message_type"`
	Text         string `json:"text"`
	LanguageCode string `json:"language_code"`
	Error        string `json:"error"`
}

func (s *RealtimeSTTService) onMessage(sink stt.Sink, data []byte) {
	var msg realtimeSTTMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return
	}
	switch msg.MessageType {
	case "session_started":
	case "partial_transcript":
		if text := strings.TrimSpace(msg.Text); text != "" {
			sink.Transcript(stt.Transcript{Text: text, Language: msg.LanguageCode})
		}
	case "committed_transcript":
		if s.timestamps || s.langDetect {
			return
		}
		s.committed(sink, &msg)
	case "committed_transcript_with_timestamps":
		s.committed(sink, &msg)
	case "error", "auth_error", "quota_exceeded", "transcriber_error", "input_error", "invalid_request",
		"commit_throttled", "unaccepted_terms", "rate_limited", "queue_overflow", "resource_exhausted",
		"session_time_limit_exceeded", "chunk_size_exceeded", "insufficient_audio_activity":
		text := msg.Error
		if text == "" {
			text = "unknown error - " + msg.MessageType
		}
		sink.Error(errors.New("elevenlabs realtime stt: "+text), false)
	}
}

func (s *RealtimeSTTService) committed(sink stt.Sink, msg *realtimeSTTMessage) {
	if text := strings.TrimSpace(msg.Text); text != "" {
		sink.Transcript(stt.Transcript{Text: text, Final: true, Finalized: s.commit == CommitManual, Language: msg.LanguageCode})
	}
}

func sttWSBase(host string) string {
	host = strings.TrimRight(host, "/")
	if strings.HasPrefix(host, "ws://") || strings.HasPrefix(host, "wss://") {
		return host
	}
	return "wss://" + strings.TrimPrefix(host, "https://")
}

func sttStrings(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []any:
		out := make([]string, 0, len(x))
		for _, e := range x {
			if str, ok := e.(string); ok {
				out = append(out, str)
			}
		}
		return out
	}
	return nil
}

func sttNumber(v any) (string, bool) {
	switch x := v.(type) {
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64), true
	case float32:
		return strconv.FormatFloat(float64(x), 'f', -1, 32), true
	case int:
		return strconv.Itoa(x), true
	case int64:
		return strconv.FormatInt(x, 10), true
	}
	return "", false
}

func sttLanguage(lang string) string {
	if lang == "" {
		return ""
	}
	base := strings.ToLower(lang)
	if i := strings.IndexAny(base, "-_"); i > 0 {
		base = base[:i]
	}
	if code, ok := sttLanguageCodes[base]; ok {
		return code
	}
	return lang
}

var sttLanguageCodes = map[string]string{
	"af": "afr", "am": "amh", "ar": "ara", "hy": "hye", "as": "asm", "ast": "ast", "az": "aze", "be": "bel",
	"bn": "ben", "bs": "bos", "bg": "bul", "my": "mya", "yue": "yue", "ca": "cat", "ceb": "ceb", "ny": "nya",
	"hr": "hrv", "cs": "ces", "da": "dan", "nl": "nld", "en": "eng", "et": "est", "fil": "fil", "fi": "fin",
	"fr": "fra", "ff": "ful", "gl": "glg", "lg": "lug", "ka": "kat", "de": "deu", "el": "ell", "gu": "guj",
	"ha": "hau", "he": "heb", "hi": "hin", "hu": "hun", "is": "isl", "ig": "ibo", "id": "ind", "ga": "gle",
	"it": "ita", "ja": "jpn", "jv": "jav", "kea": "kea", "kn": "kan", "kk": "kaz", "km": "khm", "ko": "kor",
	"ku": "kur", "ky": "kir", "lo": "lao", "lv": "lav", "ln": "lin", "lt": "lit", "luo": "luo", "lb": "ltz",
	"mk": "mkd", "ms": "msa", "ml": "mal", "mt": "mlt", "zh": "zho", "mi": "mri", "mr": "mar", "mn": "mon",
	"ne": "nep", "nso": "nso", "no": "nor", "oc": "oci", "or": "ori", "ps": "pus", "fa": "fas", "pl": "pol",
	"pt": "por", "pa": "pan", "ro": "ron", "ru": "rus", "sr": "srp", "sn": "sna", "sd": "snd", "sk": "slk",
	"sl": "slv", "so": "som", "es": "spa", "sw": "swa", "sv": "swe", "ta": "tam", "tg": "tgk", "te": "tel",
	"th": "tha", "tr": "tur", "uk": "ukr", "umb": "umb", "ur": "urd", "uz": "uzb", "vi": "vie", "cy": "cym",
	"wo": "wol", "xh": "xho", "zu": "zul",
}
