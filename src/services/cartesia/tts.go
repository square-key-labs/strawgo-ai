package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultURL     = "wss://api.cartesia.ai/tts/websocket"
	DefaultVersion = "2025-04-16"
	DefaultModel   = "sonic-3"
)

type GenerationConfig struct {
	Volume  float64 `json:"volume,omitempty"`
	Speed   float64 `json:"speed,omitempty"`
	Emotion string  `json:"emotion,omitempty"`
}

type TTSConfig struct {
	APIKey              string
	VoiceID             string
	Model               string
	CartesiaVersion     string
	Language            string
	SampleRate          int
	Encoding            string
	Container           string
	GenerationConfig    *GenerationConfig
	AggregateSentences  bool
	StreamTokens        bool
	PronunciationDictID string
	BaseURL             string
}

type TTSService struct {
	*tts.Base
	apiKey     string
	version    string
	encoding   string
	container  string
	generation *GenerationConfig
	dictID     string
	baseURL    string
	session    atomic.Pointer[reconnect.Session]
}

func NewTTSService(config TTSConfig) *TTSService {
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	version := config.CartesiaVersion
	if version == "" {
		version = DefaultVersion
	}
	language := config.Language
	if language == "" {
		language = "en"
	}
	encoding := config.Encoding
	if encoding == "" {
		encoding = "pcm_s16le"
	}
	container := config.Container
	if container == "" {
		container = "raw"
	}
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultURL
	}
	codec, bytesPerSample := encodingFormat(encoding)
	aggregation := tts.AggregateSentences
	if config.StreamTokens {
		aggregation = tts.AggregateTokens
	}
	s := &TTSService{
		apiKey:     config.APIKey,
		version:    version,
		encoding:   encoding,
		container:  container,
		generation: config.GenerationConfig,
		dictID:     config.PronunciationDictID,
		baseURL:    baseURL,
	}
	s.Base = tts.NewStreaming("CartesiaTTS", s, tts.Options{
		Settings:       frames.TTSSettings{Model: model, Voice: config.VoiceID, Language: language},
		SampleRate:     config.SampleRate,
		Codec:          codec,
		BytesPerSample: bytesPerSample,
		Aggregation:    aggregation,
		WordTimestamps: true,
	})
	return s
}

func encodingFormat(encoding string) (string, int) {
	switch encoding {
	case "pcm_mulaw":
		return "mulaw", 1
	case "pcm_alaw":
		return "alaw", 1
	case "pcm_f32le":
		return "float32", 4
	default:
		return "linear16", 2
	}
}

func (s *TTSService) endpoint() (string, http.Header) {
	header := http.Header{}
	header.Set("X-API-Key", s.apiKey)
	header.Set("Cartesia-Version", s.version)
	return s.baseURL, header
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:         "CartesiaTTS",
		Endpoint:     s.endpoint,
		OnMessage:    func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:      sink.Error,
		PingInterval: 20 * time.Second,
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

type voice struct {
	Mode string `json:"mode"`
	ID   string `json:"id"`
}

type outputFormat struct {
	Container  string `json:"container"`
	Encoding   string `json:"encoding"`
	SampleRate int    `json:"sample_rate"`
}

type request struct {
	Transcript          string            `json:"transcript"`
	Continue            bool              `json:"continue"`
	ContextID           string            `json:"context_id"`
	ModelID             string            `json:"model_id"`
	Voice               voice             `json:"voice"`
	OutputFormat        outputFormat      `json:"output_format"`
	Language            string            `json:"language,omitempty"`
	AddTimestamps       bool              `json:"add_timestamps"`
	GenerationConfig    *GenerationConfig `json:"generation_config,omitempty"`
	PronunciationDictID string            `json:"pronunciation_dict_id,omitempty"`
}

func (s *TTSService) buildRequest(contextID, text string, cont bool) request {
	settings := s.Settings()
	gen := s.generation
	if settings.Speed != nil {
		g := GenerationConfig{Speed: *settings.Speed}
		if gen != nil {
			g.Volume, g.Emotion = gen.Volume, gen.Emotion
		}
		gen = &g
	}
	return request{
		Transcript:          text,
		Continue:            cont,
		ContextID:           contextID,
		ModelID:             settings.Model,
		Voice:               voice{Mode: "id", ID: settings.Voice},
		OutputFormat:        outputFormat{Container: s.container, Encoding: s.encoding, SampleRate: s.SampleRate()},
		Language:            settings.Language,
		AddTimestamps:       true,
		GenerationConfig:    gen,
		PronunciationDictID: s.dictID,
	}
}

func (s *TTSService) send(v any) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("cartesia not connected")
	}
	return session.SendJSON(v)
}

func (s *TTSService) Speak(contextID, text string) error {
	return s.send(s.buildRequest(contextID, text, true))
}

func (s *TTSService) Flush(contextID string) error {
	return s.send(s.buildRequest(contextID, "", false))
}

func (s *TTSService) Cancel(contextID string) error {
	return s.send(struct {
		ContextID string `json:"context_id"`
		Cancel    bool   `json:"cancel"`
	}{ContextID: contextID, Cancel: true})
}

type response struct {
	Type           string `json:"type"`
	ContextID      string `json:"context_id"`
	Data           string `json:"data"`
	Error          string `json:"error"`
	WordTimestamps *struct {
		Words []string  `json:"words"`
		Start []float64 `json:"start"`
	} `json:"word_timestamps"`
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var resp response
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}
	switch resp.Type {
	case "chunk":
		if resp.Data == "" {
			return
		}
		audio, err := base64.StdEncoding.DecodeString(resp.Data)
		if err != nil {
			sink.Error(err, false)
			return
		}
		sink.Audio(resp.ContextID, audio)
	case "timestamps":
		ts := resp.WordTimestamps
		if ts == nil || len(ts.Words) != len(ts.Start) {
			return
		}
		words := make([]tts.Word, len(ts.Words))
		for i, w := range ts.Words {
			words[i] = tts.Word{Text: w, Start: time.Duration(ts.Start[i] * float64(time.Second))}
		}
		sink.Words(resp.ContextID, words)
	case "done":
		sink.Done(resp.ContextID)
	case "error":
		sink.Error(errors.New("cartesia: "+resp.Error), false)
		if resp.ContextID != "" {
			sink.Done(resp.ContextID)
		}
	}
}
