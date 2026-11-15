package azure

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	AzureSTTURLTemplate = "wss://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
	DefaultRegion       = "eastus"
	DefaultLanguage     = "en-US"
	DefaultSampleRate   = 16000
	timestampLayout     = "2006-01-02T15:04:05.000Z"
)

type STTConfig struct {
	SubscriptionKey string
	Region          string
	Language        string
	SampleRate      int
	Endpoint        string
}

type STTService struct {
	*stt.Base
	subscriptionKey string
	endpointURL     string
	session         atomic.Pointer[reconnect.Session]

	mu         sync.Mutex
	requestID  string
	needHeader bool
	opened     bool
}

func NewSTTService(config STTConfig) *STTService {
	region := config.Region
	if region == "" {
		region = DefaultRegion
	}
	language := config.Language
	if language == "" {
		language = DefaultLanguage
	}
	endpoint := config.Endpoint
	if endpoint == "" {
		endpoint = fmt.Sprintf(AzureSTTURLTemplate, region)
	}
	s := &STTService{subscriptionKey: config.SubscriptionKey, endpointURL: endpoint}
	s.Base = stt.NewStreaming("AzureSTT", s, stt.Options{
		Provider:   "azure",
		Settings:   frames.STTSettings{Language: language},
		SampleRate: config.SampleRate,
		TTFSP99:    500 * time.Millisecond,
	})
	return s
}

func newID() string {
	return strings.ToUpper(strings.ReplaceAll(uuid.NewString(), "-", ""))
}

func timestamp() string {
	return time.Now().UTC().Format(timestampLayout)
}

func (s *STTService) endpoint() (string, http.Header) {
	params := url.Values{}
	params.Set("language", s.Settings().Language)
	params.Set("format", "detailed")
	header := http.Header{}
	header.Set("Ocp-Apim-Subscription-Key", s.subscriptionKey)
	header.Set("X-ConnectionId", newID())
	return s.endpointURL + "?" + params.Encode(), header
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:         "AzureSTT",
		Endpoint:     s.endpoint,
		OnOpen:       s.onOpen,
		OnMessage:    func(mt int, data []byte) { s.onMessage(sink, mt, data) },
		OnError:      sink.Error,
		DrainOnClose: true,
	})
	s.mu.Lock()
	s.requestID = newID()
	s.needHeader = true
	s.opened = false
	s.mu.Unlock()
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) onOpen(w reconnect.Writer) error {
	s.mu.Lock()
	if s.opened {
		s.requestID = newID()
		s.needHeader = true
		if session := s.session.Load(); session != nil {
			session.Clear()
		}
	}
	s.opened = true
	requestID := s.requestID
	s.mu.Unlock()
	config, err := json.Marshal(map[string]any{
		"context": map[string]any{
			"system": map[string]any{"version": "1.0.00000"},
			"os":     map[string]any{"platform": "Linux", "name": "StrawGo", "version": "1.0.0"},
			"audio": map[string]any{"source": map[string]any{
				"connectivity": "Unknown", "manufacturer": "StrawGo", "model": "StrawGo", "type": "Stream",
			}},
		},
	})
	if err != nil {
		return err
	}
	msg := "Path: speech.config\r\nX-RequestId: " + requestID + "\r\nX-Timestamp: " + timestamp() +
		"\r\nContent-Type: application/json\r\n\r\n" + string(config)
	return w.WriteMessage(websocket.TextMessage, []byte(msg))
}

func (s *STTService) audioMessage(audio []byte) ([]byte, bool) {
	header := "Path: audio\r\nX-RequestId: " + s.requestID + "\r\nX-Timestamp: " + timestamp() +
		"\r\nContent-Type: audio/x-wav\r\n"
	first := s.needHeader
	size := 2 + len(header) + len(audio)
	if first {
		size += 44
	}
	buf := make([]byte, 2, size)
	binary.BigEndian.PutUint16(buf, uint16(len(header)))
	buf = append(buf, header...)
	if first {
		buf = append(buf, stt.WAVHeader(0, s.SampleRate(), 1)...)
		s.needHeader = false
	}
	return append(buf, audio...), first
}

func (s *STTService) SendAudio(audio []byte) {
	session := s.session.Load()
	if session == nil {
		return
	}
	s.mu.Lock()
	msg, first := s.audioMessage(audio)
	if first {
		session.Send(websocket.BinaryMessage, msg)
	} else {
		session.SendDroppable(websocket.BinaryMessage, msg)
	}
	s.mu.Unlock()
}

func (s *STTService) Disconnect(graceful bool) {
	session := s.session.Load()
	if session == nil {
		return
	}
	if graceful {
		s.mu.Lock()
		if s.requestID != "" && !s.needHeader {
			msg, _ := s.audioMessage(nil)
			session.Send(websocket.BinaryMessage, msg)
		}
		s.mu.Unlock()
	}
	session.Close(graceful)
}

func (s *STTService) ApplySettings(frames.STTSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

type phrase struct {
	RecognitionStatus string `json:"RecognitionStatus"`
	DisplayText       string `json:"DisplayText"`
	Text              string `json:"Text"`
	PrimaryLanguage   *struct {
		Language string `json:"Language"`
	} `json:"PrimaryLanguage"`
	NBest []struct {
		Display string `json:"Display"`
	} `json:"NBest"`
}

func parseMessage(data []byte) (path string, body []byte, ok bool) {
	idx := bytes.Index(data, []byte("\r\n\r\n"))
	if idx < 0 {
		return "", nil, false
	}
	for _, line := range strings.Split(string(data[:idx]), "\r\n") {
		name, value, found := strings.Cut(line, ":")
		if found && strings.EqualFold(strings.TrimSpace(name), "Path") {
			path = strings.ToLower(strings.TrimSpace(value))
		}
	}
	return path, data[idx+4:], path != ""
}

func (s *STTService) onMessage(sink stt.Sink, mt int, data []byte) {
	if mt != websocket.TextMessage {
		return
	}
	path, body, ok := parseMessage(data)
	if !ok {
		return
	}
	switch path {
	case "speech.hypothesis", "speech.fragment":
		var p phrase
		if json.Unmarshal(body, &p) == nil && p.Text != "" {
			sink.Transcript(stt.Transcript{Text: p.Text})
		}
	case "speech.phrase":
		var p phrase
		if json.Unmarshal(body, &p) != nil {
			return
		}
		switch p.RecognitionStatus {
		case "Success":
			text := p.DisplayText
			if text == "" && len(p.NBest) > 0 {
				text = p.NBest[0].Display
			}
			t := stt.Transcript{Text: text, Final: true}
			if p.PrimaryLanguage != nil {
				t.Language = p.PrimaryLanguage.Language
			}
			sink.Transcript(t)
		case "Error", "BadRequest", "TooManyRequests", "Forbidden", "ServiceUnavailable":
			sink.Error(errors.New("azure STT: "+p.RecognitionStatus), false)
		}
	case "turn.end":
		s.mu.Lock()
		s.requestID = newID()
		s.needHeader = true
		s.mu.Unlock()
	}
}
