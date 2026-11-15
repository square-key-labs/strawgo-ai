package neuphonic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

type HTTPTTSService struct {
	*tts.Base
	apiKey string
	url    string
	client *http.Client
}

func NewHTTPTTSService(config TTSConfig) *HTTPTTSService {
	u := strings.TrimRight(config.URL, "/")
	if u == "" {
		u = DefaultHTTPURL
	}
	client := config.HTTPClient
	if client == nil {
		client = &http.Client{}
	}
	s := &HTTPTTSService{apiKey: config.APIKey, url: u, client: client}
	s.Base = tts.NewHTTP("NeuphonicHTTPTTS", s, tts.Options{Settings: defaultSettings(config), SampleRate: config.SampleRate})
	return s
}

func (s *HTTPTTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	body := fields(req.Settings, req.SampleRate)
	body["text"] = req.Text
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	endpoint := s.url + "/sse/speak/" + url.PathEscape(Language(req.Settings.Language))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("X-API-KEY", s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("neuphonic HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
	}
	return tts.NewReadCloser(&sseReader{r: bufio.NewReaderSize(resp.Body, 64*1024)}, resp.Body), nil
}

type sseReader struct {
	r   *bufio.Reader
	buf []byte
	off int
}

func (s *sseReader) Read(p []byte) (int, error) {
	for s.off == len(s.buf) {
		line, err := s.r.ReadSlice('\n')
		if err == bufio.ErrBufferFull {
			var rest []byte
			rest, err = s.r.ReadBytes('\n')
			line = append(append([]byte(nil), line...), rest...)
		}
		if data, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:")); ok {
			data = bytes.TrimSpace(data)
			if len(data) > 0 && !bytes.Equal(data, []byte("[DONE]")) {
				audio, derr := decode(data)
				if derr != nil {
					return 0, derr
				}
				s.buf, s.off = audio, 0
			}
		}
		if err != nil && s.off == len(s.buf) {
			return 0, err
		}
		if err != nil {
			break
		}
	}
	n := copy(p, s.buf[s.off:])
	s.off += n
	return n, nil
}
