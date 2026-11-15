package azure

import (
	"bytes"
	"encoding/binary"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type azureAudio struct {
	requestID string
	hasRIFF   bool
	payload   int
}

type fakeAzure struct {
	mu        sync.Mutex
	header    http.Header
	query     string
	texts     []string
	audio     []azureAudio
	sendTurns func(conn *websocket.Conn, n int)
}

func (f *fakeAzure) snapshot() ([]string, []azureAudio) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.texts...), append([]azureAudio(nil), f.audio...)
}

func headerValue(head, name string) string {
	for _, line := range strings.Split(head, "\r\n") {
		if k, v, ok := strings.Cut(line, ":"); ok && strings.EqualFold(k, name) {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func (f *fakeAzure) serve(t *testing.T) string {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.header = r.Header.Clone()
		f.query = r.URL.RawQuery
		f.mu.Unlock()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			mt, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if mt == websocket.TextMessage {
				f.mu.Lock()
				f.texts = append(f.texts, string(data))
				f.mu.Unlock()
				continue
			}
			n := int(binary.BigEndian.Uint16(data[:2]))
			head := string(data[2 : 2+n])
			body := data[2+n:]
			if headerValue(head, "Path") != "audio" {
				t.Errorf("binary message without audio path: %q", head)
			}
			f.mu.Lock()
			f.audio = append(f.audio, azureAudio{requestID: headerValue(head, "X-RequestId"), hasRIFF: bytes.HasPrefix(body, []byte("RIFF")), payload: len(body)})
			count := len(f.audio)
			f.mu.Unlock()
			if f.sendTurns != nil {
				f.sendTurns(conn, count)
			}
		}
	}))
	t.Cleanup(server.Close)
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func azureText(path, body string) []byte {
	return []byte("X-RequestId: ABC\r\nContent-Type: application/json; charset=utf-8\r\nPath: " + path + "\r\n\r\n" + body)
}

func TestAzureSTT_Defaults(t *testing.T) {
	s := NewSTTService(STTConfig{SubscriptionKey: "k"})
	if s.Settings().Language != DefaultLanguage {
		t.Fatalf("language %q", s.Settings().Language)
	}
	u, h := s.endpoint()
	if !strings.HasPrefix(u, "wss://eastus.stt.speech.microsoft.com/") || !strings.Contains(u, "language=en-US") || !strings.Contains(u, "format=detailed") {
		t.Fatalf("url %s", u)
	}
	if h.Get("Ocp-Apim-Subscription-Key") != "k" || len(h.Get("X-ConnectionId")) != 32 {
		t.Fatalf("headers %v", h)
	}
}

func TestAzureSTT_ProtocolFramingAndTranscripts(t *testing.T) {
	fake := &fakeAzure{}
	fake.sendTurns = func(conn *websocket.Conn, n int) {
		switch n {
		case 1:
			_ = conn.WriteMessage(websocket.TextMessage, azureText("speech.hypothesis", `{"Text":"hello wor","Offset":0,"Duration":1}`))
		case 2:
			_ = conn.WriteMessage(websocket.TextMessage, azureText("speech.phrase", `{"RecognitionStatus":"Success","Offset":0,"Duration":1,"NBest":[{"Display":"Hello world."}]}`))
			_ = conn.WriteMessage(websocket.TextMessage, azureText("turn.end", `{}`))
		}
	}
	url := fake.serve(t)
	s := NewSTTService(STTConfig{SubscriptionKey: "secret", Endpoint: url, SampleRate: 16000})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	waitFor(t, "interim", func() bool { return len(transcriptFrames(down.frames())) == 1 })
	_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	waitFor(t, "final", func() bool { return len(transcriptFrames(down.frames())) == 2 })
	_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	waitFor(t, "third audio", func() bool { _, a := fake.snapshot(); return len(a) == 3 })

	ts := transcriptFrames(down.frames())
	if ts[0].IsFinal || ts[0].Text != "hello wor" || !ts[1].IsFinal || ts[1].Text != "Hello world." {
		t.Fatalf("transcripts %+v %+v", ts[0], ts[1])
	}
	texts, audio := fake.snapshot()
	if len(texts) != 1 || headerValue(texts[0], "Path") != "speech.config" || !strings.Contains(texts[0], "\r\n\r\n{") {
		t.Fatalf("expected speech.config first, got %q", texts)
	}
	if !audio[0].hasRIFF || audio[0].payload != 44+640 || audio[1].hasRIFF || audio[1].payload != 640 {
		t.Fatalf("first chunk of a turn must carry the WAV header: %+v", audio)
	}
	if audio[0].requestID != audio[1].requestID || audio[2].requestID == audio[1].requestID || !audio[2].hasRIFF {
		t.Fatalf("turn.end must start a new request: %+v", audio)
	}
	fake.mu.Lock()
	h := fake.header
	fake.mu.Unlock()
	if h.Get("Ocp-Apim-Subscription-Key") != "secret" || h.Get("X-ConnectionId") == "" {
		t.Fatalf("handshake headers %v", h)
	}
}

func transcriptFrames(fs []frames.Frame) []*frames.TranscriptionFrame {
	var out []*frames.TranscriptionFrame
	for _, f := range fs {
		switch tf := f.(type) {
		case *frames.TranscriptionFrame:
			if tf.IsFinal {
				out = append(out, tf)
			}
		case *frames.InterimTranscriptionFrame:
			out = append(out, &frames.TranscriptionFrame{Text: tf.Text, Language: tf.Language})
		}
	}
	return out
}
