package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type collector struct {
	*processors.BaseProcessor
	mu  sync.Mutex
	got []frames.Frame
}

func newCollector(name string) *collector {
	c := &collector{}
	c.BaseProcessor = processors.NewBaseProcessor(name, c)
	return c
}

func (c *collector) HandleFrame(_ context.Context, frame frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.got = append(c.got, frame)
	c.mu.Unlock()
	return nil
}

func (c *collector) frames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.got...)
}

func wire(t *testing.T, svc processors.FrameProcessor) (*collector, *collector) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	up, down := newCollector("up"), newCollector("down")
	up.Link(svc)
	svc.Link(down)
	for _, p := range []processors.FrameProcessor{up, svc, down} {
		if err := p.Start(ctx); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		_ = svc.Stop()
		_ = up.Stop()
		_ = down.Stop()
		cancel()
	})
	return up, down
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func count[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

type fakeCartesia struct {
	mu       sync.Mutex
	msgs     []map[string]any
	header   http.Header
	rawQuery string
}

func (f *fakeCartesia) received() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.msgs...)
}

func (f *fakeCartesia) serve(t *testing.T) string {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.header = r.Header.Clone()
		f.rawQuery = r.URL.RawQuery
		f.mu.Unlock()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var msg map[string]any
			_ = json.Unmarshal(data, &msg)
			f.mu.Lock()
			f.msgs = append(f.msgs, msg)
			f.mu.Unlock()
			id, _ := msg["context_id"].(string)
			if msg["cancel"] == true {
				continue
			}
			if transcript, _ := msg["transcript"].(string); transcript != "" {
				chunk := fmt.Sprintf(`{"type":"chunk","context_id":%q,"data":%q}`, id, base64.StdEncoding.EncodeToString(make([]byte, 320)))
				_ = conn.WriteMessage(websocket.TextMessage, []byte(chunk))
				ts := fmt.Sprintf(`{"type":"timestamps","context_id":%q,"word_timestamps":{"words":["hi","there"],"start":[0,0.2],"end":[0.2,0.4]}}`, id)
				_ = conn.WriteMessage(websocket.TextMessage, []byte(ts))
			}
			if msg["continue"] == false {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf(`{"type":"done","context_id":%q}`, id)))
			}
		}
	}))
	t.Cleanup(server.Close)
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func TestCartesiaTTS_Defaults(t *testing.T) {
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "v"})
	st := s.Settings()
	if st.Model != DefaultModel || st.Language != "en" || st.Voice != "v" || s.Codec() != "linear16" {
		t.Fatalf("defaults: %+v %s", st, s.Codec())
	}
	if c, n := encodingFormat("pcm_mulaw"); c != "mulaw" || n != 1 {
		t.Fatalf("mulaw mapping %s %d", c, n)
	}
}

func TestCartesiaTTS_HeaderAuthAndCompletionWithoutCancel(t *testing.T) {
	fake := &fakeCartesia{}
	url := fake.serve(t)
	s := NewTTSService(TTSConfig{APIKey: "secret", VoiceID: "voice", BaseURL: url, SampleRate: 16000})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame("Hi there. "), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
	waitFor(t, "turn end", func() bool { return count[*frames.LLMFullResponseEndFrame](down.frames()) == 1 })

	fake.mu.Lock()
	header, query := fake.header, fake.rawQuery
	fake.mu.Unlock()
	if header.Get("X-API-Key") != "secret" || header.Get("Cartesia-Version") == "" || strings.Contains(query, "secret") {
		t.Fatalf("auth must be in headers only: header=%v query=%q", header, query)
	}
	msgs := fake.received()
	if len(msgs) != 2 || msgs[0]["continue"] != true || msgs[1]["continue"] != false || msgs[1]["transcript"] != "" {
		t.Fatalf("unexpected messages %v", msgs)
	}
	for _, m := range msgs {
		if m["cancel"] != nil {
			t.Fatalf("cancel sent on normal completion: %v", m)
		}
	}
	fs := down.frames()
	var words []string
	for _, f := range fs {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			words = append(words, tf.Text)
		}
	}
	if strings.Join(words, " ") != "hi there" {
		t.Fatalf("word frames %v", words)
	}
	if count[*frames.TTSAudioFrame](fs) != 1 || count[*frames.TTSStoppedFrame](fs) != 1 {
		t.Fatalf("expected one audio and one stop frame")
	}
}

func TestCartesiaTTS_InterruptionCancelsContext(t *testing.T) {
	fake := &fakeCartesia{}
	url := fake.serve(t)
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "v", BaseURL: url, SampleRate: 16000})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame("Hello there. "), frames.Downstream)
	waitFor(t, "audio", func() bool { return count[*frames.TTSAudioFrame](down.frames()) == 1 })
	_ = s.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	waitFor(t, "cancel", func() bool {
		for _, m := range fake.received() {
			if m["cancel"] == true {
				return true
			}
		}
		return false
	})
	first := fake.received()[0]["context_id"]
	for _, m := range fake.received() {
		if m["cancel"] == true && m["context_id"] != first {
			t.Fatalf("cancelled wrong context: %v", m)
		}
	}
}
