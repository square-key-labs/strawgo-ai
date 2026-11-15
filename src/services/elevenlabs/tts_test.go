package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
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

type fakeEleven struct {
	mu          sync.Mutex
	msgs        []map[string]any
	keys        []string
	connections atomic.Int32
	dropFirst   bool
}

func (f *fakeEleven) received() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.msgs...)
}

func (f *fakeEleven) serve(t *testing.T) string {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.keys = append(f.keys, r.Header.Get("xi-api-key"))
		f.mu.Unlock()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		n := f.connections.Add(1)
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
			if f.dropFirst && n == 1 {
				return
			}
			id, _ := msg["context_id"].(string)
			if text, _ := msg["text"].(string); strings.TrimSpace(text) != "" {
				resp := fmt.Sprintf(`{"audio":%q,"contextId":%q,"alignment":{"chars":["h","i"," ","y","o"],"charStartTimesMs":[0,10,20,30,40],"charDurationsMs":[10,10,10,10,10]}}`,
					base64.StdEncoding.EncodeToString(make([]byte, 320)), id)
				_ = conn.WriteMessage(websocket.TextMessage, []byte(resp))
			}
			if msg["close_context"] == true {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf(`{"isFinal":true,"contextId":%q}`, id)))
			}
		}
	}))
	t.Cleanup(server.Close)
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func speakTurn(s *TTSService, text string) {
	_ = s.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame(text), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
}

func TestElevenLabsTTS_OutputFormat(t *testing.T) {
	for format, want := range map[string]string{"pcm_24000": "24000/linear16", "ulaw_8000": "8000/mulaw", "alaw_8000": "8000/alaw", "pcm_16000": "16000/linear16"} {
		s := NewTTSService(TTSConfig{OutputFormat: format})
		if got := fmt.Sprintf("%d/%s", s.SampleRate(), s.Codec()); got != want {
			t.Errorf("%s: got %s want %s", format, got, want)
		}
	}
}

func TestElevenLabsTTS_WebsocketTurnAndWords(t *testing.T) {
	fake := &fakeEleven{}
	url := fake.serve(t)
	s := NewTTSService(TTSConfig{APIKey: "secret", VoiceID: "voice", Model: "eleven_turbo_v2_5", OutputFormat: "pcm_16000", UseStreaming: true, WebsocketURL: url})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	speakTurn(s, "Hi you. ")
	waitFor(t, "turn end", func() bool { return count[*frames.LLMFullResponseEndFrame](down.frames()) == 1 })

	msgs := fake.received()
	if len(msgs) != 4 || msgs[0]["text"] != " " || msgs[1]["text"] != "Hi you. " || msgs[2]["flush"] != true || msgs[3]["close_context"] != true {
		t.Fatalf("unexpected messages %v", msgs)
	}
	var words []string
	for _, f := range down.frames() {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			words = append(words, tf.Text)
		}
	}
	if strings.Join(words, " ") != "hi yo" {
		t.Fatalf("words %v", words)
	}
	fake.mu.Lock()
	key := fake.keys[0]
	fake.mu.Unlock()
	if key != "secret" {
		t.Fatalf("api key header %q", key)
	}
}

func TestElevenLabsTTS_ReconnectsAfterDrop(t *testing.T) {
	fake := &fakeEleven{dropFirst: true}
	url := fake.serve(t)
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "voice", OutputFormat: "pcm_16000", UseStreaming: true, WebsocketURL: url})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	waitFor(t, "first connection", func() bool { return fake.connections.Load() == 1 })
	speakTurn(s, "First. ")
	waitFor(t, "reconnect", func() bool { return fake.connections.Load() >= 2 })
	waitFor(t, "first turn end", func() bool { return count[*frames.LLMFullResponseEndFrame](down.frames()) == 1 })
	speakTurn(s, "Second. ")
	waitFor(t, "second audio", func() bool { return count[*frames.TTSAudioFrame](down.frames()) >= 1 })
	waitFor(t, "second turn end", func() bool { return count[*frames.LLMFullResponseEndFrame](down.frames()) == 2 })
	for i := 0; i < 20; i++ {
		_ = s.QueueFrame(frames.NewTTSSpeakFrame(fmt.Sprintf("Burst %d.", i)), frames.Downstream)
	}
	waitFor(t, "burst audio", func() bool { return count[*frames.TTSStoppedFrame](down.frames()) >= 21 })
}

func TestElevenLabsTTS_HTTPStreamsChunksAndCancels(t *testing.T) {
	var cancelled atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("xi-api-key") != "k" || !strings.HasSuffix(r.URL.Path, "/stream") {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		flusher := w.(http.Flusher)
		w.WriteHeader(http.StatusOK)
		chunk := make([]byte, 1280)
		for i := 0; i < 100; i++ {
			if _, err := w.Write(chunk); err != nil {
				cancelled.Store(true)
				return
			}
			flusher.Flush()
			select {
			case <-r.Context().Done():
				cancelled.Store(true)
				return
			case <-time.After(20 * time.Millisecond):
			}
		}
	}))
	t.Cleanup(server.Close)
	s := NewTTSService(TTSConfig{APIKey: "k", VoiceID: "voice", OutputFormat: "pcm_16000", HTTPURL: server.URL})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame("A long sentence. "), frames.Downstream)
	waitFor(t, "streamed chunks", func() bool { return count[*frames.TTSAudioFrame](down.frames()) >= 3 })
	_ = s.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	waitFor(t, "request cancelled", cancelled.Load)
	n := count[*frames.TTSAudioFrame](down.frames())
	time.Sleep(100 * time.Millisecond)
	if after := count[*frames.TTSAudioFrame](down.frames()); after != n {
		t.Fatalf("audio kept flowing after interruption: %d -> %d", n, after)
	}
	for _, f := range down.frames() {
		if a, ok := f.(*frames.TTSAudioFrame); ok && len(a.Data) != 1280 {
			t.Fatalf("expected 40ms chunks, got %d bytes", len(a.Data))
		}
	}
}
