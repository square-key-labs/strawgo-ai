package assemblyai

import (
	"context"
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

func wire(t *testing.T, svc processors.FrameProcessor) *collector {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	down := newCollector("down")
	svc.Link(down)
	for _, p := range []processors.FrameProcessor{svc, down} {
		if err := p.Start(ctx); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		_ = svc.Stop()
		_ = down.Stop()
		cancel()
	})
	return down
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

type fakeAssembly struct {
	mu       sync.Mutex
	auth     string
	query    string
	binary   []int
	texts    []string
	onBinary func(conn *websocket.Conn, n int)
}

func (f *fakeAssembly) snapshot() ([]int, []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]int(nil), f.binary...), append([]string(nil), f.texts...)
}

func (f *fakeAssembly) serve(t *testing.T) string {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.auth = r.Header.Get("Authorization")
		f.query = r.URL.RawQuery
		f.mu.Unlock()
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Begin","id":"s1","expires_at":0}`))
		for {
			mt, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			f.mu.Lock()
			if mt == websocket.BinaryMessage {
				f.binary = append(f.binary, len(data))
			} else {
				f.texts = append(f.texts, string(data))
			}
			n := len(f.binary)
			f.mu.Unlock()
			if mt == websocket.BinaryMessage && f.onBinary != nil {
				f.onBinary(conn, n)
			}
			if strings.Contains(string(data), "ForceEndpoint") {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Turn","transcript":"hello world","end_of_turn":true,"turn_is_formatted":true}`))
			}
			if strings.Contains(string(data), "Terminate") {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Termination","audio_duration_seconds":1,"session_duration_seconds":1}`))
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	return "ws" + strings.TrimPrefix(server.URL, "http")
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

func TestAssemblyAI_Defaults(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "k"})
	if s.Settings().Model != DefaultModel || s.baseURL != DefaultBaseURL || !s.formatTurns || !s.vadEndpoint {
		t.Fatalf("defaults %+v", s.Settings())
	}
	if s.SampleRate() != DefaultSampleRate {
		t.Fatalf("sample rate %d", s.SampleRate())
	}
	s.SetLanguage("es")
	s.SetModel("universal-streaming-english")
	u, h := s.endpoint()
	if strings.Contains(u, "token=") || strings.Contains(u, "=k&") || h.Get("Authorization") != "k" {
		t.Fatalf("api key must be a header: %s %v", u, h)
	}
	if !strings.Contains(u, "language_code=es") || !strings.Contains(u, "speech_model=universal-streaming-english") || !strings.Contains(u, "encoding=pcm_s16le") {
		t.Fatalf("url %s", u)
	}
}

func TestAssemblyAI_ChunksAudioAndEmitsTurns(t *testing.T) {
	ended := make(chan string, 1)
	fake := &fakeAssembly{onBinary: func(conn *websocket.Conn, n int) {
		if n == 1 {
			_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Turn","transcript":"hello","end_of_turn":false,"turn_is_formatted":false}`))
			_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Turn","transcript":"hello world","end_of_turn":true,"turn_is_formatted":false}`))
		}
	}}
	url := fake.serve(t)
	s := NewSTTService(STTConfig{APIKey: "secret", SampleRate: 16000, BaseURL: url, OnEndOfTurn: func(tr string) { ended <- tr }})
	down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	for i := 0; i < 5; i++ {
		_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	}
	waitFor(t, "interim", func() bool { return len(transcriptFrames(down.frames())) == 1 })
	binary, _ := fake.snapshot()
	if len(binary) != 1 || binary[0] < 1600 || binary[0] >= 1600+640 {
		t.Fatalf("audio must be sent in 50ms chunks, got %v", binary)
	}
	ts := transcriptFrames(down.frames())
	if ts[0].IsFinal || ts[0].Text != "hello" {
		t.Fatalf("interim %+v", ts[0])
	}
	_ = s.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	waitFor(t, "forced final", func() bool { return len(transcriptFrames(down.frames())) == 2 })
	ts = transcriptFrames(down.frames())
	if !ts[1].IsFinal || ts[1].Text != "hello world" {
		t.Fatalf("final %+v", ts[1])
	}
	select {
	case tr := <-ended:
		if tr != "hello world" {
			t.Fatalf("end of turn %q", tr)
		}
	case <-time.After(time.Second):
		t.Fatal("OnEndOfTurn not called")
	}
	_, texts := fake.snapshot()
	if len(texts) != 1 || !strings.Contains(texts[0], "ForceEndpoint") {
		t.Fatalf("expected ForceEndpoint on VAD stop, got %v", texts)
	}
	fake.mu.Lock()
	auth, query := fake.auth, fake.query
	fake.mu.Unlock()
	if auth != "secret" || strings.Contains(query, "secret") {
		t.Fatalf("auth header %q query %q", auth, query)
	}
	_ = s.QueueFrame(frames.NewEndFrame(), frames.Downstream)
	waitFor(t, "terminate", func() bool {
		_, texts := fake.snapshot()
		return len(texts) == 2 && strings.Contains(texts[1], "Terminate")
	})
}

func TestAssemblyAI_ConnectionFailureReportsError(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "k", BaseURL: "ws://127.0.0.1:1"})
	ctx, cancel := context.WithCancel(context.Background())
	up := newCollector("up")
	up.Link(s)
	_ = up.Start(ctx)
	_ = s.Start(ctx)
	t.Cleanup(func() { _ = s.Stop(); _ = up.Stop(); cancel() })
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	waitFor(t, "error frame", func() bool {
		for _, f := range up.frames() {
			if _, ok := f.(*frames.ErrorFrame); ok {
				return true
			}
		}
		return false
	})
}
