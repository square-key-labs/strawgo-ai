package deepgram

import (
	"context"
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

func wsServer(t *testing.T, handler func(r *http.Request, conn *websocket.Conn)) string {
	t.Helper()
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		handler(r, conn)
	}))
	t.Cleanup(server.Close)
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func transcripts(fs []frames.Frame) []*frames.TranscriptionFrame {
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

func TestDeepgramSTT_EncodingNormalization(t *testing.T) {
	for in, want := range map[string]string{"ulaw": "mulaw", "PCMU": "mulaw", "PCMA": "alaw", "pcm": "linear16", "linear16": "linear16", "": "linear16"} {
		s := NewSTTService(STTConfig{Encoding: in})
		if s.encoding != want {
			t.Errorf("encoding %q: got %q want %q", in, s.encoding, want)
		}
	}
	if NewSTTService(STTConfig{Encoding: "mulaw"}).SampleRate() != 8000 {
		t.Error("mulaw should default to 8 kHz")
	}
}

func TestDeepgramSTT_SettingsAndHeaderAuth(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "secret", Model: "nova-2", Language: "en"})
	s.SetLanguage("es")
	s.SetModel("nova-3")
	u, h := s.endpoint()
	if strings.Contains(u, "secret") {
		t.Fatal("API key must not be in the URL")
	}
	if h.Get("Authorization") != "Token secret" {
		t.Fatalf("unexpected auth header %q", h.Get("Authorization"))
	}
	if !strings.Contains(u, "language=es") || !strings.Contains(u, "model=nova-3") {
		t.Fatalf("settings not applied: %s", u)
	}
}

func TestDeepgramSTT_TranscriptsAndFinalize(t *testing.T) {
	finalized := make(chan struct{}, 1)
	url := wsServer(t, func(_ *http.Request, conn *websocket.Conn) {
		sent := false
		for {
			mt, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if mt == websocket.BinaryMessage && !sent {
				sent = true
				_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Results","is_final":false,"channel":{"alternatives":[{"transcript":"hel"}]}}`))
			}
			if mt == websocket.TextMessage && strings.Contains(string(data), "Finalize") {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Results","is_final":true,"from_finalize":true,"channel":{"alternatives":[{"transcript":"hello"}]}}`))
				finalized <- struct{}{}
			}
		}
	})
	s := NewSTTService(STTConfig{APIKey: "k", BaseURL: url, SampleRate: 16000})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	waitFor(t, "interim", func() bool { return len(transcripts(down.frames())) == 1 })
	_ = s.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	select {
	case <-finalized:
	case <-time.After(5 * time.Second):
		t.Fatal("Finalize not sent on user stop")
	}
	waitFor(t, "final", func() bool { return len(transcripts(down.frames())) == 2 })
	ts := transcripts(down.frames())
	if ts[0].IsFinal || !ts[1].IsFinal || ts[1].Text != "hello" {
		t.Fatalf("unexpected transcripts: %+v %+v", ts[0], ts[1])
	}
	audioSeen := false
	for _, f := range down.frames() {
		if _, ok := f.(*frames.AudioFrame); ok {
			audioSeen = true
		}
	}
	if !audioSeen {
		t.Fatal("audio must pass through")
	}
}

func TestDeepgramSTT_WriteFailureDoesNotDeadlockAndReconnects(t *testing.T) {
	var connections atomic.Int32
	url := wsServer(t, func(_ *http.Request, conn *websocket.Conn) {
		if connections.Add(1) == 1 {
			_, _, _ = conn.ReadMessage()
			return
		}
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	})
	s := NewSTTService(STTConfig{APIKey: "k", BaseURL: url, SampleRate: 16000})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	for i := 0; i < 200; i++ {
		_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	}
	waitFor(t, "audio to flow through", func() bool {
		n := 0
		for _, f := range down.frames() {
			if _, ok := f.(*frames.AudioFrame); ok {
				n++
			}
		}
		return n == 200
	})
	waitFor(t, "reconnect", func() bool { return connections.Load() >= 2 })
	_ = s.QueueFrame(frames.NewEndFrame(), frames.Downstream)
	waitFor(t, "end frame", func() bool {
		for _, f := range down.frames() {
			if _, ok := f.(*frames.EndFrame); ok {
				return true
			}
		}
		return false
	})
}

func TestDeepgramSTT_MuteDropsAudio(t *testing.T) {
	var binary atomic.Int32
	url := wsServer(t, func(_ *http.Request, conn *websocket.Conn) {
		for {
			mt, _, err := conn.ReadMessage()
			if err != nil {
				return
			}
			if mt == websocket.BinaryMessage {
				binary.Add(1)
			}
		}
	})
	s := NewSTTService(STTConfig{APIKey: "k", BaseURL: url, SampleRate: 16000})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewSTTMuteFrame(true), frames.Downstream)
	waitFor(t, "mute", s.Muted)
	for i := 0; i < 5; i++ {
		_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	}
	countAudio := func() int {
		n := 0
		for _, f := range down.frames() {
			if _, ok := f.(*frames.AudioFrame); ok {
				n++
			}
		}
		return n
	}
	waitFor(t, "muted audio passthrough", func() bool { return countAudio() == 5 })
	_ = s.QueueFrame(frames.NewSTTMuteFrame(false), frames.Downstream)
	waitFor(t, "unmute", func() bool { return !s.Muted() })
	_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	waitFor(t, "unmuted audio", func() bool { return binary.Load() == 1 })
	time.Sleep(100 * time.Millisecond)
	if binary.Load() != 1 {
		t.Fatalf("muted audio was sent: %d", binary.Load())
	}
}
