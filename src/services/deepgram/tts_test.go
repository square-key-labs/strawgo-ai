package deepgram

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type fakeSpeakServer struct {
	mu       sync.Mutex
	types    []string
	auth     string
	query    string
	holdNext bool
	conn     *websocket.Conn
}

func (f *fakeSpeakServer) received() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.types...)
}

func (f *fakeSpeakServer) handler(r *http.Request, conn *websocket.Conn) {
	f.mu.Lock()
	f.auth = r.Header.Get("Authorization")
	f.query = r.URL.RawQuery
	f.conn = conn
	f.mu.Unlock()
	for {
		_, data, err := conn.ReadMessage()
		if err != nil {
			return
		}
		var msg struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal(data, &msg)
		f.mu.Lock()
		f.types = append(f.types, msg.Type)
		hold := f.holdNext
		f.mu.Unlock()
		switch msg.Type {
		case "Speak":
			if !hold {
				_ = conn.WriteMessage(websocket.BinaryMessage, make([]byte, 320))
			}
		case "Flush":
			_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Flushed","sequence_id":0}`))
		case "Clear":
			_ = conn.WriteMessage(websocket.BinaryMessage, make([]byte, 320))
			_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"Cleared","sequence_id":0}`))
		case "Close":
			return
		}
	}
}

func indexOf(fs []frames.Frame, match func(frames.Frame) bool) int {
	for i, f := range fs {
		if match(f) {
			return i
		}
	}
	return -1
}

func isType[T frames.Frame](f frames.Frame) bool {
	_, ok := f.(T)
	return ok
}

func count[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if isType[T](f) {
			n++
		}
	}
	return n
}

func TestDeepgramTTS_Defaults(t *testing.T) {
	s := NewTTSService(TTSConfig{APIKey: "k"})
	if s.Settings().Model != DefaultTTSModel || s.encoding != DefaultTTSEncoding || s.Codec() != "linear16" {
		t.Fatalf("unexpected defaults: %+v %s %s", s.Settings(), s.encoding, s.Codec())
	}
	m := NewTTSService(TTSConfig{Encoding: "mulaw"})
	if m.Codec() != "mulaw" || m.SampleRate() != 8000 {
		t.Fatalf("mulaw format: %s %d", m.Codec(), m.SampleRate())
	}
	s.SetVoice("aura-luna-en")
	u, h := s.endpoint()
	if !strings.Contains(u, "model=aura-luna-en") || h.Get("Authorization") != "Token k" {
		t.Fatalf("endpoint %s %v", u, h)
	}
}

func TestDeepgramTTS_TurnFlushesWithoutClose(t *testing.T) {
	srv := &fakeSpeakServer{}
	url := wsServer(t, srv.handler)
	s := NewTTSService(TTSConfig{APIKey: "k", BaseURL: url, SampleRate: 16000})
	up, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame("Hello world. How"), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame(" are you?"), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)

	waitFor(t, "turn to finish", func() bool { return count[*frames.LLMFullResponseEndFrame](down.frames()) == 1 })
	fs := down.frames()
	started := indexOf(fs, isType[*frames.TTSStartedFrame])
	stopped := indexOf(fs, isType[*frames.TTSStoppedFrame])
	end := indexOf(fs, isType[*frames.LLMFullResponseEndFrame])
	firstAudio := indexOf(fs, isType[*frames.TTSAudioFrame])
	if !(started >= 0 && started < firstAudio && firstAudio < stopped && stopped < end) {
		t.Fatalf("bad ordering started=%d audio=%d stopped=%d end=%d", started, firstAudio, stopped, end)
	}
	ctxID := fs[started].(*frames.TTSStartedFrame).ContextID
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioFrame); ok && (a.ContextID != ctxID || a.Metadata()["codec"] != "linear16") {
			t.Fatalf("audio frame not stamped: %q %v", a.ContextID, a.Metadata())
		}
	}
	var spoken []string
	for _, f := range fs {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			spoken = append(spoken, tf.Text)
		}
	}
	if strings.Join(spoken, "|") != "Hello world.|How are you?" {
		t.Fatalf("spoken text %q", spoken)
	}
	if count[*frames.TTSStartedFrame](up.frames()) != 1 || count[*frames.TTSStoppedFrame](up.frames()) != 1 {
		t.Fatal("expected started/stopped upstream")
	}
	types := strings.Join(srv.received(), ",")
	if types != "Speak,Speak,Flush" {
		t.Fatalf("server received %s", types)
	}
	_ = s.QueueFrame(frames.NewEndFrame(), frames.Downstream)
	waitFor(t, "Close on EndFrame", func() bool {
		r := srv.received()
		return len(r) > 0 && r[len(r)-1] == "Close"
	})
	waitFor(t, "end frame", func() bool { return count[*frames.EndFrame](down.frames()) == 1 })
}

func TestDeepgramTTS_InterruptionClearsAndDropsStaleAudio(t *testing.T) {
	srv := &fakeSpeakServer{}
	url := wsServer(t, srv.handler)
	s := NewTTSService(TTSConfig{APIKey: "k", BaseURL: url, SampleRate: 16000})
	up, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewLLMTextFrame("First sentence. "), frames.Downstream)
	waitFor(t, "first audio", func() bool { return count[*frames.TTSAudioFrame](down.frames()) == 1 })

	_ = s.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	waitFor(t, "Clear", func() bool { return strings.Contains(strings.Join(srv.received(), ","), "Clear") })
	time.Sleep(150 * time.Millisecond)
	for _, typ := range srv.received() {
		if typ == "Flush" || typ == "Close" {
			t.Fatalf("interruption must send Clear only, got %v", srv.received())
		}
	}
	if n := count[*frames.TTSAudioFrame](down.frames()); n != 1 {
		t.Fatalf("stale audio after interruption was forwarded: %d frames", n)
	}
	if count[*frames.TTSStoppedFrame](up.frames()) != 1 {
		t.Fatal("expected TTSStoppedFrame upstream on interruption")
	}
	if count[*frames.InterruptionFrame](down.frames()) != 1 {
		t.Fatal("interruption must be forwarded")
	}

	_ = s.QueueFrame(frames.NewTTSSpeakFrame("New turn."), frames.Downstream)
	waitFor(t, "new context audio", func() bool { return count[*frames.TTSAudioFrame](down.frames()) == 2 })
	waitFor(t, "new context stopped", func() bool { return count[*frames.TTSStoppedFrame](down.frames()) == 1 })
}
