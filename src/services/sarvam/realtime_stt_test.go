package sarvam_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/sarvam"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

type recorder struct {
	mu          sync.Mutex
	transcripts []stt.Transcript
	errs        []string
	fatal       []bool
	pushed      []string
}

func (r *recorder) Transcript(t stt.Transcript) {
	r.mu.Lock()
	r.transcripts = append(r.transcripts, t)
	r.mu.Unlock()
}

func (r *recorder) Push(f frames.Frame, _ frames.FrameDirection) {
	r.mu.Lock()
	r.pushed = append(r.pushed, fmt.Sprintf("%T", f))
	r.mu.Unlock()
}

func (r *recorder) Error(err error, fatal bool) {
	r.mu.Lock()
	r.errs = append(r.errs, err.Error())
	r.fatal = append(r.fatal, fatal)
	r.mu.Unlock()
}

func realtimeServer(t *testing.T, script func(c *websocket.Conn, r *http.Request)) string {
	t.Helper()
	up := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := up.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer c.Close()
		script(c, r)
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http") + "/speech-to-text-realtime/ws"
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestRealtimeSTTValidatesModelAndSampleRate(t *testing.T) {
	for _, model := range []string{"", "saaras:v3-realtime", "saaras:v4"} {
		if _, err := sarvam.NewRealtimeSTTService(sarvam.RealtimeSTTConfig{Model: model}); err != nil {
			t.Errorf("model %q rejected: %v", model, err)
		}
	}
	if _, err := sarvam.NewRealtimeSTTService(sarvam.RealtimeSTTConfig{Model: "saaras:v4-realtime"}); err == nil {
		t.Error("model saaras:v4-realtime accepted, want an error")
	}
	if _, err := sarvam.NewRealtimeSTTService(sarvam.RealtimeSTTConfig{SampleRate: 24000}); err == nil {
		t.Error("sample rate 24000 accepted, want an error")
	}
}

func TestRealtimeSTTTranscriptRoundTrip(t *testing.T) {
	requests := make(chan *http.Request, 1)
	audio := make(chan string, 1)
	url := realtimeServer(t, func(c *websocket.Conn, r *http.Request) {
		requests <- r
		_ = c.WriteMessage(websocket.TextMessage, []byte(`{"event":"session.begin","request_id":"r1"}`))
		if _, msg, err := c.ReadMessage(); err == nil {
			audio <- string(msg)
		}
		for _, m := range []string{
			`{"event":"vad.speech_start"}`,
			`{"event":"transcript.partial","text":" namaste ","language":"hi"}`,
			`{"event":"vad.speech_end"}`,
			`{"event":"transcript.final","text":"namaste duniya","language":"hi-IN"}`,
			`{"event":"error","message":"bad audio"}`,
		} {
			_ = c.WriteMessage(websocket.TextMessage, []byte(m))
		}
		for {
			if _, _, err := c.ReadMessage(); err != nil {
				return
			}
		}
	})
	svc, err := sarvam.NewRealtimeSTTService(sarvam.RealtimeSTTConfig{APIKey: "k1", Language: "hi", BaseURL: url})
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := svc.Connect(ctx, rec); err != nil {
		t.Fatal(err)
	}
	defer svc.Disconnect(true)

	req := <-requests
	if req.URL.Path != "/speech-to-text-realtime/ws" {
		t.Errorf("path = %q", req.URL.Path)
	}
	if got := req.Header.Get("API-SUBSCRIPTION-KEY"); got != "k1" {
		t.Errorf("API-SUBSCRIPTION-KEY = %q", got)
	}
	q := req.URL.Query()
	for key, want := range map[string]string{
		"model":         "saaras:v3-realtime",
		"language_code": "hi-IN",
		"encoding":      "linear16",
		"sample_rate":   "16000",
		"endpointing":   "vad",
		"stream_type":   "balanced",
	} {
		if got := q.Get(key); got != want {
			t.Errorf("query %s = %q, want %q", key, got, want)
		}
	}

	svc.SendAudio(make([]byte, 1600))
	select {
	case msg := <-audio:
		if !strings.HasPrefix(msg, `{"event":"audio_input","audio":"`) {
			t.Errorf("audio message = %.60s", msg)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("server received no audio")
	}

	eventually(t, "two transcripts and one error", func() bool {
		rec.mu.Lock()
		defer rec.mu.Unlock()
		return len(rec.transcripts) >= 2 && len(rec.errs) >= 1
	})
	rec.mu.Lock()
	defer rec.mu.Unlock()
	interim, final := rec.transcripts[0], rec.transcripts[1]
	if interim.Final || interim.Text != "namaste" || interim.Language != "hi-IN" {
		t.Errorf("interim = %+v", interim)
	}
	if !final.Final || !final.Finalized || final.Text != "namaste duniya" || final.Language != "hi-IN" {
		t.Errorf("final = %+v", final)
	}
	if got := strings.Join(rec.pushed, ","); got != "*frames.UserStartedSpeakingFrame,*frames.UserStoppedSpeakingFrame" {
		t.Errorf("pushed frames = %s", got)
	}
	if len(rec.errs) != 1 || rec.fatal[0] || !strings.Contains(rec.errs[0], "bad audio") {
		t.Errorf("errors = %v fatal = %v", rec.errs, rec.fatal)
	}
}

func TestRealtimeSTTRejectedSettingsAreFatalAndNotRetried(t *testing.T) {
	url := realtimeServer(t, func(c *websocket.Conn, _ *http.Request) {
		_ = c.WriteMessage(websocket.TextMessage, []byte(`{"event":"transcript.partial","text":"half said"}`))
		_ = c.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(4000, "unknown language"), time.Now().Add(time.Second))
		time.Sleep(100 * time.Millisecond)
	})
	svc, err := sarvam.NewRealtimeSTTService(sarvam.RealtimeSTTConfig{APIKey: "k", BaseURL: url})
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_ = svc.Connect(ctx, rec)
	defer svc.Disconnect(false)

	eventually(t, "a fatal error", func() bool {
		rec.mu.Lock()
		defer rec.mu.Unlock()
		for i, fatal := range rec.fatal {
			if fatal && strings.Contains(rec.errs[i], "unknown language") {
				return true
			}
		}
		return false
	})
	time.Sleep(500 * time.Millisecond)
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.errs) != 1 {
		t.Errorf("errors = %v, want one and no retry", rec.errs)
	}
	if len(rec.transcripts) != 2 {
		t.Fatalf("transcripts = %+v, want the partial and its promotion", rec.transcripts)
	}
	last := rec.transcripts[1]
	if !last.Final || last.Finalized || last.Text != "half said" {
		t.Errorf("promoted transcript = %+v", last)
	}
}
