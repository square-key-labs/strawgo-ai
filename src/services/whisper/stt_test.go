package whisper

import (
	"context"
	"encoding/binary"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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

func sendAudio(t *testing.T, s *STTService, down *collector, n int) {
	t.Helper()
	before := countAudio(down.frames())
	for i := 0; i < n; i++ {
		_ = s.QueueFrame(frames.NewAudioFrame(make([]byte, 640), 16000, 1), frames.Downstream)
	}
	waitFor(t, "audio passthrough", func() bool { return countAudio(down.frames()) == before+n })
}

func countType[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

func countAudio(fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(*frames.AudioFrame); ok {
			n++
		}
	}
	return n
}

func transcriptFrames(fs []frames.Frame) []*frames.TranscriptionFrame {
	var out []*frames.TranscriptionFrame
	for _, f := range fs {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			out = append(out, tf)
		}
	}
	return out
}

type upload struct {
	model, language string
	wav             []byte
}

func fakeWhisper(t *testing.T, status int, reply string) (string, func() []upload) {
	var mu sync.Mutex
	var uploads []upload
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		data, _ := io.ReadAll(file)
		mu.Lock()
		uploads = append(uploads, upload{model: r.FormValue("model"), language: r.FormValue("language"), wav: data})
		mu.Unlock()
		w.WriteHeader(status)
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(server.Close)
	return server.URL, func() []upload {
		mu.Lock()
		defer mu.Unlock()
		return append([]upload(nil), uploads...)
	}
}

func TestWhisper_Defaults(t *testing.T) {
	s := NewSTTService(STTConfig{APIKey: "key"})
	if s.Settings().Model != DefaultModel || s.apiURL != WhisperAPIURL {
		t.Fatalf("defaults %+v %s", s.Settings(), s.apiURL)
	}
	s.SetLanguage("fr")
	s.SetModel("whisper-2")
	if s.Settings().Language != "fr" || s.Settings().Model != "whisper-2" {
		t.Fatalf("settings %+v", s.Settings())
	}
}

func TestWhisper_TranscribesVADSegmentWithPreRollAndTrailingSilence(t *testing.T) {
	url, uploads := fakeWhisper(t, http.StatusOK, `{"text":" hello world "}`)
	s := NewSTTService(STTConfig{APIKey: "key", Language: "en", SampleRate: 16000, BaseURL: url})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	sendAudio(t, s, down, 50)
	_ = s.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	waitFor(t, "user started", func() bool { return countType[*frames.UserStartedSpeakingFrame](down.frames()) == 1 })
	sendAudio(t, s, down, 10)
	_ = s.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	waitFor(t, "transcript", func() bool { return len(transcriptFrames(down.frames())) == 1 })

	tf := transcriptFrames(down.frames())[0]
	if !tf.IsFinal || tf.Text != "hello world" {
		t.Fatalf("transcript %+v", tf)
	}
	up := uploads()
	if len(up) != 1 || up[0].model != DefaultModel || up[0].language != "en" {
		t.Fatalf("uploads %+v", up)
	}
	wav := up[0].wav
	if string(wav[0:4]) != "RIFF" || binary.LittleEndian.Uint32(wav[24:28]) != 16000 {
		t.Fatal("expected a 16 kHz WAV upload")
	}
	dataLen := int(binary.LittleEndian.Uint32(wav[40:44]))
	want := 16000 + 10*640 + 16000
	if dataLen != want {
		t.Fatalf("segment = 500ms pre-roll + speech + 500ms trailing silence: got %d bytes, want %d", dataLen, want)
	}
	if audio := countAudio(down.frames()); audio != 60 {
		t.Fatalf("audio passthrough: %d", audio)
	}
}

func TestWhisper_InterruptionDoesNotDropSpeech(t *testing.T) {
	url, uploads := fakeWhisper(t, http.StatusOK, `{"text":"still here"}`)
	s := NewSTTService(STTConfig{APIKey: "key", SampleRate: 16000, BaseURL: url})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	waitFor(t, "interruption", func() bool { return countType[*frames.InterruptionFrame](down.frames()) == 1 })
	sendAudio(t, s, down, 1)
	_ = s.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	waitFor(t, "transcript", func() bool { return len(transcriptFrames(down.frames())) == 1 })
	if len(uploads()) != 1 {
		t.Fatal("expected one upload")
	}
}

func TestWhisper_APIErrorBecomesErrorFrame(t *testing.T) {
	url, _ := fakeWhisper(t, http.StatusInternalServerError, `{"error":"boom"}`)
	s := NewSTTService(STTConfig{APIKey: "key", SampleRate: 16000, BaseURL: url})
	up, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	waitFor(t, "user started", func() bool { return countType[*frames.UserStartedSpeakingFrame](down.frames()) == 1 })
	sendAudio(t, s, down, 1)
	_ = s.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	waitFor(t, "error", func() bool {
		for _, f := range up.frames() {
			if e, ok := f.(*frames.ErrorFrame); ok && strings.Contains(e.Error.Error(), "500") {
				return true
			}
		}
		return false
	})
}

func TestWhisper_EndFrameWaitsForQueuedSegment(t *testing.T) {
	var release atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for !release.Load() {
			time.Sleep(5 * time.Millisecond)
		}
		_, _ = w.Write([]byte(`{"text":"late"}`))
	}))
	t.Cleanup(server.Close)
	s := NewSTTService(STTConfig{APIKey: "key", SampleRate: 16000, BaseURL: server.URL})
	_, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	waitFor(t, "user started", func() bool { return countType[*frames.UserStartedSpeakingFrame](down.frames()) == 1 })
	sendAudio(t, s, down, 1)
	_ = s.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	waitFor(t, "user stopped", func() bool { return countType[*frames.UserStoppedSpeakingFrame](down.frames()) == 1 })
	_ = s.QueueFrame(frames.NewEndFrame(), frames.Downstream)
	time.Sleep(50 * time.Millisecond)
	release.Store(true)
	waitFor(t, "end frame", func() bool {
		fs := down.frames()
		return len(fs) > 0 && func() bool { _, ok := fs[len(fs)-1].(*frames.EndFrame); return ok }()
	})
	fs := down.frames()
	if len(transcriptFrames(fs)) != 1 {
		t.Fatal("queued segment must be transcribed before EndFrame")
	}
}
