package google

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
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

func TestGoogleTTS_Defaults(t *testing.T) {
	s := NewTTSService(TTSConfig{APIKey: "k"})
	st := s.Settings()
	if st.Language != DefaultLanguageCode || st.Voice != DefaultVoiceName || s.SampleRate() != DefaultSampleRate || s.Codec() != "linear16" {
		t.Fatalf("defaults: %+v %d %s", st, s.SampleRate(), s.Codec())
	}
	for enc, want := range map[AudioEncoding]string{EncodingMP3: "mp3", EncodingOggOpus: "opus", EncodingMulaw: "mulaw", EncodingAlaw: "alaw", EncodingLinear16: "linear16"} {
		if got := getCodec(enc); got != want {
			t.Errorf("%s: %s", enc, got)
		}
	}
}

func TestGoogleTTS_StreamsDecodedAudioWithoutWAVHeader(t *testing.T) {
	pcm := make([]byte, 4000)
	for i := range pcm {
		pcm[i] = byte(i%250 + 1)
	}
	wav := stt.WAV(pcm, 16000, 1)
	var gotKey string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-Goog-Api-Key")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		_ = json.NewEncoder(w).Encode(map[string]any{"audioContent": base64.StdEncoding.EncodeToString(wav), "timepoints": []any{}})
	}))
	t.Cleanup(server.Close)

	s := NewTTSService(TTSConfig{APIKey: "secret", Endpoint: server.URL})
	ctx, cancel := context.WithCancel(context.Background())
	down := newCollector("down")
	s.Link(down)
	_ = s.Start(ctx)
	_ = down.Start(ctx)
	t.Cleanup(func() { _ = s.Stop(); _ = down.Stop(); cancel() })

	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewTTSSpeakFrame("Hello & goodbye"), frames.Downstream)
	waitFor(t, "stopped", func() bool {
		for _, f := range down.frames() {
			if _, ok := f.(*frames.TTSStoppedFrame); ok {
				return true
			}
		}
		return false
	})
	var audio []byte
	chunks := 0
	for _, f := range down.frames() {
		if a, ok := f.(*frames.TTSAudioFrame); ok {
			audio = append(audio, a.Data...)
			chunks++
		}
	}
	if string(audio) != string(pcm) {
		t.Fatalf("audio mismatch: got %d bytes want %d", len(audio), len(pcm))
	}
	if chunks < 2 {
		t.Fatalf("expected chunked audio, got %d chunks", chunks)
	}
	if gotKey != "secret" {
		t.Fatalf("api key header %q", gotKey)
	}
	if gotBody["input"].(map[string]any)["text"] != "Hello & goodbye" {
		t.Fatalf("request body %v", gotBody)
	}
}
