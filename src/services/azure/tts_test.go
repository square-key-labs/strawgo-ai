package azure

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
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

func TestAzureTTS_FormatParsing(t *testing.T) {
	cases := map[string]string{
		"raw-8khz-8bit-mono-mulaw":        "8000/mulaw",
		"raw-16khz-16bit-mono-pcm":        "16000/linear16",
		"riff-24khz-16bit-mono-pcm":       "24000/linear16",
		"riff-22050hz-16bit-mono-pcm":     "22050/linear16",
		"audio-24khz-48kbitrate-mono-mp3": "24000/mp3",
		"ogg-48khz-16bit-mono-opus":       "48000/opus",
	}
	for format, want := range cases {
		rate, _ := parseOutputFormat(format)
		if got := strconv.Itoa(rate) + "/" + getCodec(format); got != want {
			t.Errorf("%s: got %s want %s", format, got, want)
		}
	}
}

func TestAzureTTS_SSMLEscapesText(t *testing.T) {
	speed := 1.2
	ssml := buildSSML(`Tom & "Jerry" <3`, frames.TTSSettings{Voice: "en-US-AriaNeural", Language: "en-US", Speed: &speed})
	if !strings.Contains(ssml, "Tom &amp; &#34;Jerry&#34; &lt;3") {
		t.Fatalf("text not escaped: %s", ssml)
	}
	if !strings.Contains(ssml, "<voice name='en-US-AriaNeural'>") || !strings.Contains(ssml, "rate='1.20'") {
		t.Fatalf("voice/prosody missing: %s", ssml)
	}
}

func TestAzureTTS_StripsRIFFHeaderAndStreams(t *testing.T) {
	pcm := make([]byte, 3200)
	for i := range pcm {
		pcm[i] = byte(i%200 + 1)
	}
	var gotFormat, gotKey string
	var gotSSML string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotFormat = r.Header.Get("X-Microsoft-OutputFormat")
		gotKey = r.Header.Get("Ocp-Apim-Subscription-Key")
		body, _ := io.ReadAll(r.Body)
		gotSSML = string(body)
		_, _ = w.Write(stt.WAV(pcm, 16000, 1))
	}))
	t.Cleanup(server.Close)
	s := NewTTSService(TTSConfig{SubscriptionKey: "secret", OutputFormat: "riff-16khz-16bit-mono-pcm", Endpoint: server.URL})
	up, down := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewTTSSpeakFrame("Hello"), frames.Downstream)
	waitFor(t, "stopped", func() bool {
		for _, f := range down.frames() {
			if _, ok := f.(*frames.TTSStoppedFrame); ok {
				return true
			}
		}
		return false
	})
	var audio []byte
	for _, f := range down.frames() {
		if a, ok := f.(*frames.TTSAudioFrame); ok {
			audio = append(audio, a.Data...)
		}
	}
	if string(audio) != string(pcm) {
		t.Fatalf("audio mismatch (%d bytes, want %d)", len(audio), len(pcm))
	}
	if gotFormat != "riff-16khz-16bit-mono-pcm" || gotKey != "secret" || !strings.Contains(gotSSML, ">Hello<") {
		t.Fatalf("request: %s %s %s", gotFormat, gotKey, gotSSML)
	}
	started := 0
	for _, f := range up.frames() {
		if _, ok := f.(*frames.TTSStartedFrame); ok {
			started++
		}
	}
	if started != 1 {
		t.Fatalf("expected TTSStartedFrame upstream, got %d", started)
	}
}

func TestAzureTTS_ErrorPropagation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "denied", http.StatusUnauthorized)
	}))
	t.Cleanup(server.Close)
	s := NewTTSService(TTSConfig{SubscriptionKey: "bad", Endpoint: server.URL})
	up, _ := wire(t, s)
	_ = s.QueueFrame(frames.NewStartFrame(), frames.Downstream)
	_ = s.QueueFrame(frames.NewTTSSpeakFrame("Hello"), frames.Downstream)
	waitFor(t, "error frame", func() bool {
		for _, f := range up.frames() {
			if e, ok := f.(*frames.ErrorFrame); ok && strings.Contains(e.Error.Error(), "401") {
				return true
			}
		}
		return false
	})
}
