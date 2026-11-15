package serializers

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func tonePCM(samples, rate int, freq float64, amp float64) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		v := int16(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out
}

func samplesOf(pcm []byte) []int16 {
	out := make([]int16, len(pcm)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
	}
	return out
}

func assertClosePCM(t *testing.T, got, want []byte, tol int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("pcm length = %d, want %d", len(got), len(want))
	}
	g, w := samplesOf(got), samplesOf(want)
	for i := range g {
		d := int(g[i]) - int(w[i])
		if d < 0 {
			d = -d
		}
		if d > tol {
			t.Fatalf("sample %d = %d, want %d (tol %d)", i, g[i], w[i], tol)
		}
	}
}

func mustEncode(t *testing.T, pcm []byte, codec string) []byte {
	t.Helper()
	b, err := audio.EncodePCM16(nil, pcm, codec)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func b64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

func jsonMsg(t *testing.T, m Message) map[string]any {
	t.Helper()
	if m.Type != TextMessage {
		t.Fatalf("message type = %v, want text", m.Type)
	}
	var out map[string]any
	if err := json.Unmarshal(m.Data, &out); err != nil {
		t.Fatalf("invalid json %q: %v", m.Data, err)
	}
	return out
}

func textMsg(t *testing.T, v any) Message {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return Text(string(b))
}

func path(m map[string]any, keys ...string) any {
	var cur any = m
	for _, k := range keys {
		mm, ok := cur.(map[string]any)
		if !ok {
			return nil
		}
		cur = mm[k]
	}
	return cur
}

func one(t *testing.T) func([]Message, error) Message {
	return func(msgs []Message, err error) Message {
		t.Helper()
		if err != nil {
			t.Fatalf("serialize error: %v", err)
		}
		if len(msgs) != 1 {
			t.Fatalf("got %d messages, want 1", len(msgs))
		}
		return msgs[0]
	}
}

func none(t *testing.T) func([]Message, error) {
	return func(msgs []Message, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("serialize error: %v", err)
		}
		if len(msgs) != 0 {
			t.Fatalf("got %d messages, want 0: %q", len(msgs), msgs[0].Data)
		}
	}
}

func audioFrame(t *testing.T, f frames.Frame, err error) *frames.AudioFrame {
	t.Helper()
	if err != nil {
		t.Fatalf("deserialize error: %v", err)
	}
	af, ok := f.(*frames.AudioFrame)
	if !ok {
		t.Fatalf("frame = %T, want *frames.AudioFrame", f)
	}
	return af
}

func assertDTMF(t *testing.T, f frames.Frame, err error, want frames.KeypadEntry) {
	t.Helper()
	if err != nil {
		t.Fatalf("deserialize error: %v", err)
	}
	d, ok := f.(*frames.InputDTMFFrame)
	if !ok {
		t.Fatalf("frame = %T, want *frames.InputDTMFFrame", f)
	}
	if d.Button != want {
		t.Fatalf("button = %q, want %q", d.Button, want)
	}
}

func assertType[T frames.Frame](t *testing.T, f frames.Frame, err error) T {
	t.Helper()
	if err != nil {
		t.Fatalf("deserialize error: %v", err)
	}
	v, ok := f.(T)
	if !ok {
		var zero T
		t.Fatalf("frame = %T, want %T", f, zero)
	}
	return v
}

type recordedRequest struct {
	Method string
	Path   string
	Header http.Header
	Body   string
	User   string
	Pass   string
	Basic  bool
}

type apiRecorder struct {
	mu     sync.Mutex
	reqs   []recordedRequest
	status int
	body   string
	srv    *httptest.Server
}

func newAPIRecorder(t *testing.T, status int, body string) *apiRecorder {
	r := &apiRecorder{status: status, body: body}
	r.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		b, _ := io.ReadAll(req.Body)
		u, p, ok := req.BasicAuth()
		r.mu.Lock()
		r.reqs = append(r.reqs, recordedRequest{Method: req.Method, Path: req.URL.EscapedPath(), Header: req.Header.Clone(), Body: string(b), User: u, Pass: p, Basic: ok})
		st, bd := r.status, r.body
		r.mu.Unlock()
		w.WriteHeader(st)
		_, _ = io.WriteString(w, bd)
	}))
	t.Cleanup(r.srv.Close)
	return r
}

func (r *apiRecorder) requests() []recordedRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]recordedRequest(nil), r.reqs...)
}

func (r *apiRecorder) only(t *testing.T) recordedRequest {
	t.Helper()
	reqs := r.requests()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	return reqs[0]
}
