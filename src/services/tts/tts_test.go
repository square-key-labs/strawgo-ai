package tts

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type speakCall struct {
	id   string
	text string
}

type fakeTTS struct {
	mu          sync.Mutex
	sink        Sink
	speaks      []speakCall
	flushes     []string
	cancels     []string
	disconnects []bool
	applied     []frames.TTSSettings
	hold        bool
	words       bool
	speakErr    error
	connected   chan struct{}
	flushed     chan string
}

func newFakeTTS() *fakeTTS {
	return &fakeTTS{connected: make(chan struct{}), flushed: make(chan string, 16)}
}

func (f *fakeTTS) Connect(_ context.Context, sink Sink) error {
	f.mu.Lock()
	f.sink = sink
	f.mu.Unlock()
	close(f.connected)
	return nil
}

func (f *fakeTTS) Disconnect(graceful bool) {
	f.mu.Lock()
	f.disconnects = append(f.disconnects, graceful)
	f.mu.Unlock()
}

func (f *fakeTTS) Speak(id, text string) error {
	f.mu.Lock()
	f.speaks = append(f.speaks, speakCall{id, text})
	n := len(f.speaks)
	hold, words, sink, err := f.hold, f.words, f.sink, f.speakErr
	f.mu.Unlock()
	if err != nil {
		return err
	}
	if !hold {
		sink.Audio(id, bytes.Repeat([]byte{byte(n)}, 320))
		if words {
			sink.Words(id, []Word{{Text: "Hello", Start: 0}, {Text: "world.", Start: 400 * time.Millisecond}})
		}
	}
	return nil
}

func (f *fakeTTS) Flush(id string) error {
	f.mu.Lock()
	f.flushes = append(f.flushes, id)
	hold, sink := f.hold, f.sink
	f.mu.Unlock()
	f.flushed <- id
	if !hold {
		sink.Done(id)
	}
	return nil
}

func (f *fakeTTS) Cancel(id string) error {
	f.mu.Lock()
	f.cancels = append(f.cancels, id)
	f.mu.Unlock()
	return nil
}

func (f *fakeTTS) ApplySettings(s frames.TTSSettings) error {
	f.mu.Lock()
	f.applied = append(f.applied, s)
	f.mu.Unlock()
	return nil
}

func (f *fakeTTS) speakCalls() []speakCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]speakCall(nil), f.speaks...)
}

func (f *fakeTTS) getSink(t *testing.T) Sink {
	t.Helper()
	select {
	case <-f.connected:
	case <-time.After(guard):
		t.Fatal("provider never connected")
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.sink
}

func (f *fakeTTS) waitFlush(t *testing.T) string {
	t.Helper()
	select {
	case id := <-f.flushed:
		return id
	case <-time.After(guard):
		t.Fatal("no flush")
		return ""
	}
}

func relevant(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		switch v := f.(type) {
		case *frames.TTSStartedFrame:
			out = append(out, "started")
		case *frames.TTSAudioFrame:
			out = append(out, "audio")
		case *frames.TTSTextFrame:
			out = append(out, "text:"+v.Text)
		case *frames.TTSStoppedFrame:
			out = append(out, "stopped")
		case *frames.LLMFullResponseStartFrame:
			out = append(out, "llm-start")
		case *frames.LLMFullResponseEndFrame:
			out = append(out, "llm-end")
		case *frames.EndFrame:
			out = append(out, "end")
		case *frames.LLMTextFrame:
			out = append(out, "llm-text:"+v.Text)
		}
	}
	return out
}

func indexOf(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return -1
}

func lastIndexOf(xs []string, x string) int {
	for i := len(xs) - 1; i >= 0; i-- {
		if xs[i] == x {
			return i
		}
	}
	return -1
}

func llmTurn(r *rig, deltas ...string) {
	r.send(frames.NewLLMFullResponseStartFrame())
	for _, d := range deltas {
		r.send(frames.NewLLMTextFrame(d))
	}
	r.send(frames.NewLLMFullResponseEndFrame())
}

func TestSentenceAggregationFromDeltas(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	llmTurn(r, "Hello wor", "ld. How are", " you? I am", " fine")
	r.down.waitFor(t, "llm end", atLeast[*frames.LLMFullResponseEndFrame](1))
	calls := p.speakCalls()
	want := []string{"Hello world.", "How are you?", "I am fine"}
	if len(calls) != len(want) {
		t.Fatalf("speaks = %+v", calls)
	}
	for i, w := range want {
		if calls[i].text != w {
			t.Fatalf("speak %d = %q want %q", i, calls[i].text, w)
		}
		if calls[i].id != calls[0].id {
			t.Fatal("turn sentences used different context ids")
		}
	}
}

func TestSentenceAggregationKeepsAbbreviationsAndNumbers(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	llmTurn(r, "Dr. Smith paid $3.50 at 9 a.m. today. ", "Then he left!")
	r.down.waitFor(t, "llm end", atLeast[*frames.LLMFullResponseEndFrame](1))
	calls := p.speakCalls()
	if len(calls) != 2 || calls[0].text != "Dr. Smith paid $3.50 at 9 a.m. today." || calls[1].text != "Then he left!" {
		t.Fatalf("speaks = %+v", calls)
	}
}

func TestTokenAggregationSpeaksEachDelta(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{Aggregation: AggregateTokens}), nil)
	llmTurn(r, "Hel", "lo ", "there")
	fs := r.down.waitFor(t, "llm end", atLeast[*frames.LLMFullResponseEndFrame](1))
	calls := p.speakCalls()
	if len(calls) != 3 || calls[1].text != "lo " {
		t.Fatalf("speaks = %+v", calls)
	}
	texts := ofType[*frames.TTSTextFrame](fs)
	if len(texts) != 3 || texts[0].TextType != frames.TextTypeToken {
		t.Fatalf("text frames = %v", relevant(fs))
	}
}

func TestTurnFrameOrder(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	llmTurn(r, "Hello world. ", "How are you?")
	fs := r.down.waitFor(t, "llm end", atLeast[*frames.LLMFullResponseEndFrame](1))
	got := relevant(fs)
	want := []string{"llm-start", "started", "audio", "text:Hello world.", "audio", "text:How are you?", "stopped", "llm-end"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("order\n got %v\nwant %v", got, want)
	}
	audio := ofType[*frames.TTSAudioFrame](fs)
	id := p.speakCalls()[0].id
	for _, a := range audio {
		if a.ContextID != id || a.SampleRate != DefaultSampleRate {
			t.Fatalf("audio ctx=%q rate=%d", a.ContextID, a.SampleRate)
		}
	}
	if stopped := ofType[*frames.TTSStoppedFrame](fs)[0]; stopped.ContextID != id {
		t.Fatalf("stopped ctx = %q", stopped.ContextID)
	}
	up := r.up.waitFor(t, "upstream started/stopped", atLeast[*frames.TTSStoppedFrame](1))
	if u := relevant(up); strings.Join(u, "|") != "started|stopped" {
		t.Fatalf("upstream = %v", u)
	}
}

func TestSampleRateFromStartFrame(t *testing.T) {
	p := newFakeTTS()
	start := frames.NewStartFrame()
	start.AudioOutSampleRate = 16000
	b := NewStreaming("tts", p, Options{})
	r := newRig(t, b, start)
	r.send(frames.NewTTSSpeakFrame("Hi."))
	fs := r.down.waitFor(t, "audio", atLeast[*frames.TTSAudioFrame](1))
	if a := ofType[*frames.TTSAudioFrame](fs)[0]; a.SampleRate != 16000 || b.SampleRate() != 16000 {
		t.Fatalf("rate = %d", a.SampleRate)
	}
}

func TestLLMEndWaitsForTurnAudio(t *testing.T) {
	p := newFakeTTS()
	p.hold = true
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	s := p.getSink(t)
	llmTurn(r, "Hi there.")
	id := p.waitFlush(t)
	time.Sleep(20 * time.Millisecond)
	if n := len(ofType[*frames.LLMFullResponseEndFrame](r.down.snapshot())); n != 0 {
		t.Fatalf("LLMFullResponseEndFrame forwarded before audio: %v", relevant(r.down.snapshot()))
	}
	s.Audio(id, make([]byte, 320))
	s.Audio(id, make([]byte, 320))
	s.Done(id)
	fs := r.down.waitFor(t, "llm end", atLeast[*frames.LLMFullResponseEndFrame](1))
	got := relevant(fs)
	if end, last := indexOf(got, "llm-end"), lastIndexOf(got, "audio"); end < last || indexOf(got, "stopped") > end {
		t.Fatalf("order = %v", got)
	}
}

func TestTTSSpeakFrame(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	r.send(frames.NewTTSSpeakFrame("  Welcome back.  "))
	fs := r.down.waitFor(t, "stopped", atLeast[*frames.TTSStoppedFrame](1))
	if got := strings.Join(relevant(fs), "|"); got != "started|audio|text:Welcome back.|stopped" {
		t.Fatalf("frames = %s", got)
	}
	if n := len(ofType[*frames.TTSSpeakFrame](fs)); n != 0 {
		t.Fatal("TTSSpeakFrame forwarded")
	}
	calls := p.speakCalls()
	if len(calls) != 1 || calls[0].text != "Welcome back." {
		t.Fatalf("speaks = %+v", calls)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.flushes) != 1 || p.flushes[0] != calls[0].id {
		t.Fatalf("flushes = %v", p.flushes)
	}
}

func TestTextFrameOutsideTurnSpokenImmediately(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	r.send(frames.NewTextFrame("No punctuation here"))
	r.down.waitFor(t, "stopped", atLeast[*frames.TTSStoppedFrame](1))
	if calls := p.speakCalls(); len(calls) != 1 || calls[0].text != "No punctuation here" {
		t.Fatalf("speaks = %+v", calls)
	}
}

func TestSkipTTSPassesThrough(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	f := frames.NewLLMTextFrame("silent text.")
	f.SkipTTS = true
	r.send(f)
	fs := r.down.waitFor(t, "skip frame", atLeast[*frames.LLMTextFrame](1))
	if ofType[*frames.LLMTextFrame](fs)[0] != f {
		t.Fatal("different frame forwarded")
	}
	if len(p.speakCalls()) != 0 {
		t.Fatal("SkipTTS text spoken")
	}
}

func TestInterruptionCancelsAndDropsStaleAudio(t *testing.T) {
	p := newFakeTTS()
	p.hold = true
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	s := p.getSink(t)
	r.send(frames.NewLLMFullResponseStartFrame())
	r.send(frames.NewLLMTextFrame("This is a long answer. "))
	r.up.waitFor(t, "started upstream", atLeast[*frames.TTSStartedFrame](1))
	old := p.speakCalls()[0].id
	s.Audio(old, make([]byte, 320))
	r.down.waitFor(t, "first audio", atLeast[*frames.TTSAudioFrame](1))

	r.send(frames.NewInterruptionFrame())
	r.down.waitFor(t, "interruption", atLeast[*frames.InterruptionFrame](1))
	r.up.waitFor(t, "stopped upstream", atLeast[*frames.TTSStoppedFrame](1))
	p.mu.Lock()
	cancels := append([]string(nil), p.cancels...)
	p.mu.Unlock()
	if len(cancels) != 1 || cancels[0] != old {
		t.Fatalf("cancels = %v want [%s]", cancels, old)
	}

	s.Audio(old, make([]byte, 320))
	s.Words(old, []Word{{Text: "stale"}})
	s.Done(old)

	p.mu.Lock()
	p.hold = false
	p.mu.Unlock()
	llmTurn(r, "Fresh start.")
	fs := r.down.waitFor(t, "new turn end", atLeast[*frames.LLMFullResponseEndFrame](1))
	calls := p.speakCalls()
	fresh := calls[len(calls)-1].id
	if fresh == old {
		t.Fatal("context id reused after interruption")
	}
	cut := -1
	for i, f := range fs {
		if _, ok := f.(*frames.InterruptionFrame); ok {
			cut = i
		}
	}
	for _, f := range fs[cut:] {
		switch v := f.(type) {
		case *frames.TTSAudioFrame:
			if v.ContextID == old {
				t.Fatal("stale audio delivered after interruption")
			}
		case *frames.TTSTextFrame:
			if v.ContextID == old {
				t.Fatal("stale text delivered after interruption")
			}
		}
	}
	if n := len(ofType[*frames.TTSAudioFrame](fs[cut:])); n != 1 {
		t.Fatalf("audio after interruption = %d, want 1", n)
	}
}

func TestInterruptionWithoutActiveSpeechSendsNoStopped(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	r.send(frames.NewInterruptionFrame())
	r.down.waitFor(t, "interruption", atLeast[*frames.InterruptionFrame](1))
	r.systemBarrier(t)
	if n := len(ofType[*frames.TTSStoppedFrame](r.up.snapshot())); n != 0 {
		t.Fatal("TTSStoppedFrame without active speech")
	}
}

func TestWordTimestampsBecomeTextFrames(t *testing.T) {
	p := newFakeTTS()
	p.words = true
	r := newRig(t, NewStreaming("tts", p, Options{WordTimestamps: true}), nil)
	r.send(frames.NewTTSSpeakFrame("Hello world."))
	fs := r.down.waitFor(t, "stopped", atLeast[*frames.TTSStoppedFrame](1))
	texts := ofType[*frames.TTSTextFrame](fs)
	if len(texts) != 2 {
		t.Fatalf("text frames = %v", relevant(fs))
	}
	if texts[0].Text != "Hello" || texts[1].Text != "world." || texts[1].TextType != frames.TextTypeWord || !texts[1].WillBeSpoken {
		t.Fatalf("words = %+v %+v", texts[0], texts[1])
	}
	if v, _ := texts[1].Metadata()["word_start_time"].(float64); v != 0.4 {
		t.Fatalf("word_start_time = %v", texts[1].Metadata()["word_start_time"])
	}
}

func TestMetrics(t *testing.T) {
	p := newFakeTTS()
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	start.EnableUsageMetrics = true
	r := newRig(t, NewStreaming("tts", p, Options{Settings: frames.TTSSettings{Model: "sonic"}}), start)
	r.send(frames.NewTTSSpeakFrame("Héllo"))
	r.send(frames.NewTTSSpeakFrame("Again"))
	fs := r.down.waitFor(t, "two turns", atLeast[*frames.TTSStoppedFrame](2))
	usage := metrics[frames.TTSUsageMetricsData](fs)
	if len(usage) != 2 || usage[0].Value != 5 || usage[0].Model != "sonic" || usage[0].Processor != "tts" {
		t.Fatalf("usage = %+v", usage)
	}
	if ttfb := metrics[frames.TTFBMetricsData](fs); len(ttfb) != 2 || ttfb[0].Value < 0 || ttfb[0].Value > guard {
		t.Fatalf("ttfb = %+v", ttfb)
	}
}

func TestReportOnlyInitialTTFB(t *testing.T) {
	p := newFakeTTS()
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	start.ReportOnlyInitialTTFB = true
	r := newRig(t, NewStreaming("tts", p, Options{}), start)
	r.send(frames.NewTTSSpeakFrame("One"))
	r.send(frames.NewTTSSpeakFrame("Two"))
	fs := r.down.waitFor(t, "two turns", atLeast[*frames.TTSStoppedFrame](2))
	if n := len(metrics[frames.TTFBMetricsData](fs)); n != 1 {
		t.Fatalf("ttfb count = %d", n)
	}
}

func TestProviderErrors(t *testing.T) {
	p := newFakeTTS()
	p.speakErr = errors.New("quota")
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	r.send(frames.NewTTSSpeakFrame("Hi."))
	fs := r.up.waitFor(t, "speak error", atLeast[*frames.ErrorFrame](1))
	if e := ofType[*frames.ErrorFrame](fs)[0]; !errors.Is(e.Error, p.speakErr) {
		t.Fatalf("error = %v", e.Error)
	}
	boom := errors.New("socket gone")
	p.getSink(t).Error(boom, true)
	fs = r.up.waitFor(t, "sink error", atLeast[*frames.ErrorFrame](2))
	if e := ofType[*frames.ErrorFrame](fs)[1]; !errors.Is(e.Error, boom) || !e.Fatal {
		t.Fatalf("error = %v fatal=%v", e.Error, e.Fatal)
	}
}

func TestSettingsUpdate(t *testing.T) {
	p := newFakeTTS()
	b := NewStreaming("tts", p, Options{Settings: frames.TTSSettings{Voice: "a", Model: "m"}})
	r := newRig(t, b, nil)
	r.send(frames.NewTTSUpdateSettingsFrame(frames.TTSSettings{Voice: "b"}))
	r.dataBarrier(t)
	if s := b.Settings(); s.Voice != "b" || s.Model != "m" {
		t.Fatalf("settings = %+v", s)
	}
	p.mu.Lock()
	applied := append([]frames.TTSSettings(nil), p.applied...)
	p.mu.Unlock()
	if len(applied) != 1 || applied[0].Voice != "b" {
		t.Fatalf("applied = %+v", applied)
	}
	if n := len(ofType[*frames.TTSUpdateSettingsFrame](r.down.snapshot())); n != 0 {
		t.Fatal("consumed settings frame forwarded")
	}
}

func TestEndFrameWaitsForInFlightAudio(t *testing.T) {
	p := newFakeTTS()
	p.hold = true
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	s := p.getSink(t)
	r.send(frames.NewTTSSpeakFrame("Goodbye."))
	r.send(frames.NewEndFrame())
	id := p.waitFlush(t)
	time.Sleep(20 * time.Millisecond)
	if n := len(ofType[*frames.EndFrame](r.down.snapshot())); n != 0 {
		t.Fatal("EndFrame forwarded before audio finished")
	}
	p.mu.Lock()
	early := len(p.disconnects)
	p.mu.Unlock()
	if early != 0 {
		t.Fatal("provider disconnected before audio finished")
	}
	s.Audio(id, make([]byte, 320))
	s.Done(id)
	fs := r.down.waitFor(t, "end", atLeast[*frames.EndFrame](1))
	got := relevant(fs)
	if indexOf(got, "end") < indexOf(got, "stopped") || indexOf(got, "audio") < 0 {
		t.Fatalf("order = %v", got)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.disconnects) != 1 || !p.disconnects[0] {
		t.Fatalf("disconnects = %v", p.disconnects)
	}
}

func TestCancelFrameDisconnectsAndForwards(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{}), nil)
	p.getSink(t)
	r.send(frames.NewCancelFrame())
	r.down.waitFor(t, "cancel", atLeast[*frames.CancelFrame](1))
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.disconnects) != 1 || p.disconnects[0] {
		t.Fatalf("disconnects = %v", p.disconnects)
	}
}

func TestStopTimeoutClosesContextWithoutDone(t *testing.T) {
	p := newFakeTTS()
	p.hold = true
	r := newRig(t, NewStreaming("tts", p, Options{StopTimeout: 40 * time.Millisecond}), nil)
	s := p.getSink(t)
	r.send(frames.NewTTSSpeakFrame("Hi."))
	id := p.waitFlush(t)
	s.Audio(id, make([]byte, 320))
	began := time.Now()
	r.down.waitFor(t, "stopped after timeout", atLeast[*frames.TTSStoppedFrame](1))
	if d := time.Since(began); d < 30*time.Millisecond {
		t.Fatalf("stopped too early: %v", d)
	}
}

func TestMarkdownFilterAppliedBeforeProvider(t *testing.T) {
	p := newFakeTTS()
	r := newRig(t, NewStreaming("tts", p, Options{Filters: []TextFilter{MarkdownFilter{}}}), nil)
	r.send(frames.NewTTSSpeakFrame("**Bold** [link](http://x) move."))
	r.down.waitFor(t, "stopped", atLeast[*frames.TTSStoppedFrame](1))
	if calls := p.speakCalls(); len(calls) != 1 || calls[0].text != "Bold link move." {
		t.Fatalf("speaks = %+v", calls)
	}
}

type httpProvider struct {
	url string
	mu  sync.Mutex
	req []Request
}

func (h *httpProvider) Synthesize(ctx context.Context, req Request) (io.ReadCloser, error) {
	h.mu.Lock()
	h.req = append(h.req, req)
	h.mu.Unlock()
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, h.url, strings.NewReader(req.Text))
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, err
	}
	return resp.Body, nil
}

type chunkServer struct {
	*httptest.Server
	release chan struct{}
	aborted chan struct{}
}

func newChunkServer(t *testing.T, chunk int) *chunkServer {
	cs := &chunkServer{release: make(chan struct{}), aborted: make(chan struct{}, 1)}
	cs.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		_, _ = w.Write(bytes.Repeat([]byte{1}, chunk))
		w.(http.Flusher).Flush()
		select {
		case <-cs.release:
			_, _ = w.Write(bytes.Repeat([]byte{2}, chunk))
		case <-r.Context().Done():
			cs.aborted <- struct{}{}
		}
	}))
	t.Cleanup(cs.Close)
	return cs
}

func TestHTTPStreamsChunksAsTheyArrive(t *testing.T) {
	cs := newChunkServer(t, 320)
	hp := &httpProvider{url: cs.URL}
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	r := newRig(t, NewHTTP("http-tts", hp, Options{SampleRate: 8000, ChunkDuration: 20 * time.Millisecond}), start)
	r.send(frames.NewTTSSpeakFrame("Streaming please."))
	fs := r.down.waitFor(t, "first chunk before body ends", atLeast[*frames.TTSAudioFrame](1))
	if a := ofType[*frames.TTSAudioFrame](fs)[0]; len(a.Data) != 320 || a.Data[0] != 1 || a.SampleRate != 8000 {
		t.Fatalf("first chunk len=%d rate=%d", len(a.Data), a.SampleRate)
	}
	close(cs.release)
	fs = r.down.waitFor(t, "stopped", atLeast[*frames.TTSStoppedFrame](1))
	if got := strings.Join(relevant(fs), "|"); got != "started|audio|audio|text:Streaming please.|stopped" {
		t.Fatalf("frames = %s", got)
	}
	if n := len(metrics[frames.ProcessingMetricsData](fs)); n != 1 {
		t.Fatalf("processing metrics = %d", n)
	}
	hp.mu.Lock()
	defer hp.mu.Unlock()
	if len(hp.req) != 1 || hp.req[0].Text != "Streaming please." || hp.req[0].SampleRate != 8000 || hp.req[0].ContextID == "" {
		t.Fatalf("requests = %+v", hp.req)
	}
}

func TestHTTPInterruptionAbortsRequest(t *testing.T) {
	cs := newChunkServer(t, 320)
	hp := &httpProvider{url: cs.URL}
	r := newRig(t, NewHTTP("http-tts", hp, Options{SampleRate: 8000, ChunkDuration: 20 * time.Millisecond}), nil)
	r.send(frames.NewTTSSpeakFrame("Long story."))
	r.down.waitFor(t, "first chunk", atLeast[*frames.TTSAudioFrame](1))
	r.send(frames.NewInterruptionFrame())
	select {
	case <-cs.aborted:
	case <-time.After(guard):
		t.Fatal("HTTP request not aborted on interruption")
	}
	r.up.waitFor(t, "stopped upstream", atLeast[*frames.TTSStoppedFrame](1))
	r.systemBarrier(t)
	if n := len(ofType[*frames.ErrorFrame](r.up.snapshot())); n != 0 {
		t.Fatalf("aborted request reported %d errors", n)
	}
	cut := 0
	fs := r.down.snapshot()
	for i, f := range fs {
		if _, ok := f.(*frames.InterruptionFrame); ok {
			cut = i
		}
	}
	for _, f := range fs[cut:] {
		if text, ok := f.(*frames.TTSTextFrame); ok && text.Text == "Long story." {
			t.Fatal("stale text emitted after interruption")
		}
	}
}

func TestHTTPErrorBecomesErrorFrame(t *testing.T) {
	r := newRig(t, NewHTTP("http-tts", errProvider{}, Options{}), nil)
	r.send(frames.NewTTSSpeakFrame("Hi."))
	fs := r.up.waitFor(t, "error", atLeast[*frames.ErrorFrame](1))
	if e := ofType[*frames.ErrorFrame](fs)[0]; !strings.Contains(e.Error.Error(), "503") {
		t.Fatalf("error = %v", e.Error)
	}
}

type errProvider struct{}

func (errProvider) Synthesize(context.Context, Request) (io.ReadCloser, error) {
	return nil, errors.New("status 503")
}

func TestSkipWAVHeader(t *testing.T) {
	var w bytes.Buffer
	w.WriteString("RIFF")
	_ = binary.Write(&w, binary.LittleEndian, uint32(0))
	w.WriteString("WAVEfmt ")
	_ = binary.Write(&w, binary.LittleEndian, uint32(16))
	w.Write(make([]byte, 16))
	w.WriteString("LIST")
	_ = binary.Write(&w, binary.LittleEndian, uint32(3))
	w.Write([]byte{9, 9, 9, 0})
	w.WriteString("data")
	_ = binary.Write(&w, binary.LittleEndian, uint32(4))
	w.Write([]byte{1, 2, 3, 4})
	got, err := io.ReadAll(SkipWAVHeader(&w))
	if err != nil || !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Fatalf("got % x err %v", got, err)
	}
	raw := []byte("not a wav file at all")
	got, _ = io.ReadAll(SkipWAVHeader(bytes.NewReader(raw)))
	if !bytes.Equal(got, raw) {
		t.Fatalf("raw passthrough = %q", got)
	}
}
