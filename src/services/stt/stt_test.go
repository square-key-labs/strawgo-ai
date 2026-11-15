package stt

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type fakeStream struct {
	mu          sync.Mutex
	sink        Sink
	audio       []byte
	connects    int
	disconnects []bool
	finalizes   int
	applied     []frames.STTSettings
	connectErr  error
	applyErr    error
	connected   chan struct{}
}

func newFakeStream() *fakeStream {
	return &fakeStream{connected: make(chan struct{})}
}

func (f *fakeStream) Connect(_ context.Context, sink Sink) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.connects++
	f.sink = sink
	if f.connects == 1 {
		close(f.connected)
	}
	return f.connectErr
}

func (f *fakeStream) SendAudio(audio []byte) {
	f.mu.Lock()
	f.audio = append(f.audio, audio...)
	f.mu.Unlock()
}

func (f *fakeStream) Disconnect(graceful bool) {
	f.mu.Lock()
	f.disconnects = append(f.disconnects, graceful)
	f.mu.Unlock()
}

func (f *fakeStream) Finalize() {
	f.mu.Lock()
	f.finalizes++
	f.mu.Unlock()
}

func (f *fakeStream) ApplySettings(s frames.STTSettings) error {
	f.mu.Lock()
	f.applied = append(f.applied, s)
	f.mu.Unlock()
	return f.applyErr
}

func (f *fakeStream) audioLen() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.audio)
}

func (f *fakeStream) getSink(t *testing.T) Sink {
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

func TestStreamingForwardsAudioToProviderAndDownstream(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	r.send(frames.NewAudioFrame(pcm(320, 1), 16000, 1))
	r.send(frames.NewAudioFrame(pcm(320, 2), 16000, 1))
	r.down.waitFor(t, "audio", atLeast[*frames.AudioFrame](2))
	p.mu.Lock()
	defer p.mu.Unlock()
	if !bytes.Equal(p.audio, append(pcm(320, 1), pcm(320, 2)...)) {
		t.Fatalf("provider audio length %d, content mismatch", len(p.audio))
	}
	if p.connects != 1 {
		t.Fatalf("connects = %d", p.connects)
	}
}

func TestStreamingDropAudioStopsForwarding(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{DropAudio: true}), nil)
	r.send(frames.NewAudioFrame(pcm(320, 1), 16000, 1))
	r.dataBarrier(t)
	if n := len(ofType[*frames.AudioFrame](r.down.snapshot())); n != 0 {
		t.Fatalf("audio forwarded with DropAudio: %d", n)
	}
	if p.audioLen() != 320 {
		t.Fatalf("provider got %d bytes", p.audioLen())
	}
}

func TestStreamingSampleRateFromStartFrame(t *testing.T) {
	p := newFakeStream()
	start := frames.NewStartFrame()
	start.AudioInSampleRate = 8000
	b := NewStreaming("stt", p, Options{})
	newRig(t, b, start)
	p.getSink(t)
	if b.SampleRate() != 8000 {
		t.Fatalf("sample rate = %d", b.SampleRate())
	}
}

func TestStreamingTranscriptsBecomeFrames(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	s := p.getSink(t)
	s.Transcript(Transcript{Text: ""})
	s.Transcript(Transcript{Text: "hel"})
	s.Transcript(Transcript{Text: "hello there", Final: true, Language: "en-US"})
	fs := r.down.waitFor(t, "transcripts", atLeast[*frames.TranscriptionFrame](1))
	interims := ofType[*frames.InterimTranscriptionFrame](fs)
	if len(interims) != 1 || interims[0].Text != "hel" {
		t.Fatalf("interim frames = %+v", interims)
	}
	ts := ofType[*frames.TranscriptionFrame](fs)
	if len(ts) != 1 {
		t.Fatalf("got %d transcription frames", len(ts))
	}
	if ts[0].Text != "hello there" || !ts[0].IsFinal || ts[0].Language != "en-US" || !ts[0].SurvivesInterruption() {
		t.Fatalf("final = %q final=%v lang=%q", ts[0].Text, ts[0].IsFinal, ts[0].Language)
	}
}

func TestSinkPushForwardsArbitraryFrames(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	s := p.getSink(t)
	s.Push(frames.NewUserStartedSpeakingFrame(), frames.Upstream)
	r.up.waitFor(t, "pushed frame", atLeast[*frames.UserStartedSpeakingFrame](1))
}

func TestMuteStopsAudioToProvider(t *testing.T) {
	p := newFakeStream()
	b := NewStreaming("stt", p, Options{})
	r := newRig(t, b, nil)
	r.send(frames.NewAudioFrame(pcm(100, 1), 16000, 1))
	r.dataBarrier(t)

	r.send(frames.NewSTTMuteFrame(true))
	r.systemBarrier(t)
	if !b.Muted() {
		t.Fatal("not muted")
	}
	r.send(frames.NewAudioFrame(pcm(100, 2), 16000, 1))
	r.dataBarrier(t)
	if p.audioLen() != 100 {
		t.Fatalf("muted audio reached provider: %d bytes", p.audioLen())
	}
	if n := len(ofType[*frames.AudioFrame](r.down.snapshot())); n != 2 {
		t.Fatalf("audio frames downstream = %d, want 2", n)
	}
	if n := len(ofType[*frames.STTMuteFrame](r.down.snapshot())); n != 0 {
		t.Fatal("mute frame forwarded")
	}

	r.send(frames.NewSTTMuteFrame(false))
	r.systemBarrier(t)
	r.send(frames.NewAudioFrame(pcm(100, 3), 16000, 1))
	r.dataBarrier(t)
	if p.audioLen() != 200 {
		t.Fatalf("unmuted audio not delivered: %d bytes", p.audioLen())
	}
}

type named string

func (n named) Name() string { return string(n) }

func TestSettingsUpdate(t *testing.T) {
	p := newFakeStream()
	b := NewStreaming("stt", p, Options{Settings: frames.STTSettings{Model: "nova", Language: "en"}})
	r := newRig(t, b, nil)

	r.send(frames.NewSTTUpdateSettingsFrame(frames.STTSettings{Language: "de"}))
	r.dataBarrier(t)
	if got := b.Settings(); got.Model != "nova" || got.Language != "de" {
		t.Fatalf("settings = %+v", got)
	}
	p.mu.Lock()
	if len(p.applied) != 1 || p.applied[0].Language != "de" || p.applied[0].Model != "nova" {
		t.Fatalf("applied = %+v", p.applied)
	}
	p.mu.Unlock()
	if n := len(ofType[*frames.STTUpdateSettingsFrame](r.down.snapshot())); n != 0 {
		t.Fatal("consumed settings frame forwarded")
	}

	other := frames.NewSTTUpdateSettingsFrame(frames.STTSettings{Language: "fr"})
	other.Service = named("someone-else")
	r.send(other)
	r.down.waitFor(t, "foreign settings forwarded", atLeast[*frames.STTUpdateSettingsFrame](1))
	if b.Settings().Language != "de" {
		t.Fatal("foreign settings applied")
	}

	b.SetModel("nova-3")
	if b.Settings().Model != "nova-3" {
		t.Fatal("SetModel ignored")
	}
}

func TestSettingsApplyErrorPushesError(t *testing.T) {
	p := newFakeStream()
	p.applyErr = errors.New("unsupported")
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	r.send(frames.NewSTTUpdateSettingsFrame(frames.STTSettings{Model: "x"}))
	fs := r.up.waitFor(t, "error", atLeast[*frames.ErrorFrame](1))
	if e := ofType[*frames.ErrorFrame](fs)[0]; e.Fatal || !errors.Is(e.Error, p.applyErr) {
		t.Fatalf("error frame = %+v", e)
	}
}

func TestProviderErrorBecomesErrorFrame(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	s := p.getSink(t)
	boom := errors.New("socket died")
	s.Error(boom, true)
	fs := r.up.waitFor(t, "error", atLeast[*frames.ErrorFrame](1))
	e := ofType[*frames.ErrorFrame](fs)[0]
	if !e.Fatal || !errors.Is(e.Error, boom) || e.Processor != "stt" {
		t.Fatalf("error frame = fatal:%v err:%v proc:%q", e.Fatal, e.Error, e.Processor)
	}
}

func TestConnectErrorBecomesErrorFrame(t *testing.T) {
	p := newFakeStream()
	p.connectErr = errors.New("auth failed")
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	fs := r.up.waitFor(t, "error", atLeast[*frames.ErrorFrame](1))
	if e := ofType[*frames.ErrorFrame](fs)[0]; !errors.Is(e.Error, p.connectErr) {
		t.Fatalf("error = %v", e.Error)
	}
}

func TestEndFrameForwardedAndProviderDisconnectedGracefully(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	p.getSink(t)
	r.send(frames.NewEndFrame())
	r.down.waitFor(t, "EndFrame", atLeast[*frames.EndFrame](1))
	r.send(frames.NewAudioFrame(pcm(100, 1), 16000, 1))
	r.dataBarrier(t)
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.disconnects) != 1 || !p.disconnects[0] {
		t.Fatalf("disconnects = %v", p.disconnects)
	}
	if len(p.audio) != 0 {
		t.Fatal("audio after EndFrame reached provider")
	}
}

func TestCancelFrameForwardedAndProviderDisconnected(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	p.getSink(t)
	r.send(frames.NewCancelFrame())
	r.down.waitFor(t, "CancelFrame", atLeast[*frames.CancelFrame](1))
	r.send(frames.NewCancelFrame())
	r.down.waitFor(t, "second CancelFrame", atLeast[*frames.CancelFrame](2))
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.disconnects) != 1 || p.disconnects[0] {
		t.Fatalf("disconnects = %v", p.disconnects)
	}
}

func TestMetadataFrameOnStart(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{Provider: "fake", TTFSP99: 700 * time.Millisecond}), nil)
	fs := r.down.waitFor(t, "metadata", atLeast[*frames.STTMetadataFrame](1))
	if ofType[*frames.STTMetadataFrame](fs)[0] == nil {
		t.Fatal("nil metadata frame")
	}
}

func speechTurn(t *testing.T, r *rig) {
	t.Helper()
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	stop := frames.NewVADUserStoppedSpeakingFrame(0)
	stop.Timestamp = time.Now()
	r.send(stop)
	r.systemBarrier(t)
}

func TestFinalizeCalledWhenUserStops(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	speechTurn(t, r)
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.finalizes != 1 {
		t.Fatalf("finalizes = %d", p.finalizes)
	}
}

func TestTTFBMetricOnFinalizedTranscript(t *testing.T) {
	p := newFakeStream()
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	r := newRig(t, NewStreaming("stt", p, Options{Settings: frames.STTSettings{Model: "m1"}}), start)
	s := p.getSink(t)
	speechTurn(t, r)
	time.Sleep(5 * time.Millisecond)
	s.Transcript(Transcript{Text: "hi", Final: true, Finalized: true})
	fs := r.down.waitFor(t, "ttfb", func(fs []frames.Frame) bool { return len(metrics[frames.TTFBMetricsData](fs)) > 0 })
	m := metrics[frames.TTFBMetricsData](fs)[0]
	if m.Value < 5*time.Millisecond || m.Value > guard || m.Processor != "stt" || m.Model != "m1" {
		t.Fatalf("ttfb = %+v", m)
	}
	s.Transcript(Transcript{Text: "again", Final: true, Finalized: true})
	r.down.waitFor(t, "second transcript", atLeast[*frames.TranscriptionFrame](2))
	if n := len(metrics[frames.TTFBMetricsData](r.down.snapshot())); n != 1 {
		t.Fatalf("ttfb reported %d times for one turn", n)
	}
}

func TestTTFBFallsBackToTimeoutForUnfinalizedFinal(t *testing.T) {
	p := newFakeStream()
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	r := newRig(t, NewStreaming("stt", p, Options{TTFBTimeout: 30 * time.Millisecond}), start)
	s := p.getSink(t)
	speechTurn(t, r)
	s.Transcript(Transcript{Text: "hi", Final: true})
	r.down.waitFor(t, "ttfb after timeout", func(fs []frames.Frame) bool { return len(metrics[frames.TTFBMetricsData](fs)) == 1 })
}

func TestNoMetricsWhenDisabled(t *testing.T) {
	p := newFakeStream()
	r := newRig(t, NewStreaming("stt", p, Options{}), nil)
	s := p.getSink(t)
	r.send(frames.NewAudioFrame(pcm(3200, 1), 16000, 1))
	speechTurn(t, r)
	s.Transcript(Transcript{Text: "hi", Final: true, Finalized: true})
	r.down.waitFor(t, "transcript", atLeast[*frames.TranscriptionFrame](1))
	if n := len(metrics[frames.TTFBMetricsData](r.down.snapshot())) + len(metrics[frames.STTUsageMetricsData](r.down.snapshot())); n != 0 {
		t.Fatalf("metrics emitted while disabled: %d", n)
	}
}

func TestUsageMetricsCountAudioSeconds(t *testing.T) {
	p := newFakeStream()
	start := frames.NewStartFrame()
	start.EnableUsageMetrics = true
	r := newRig(t, NewStreaming("stt", p, Options{SampleRate: 16000}), start)
	s := p.getSink(t)
	for i := 0; i < 10; i++ {
		r.send(frames.NewAudioFrame(pcm(3200, 1), 16000, 1))
	}
	r.dataBarrier(t)
	s.Transcript(Transcript{Text: "x", Final: true})
	fs := r.down.waitFor(t, "usage", func(fs []frames.Frame) bool { return len(metrics[frames.STTUsageMetricsData](fs)) > 0 })
	if got := metrics[frames.STTUsageMetricsData](fs)[0].Value.AudioSeconds; got < 0.999 || got > 1.001 {
		t.Fatalf("audio seconds = %v, want 1", got)
	}
}

type fakeSegmented struct {
	mu    sync.Mutex
	wavs  [][]byte
	text  []string
	err   error
	delay time.Duration
}

func (f *fakeSegmented) Transcribe(ctx context.Context, wav []byte) (Transcript, error) {
	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return Transcript{}, ctx.Err()
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.wavs = append(f.wavs, wav)
	if f.err != nil {
		return Transcript{}, f.err
	}
	n := len(f.wavs)
	text := "segment"
	if n <= len(f.text) {
		text = f.text[n-1]
	}
	return Transcript{Text: text, Language: "en"}, nil
}

func (f *fakeSegmented) all() [][]byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([][]byte(nil), f.wavs...)
}

func sendAudio(t *testing.T, r *rig, chunks int, v byte) {
	t.Helper()
	before := len(ofType[*frames.AudioFrame](r.down.snapshot()))
	for i := 0; i < chunks; i++ {
		r.send(frames.NewAudioFrame(pcm(320, v), 16000, 1))
	}
	r.down.waitFor(t, "audio forwarded", atLeast[*frames.AudioFrame](before+chunks))
}

func checkWAV(t *testing.T, wav []byte, rate int) []byte {
	t.Helper()
	if len(wav) < 44 || string(wav[0:4]) != "RIFF" || string(wav[8:16]) != "WAVEfmt " || string(wav[36:40]) != "data" {
		t.Fatalf("bad wav header % x", wav[:min(len(wav), 44)])
	}
	if got := binary.LittleEndian.Uint32(wav[24:]); int(got) != rate {
		t.Fatalf("wav rate = %d", got)
	}
	if ch := binary.LittleEndian.Uint16(wav[22:]); ch != 1 {
		t.Fatalf("channels = %d", ch)
	}
	n := binary.LittleEndian.Uint32(wav[40:])
	if int(n) != len(wav)-44 || binary.LittleEndian.Uint32(wav[4:]) != n+36 {
		t.Fatalf("wav sizes inconsistent: data=%d len=%d", n, len(wav))
	}
	return wav[44:]
}

func TestSegmentedSendsOneWAVPerVADSegment(t *testing.T) {
	p := &fakeSegmented{text: []string{"first", "second"}}
	start := frames.NewStartFrame()
	start.EnableMetrics = true
	r := newRig(t, NewSegmented("whisper", p, Options{SampleRate: 16000, PreRoll: 20 * time.Millisecond, TrailingSilence: -1}), start)

	sendAudio(t, r, 5, 9)
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	r.systemBarrier(t)
	sendAudio(t, r, 3, 1)
	r.send(frames.NewVADUserStoppedSpeakingFrame(0))
	r.systemBarrier(t)

	sendAudio(t, r, 2, 7)
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	r.systemBarrier(t)
	sendAudio(t, r, 4, 2)
	r.send(frames.NewVADUserStoppedSpeakingFrame(0))

	fs := r.down.waitFor(t, "two transcripts", atLeast[*frames.TranscriptionFrame](2))
	ts := ofType[*frames.TranscriptionFrame](fs)
	if ts[0].Text != "first" || ts[1].Text != "second" || !ts[0].IsFinal || !ts[1].IsFinal || ts[0].Language != "en" {
		t.Fatalf("transcripts = %q %q", ts[0].Text, ts[1].Text)
	}
	wavs := p.all()
	if len(wavs) != 2 {
		t.Fatalf("wavs = %d", len(wavs))
	}
	preroll := 640
	first := checkWAV(t, wavs[0], 16000)
	if want := append(pcm(preroll, 9), pcm(3*320, 1)...); !bytes.Equal(first, want) {
		t.Fatalf("segment 1: %d bytes, want %d", len(first), len(want))
	}
	second := checkWAV(t, wavs[1], 16000)
	if want := append(pcm(preroll, 7), pcm(4*320, 2)...); !bytes.Equal(second, want) {
		t.Fatalf("segment 2: %d bytes, want %d", len(second), len(want))
	}
	r.down.waitFor(t, "processing metrics", func(fs []frames.Frame) bool {
		return len(metrics[frames.ProcessingMetricsData](fs)) >= 2
	})
}

func TestSegmentedAppendsTrailingSilence(t *testing.T) {
	p := &fakeSegmented{}
	r := newRig(t, NewSegmented("whisper", p, Options{SampleRate: 16000, PreRoll: time.Millisecond, TrailingSilence: 10 * time.Millisecond}), nil)
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	r.systemBarrier(t)
	sendAudio(t, r, 1, 5)
	r.send(frames.NewVADUserStoppedSpeakingFrame(0))
	r.down.waitFor(t, "transcript", atLeast[*frames.TranscriptionFrame](1))
	data := checkWAV(t, p.all()[0], 16000)
	if want := append(pcm(320, 5), make([]byte, 320)...); !bytes.Equal(data, want) {
		t.Fatalf("got %d bytes, want %d", len(data), len(want))
	}
}

func TestSegmentedNoSpeechNoRequest(t *testing.T) {
	p := &fakeSegmented{}
	r := newRig(t, NewSegmented("whisper", p, Options{SampleRate: 16000}), nil)
	sendAudio(t, r, 10, 1)
	r.send(frames.NewVADUserStoppedSpeakingFrame(0))
	r.systemBarrier(t)
	r.send(frames.NewEndFrame())
	r.down.waitFor(t, "EndFrame", atLeast[*frames.EndFrame](1))
	if n := len(p.all()); n != 0 {
		t.Fatalf("transcribed %d segments without speech", n)
	}
}

func TestSegmentedEndFrameFlushesActiveSegment(t *testing.T) {
	p := &fakeSegmented{}
	r := newRig(t, NewSegmented("whisper", p, Options{SampleRate: 16000, TrailingSilence: -1}), nil)
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	r.systemBarrier(t)
	sendAudio(t, r, 2, 3)
	r.send(frames.NewEndFrame())
	r.down.waitFor(t, "EndFrame", atLeast[*frames.EndFrame](1))
	r.down.waitFor(t, "transcript", atLeast[*frames.TranscriptionFrame](1))
	if n := len(p.all()); n != 1 {
		t.Fatalf("segments = %d", n)
	}
}

func TestSegmentedErrorBecomesErrorFrame(t *testing.T) {
	p := &fakeSegmented{err: errors.New("http 500")}
	r := newRig(t, NewSegmented("whisper", p, Options{SampleRate: 16000}), nil)
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	r.systemBarrier(t)
	sendAudio(t, r, 1, 3)
	r.send(frames.NewVADUserStoppedSpeakingFrame(0))
	fs := r.up.waitFor(t, "error", atLeast[*frames.ErrorFrame](1))
	if e := ofType[*frames.ErrorFrame](fs)[0]; !errors.Is(e.Error, p.err) || e.Fatal {
		t.Fatalf("error = %+v", e)
	}
}

func TestSegmentedCancelAbortsInFlight(t *testing.T) {
	p := &fakeSegmented{delay: time.Minute}
	r := newRig(t, NewSegmented("whisper", p, Options{SampleRate: 16000}), nil)
	r.send(frames.NewVADUserStartedSpeakingFrame(0.2))
	r.systemBarrier(t)
	sendAudio(t, r, 1, 3)
	r.send(frames.NewVADUserStoppedSpeakingFrame(0))
	r.systemBarrier(t)
	began := time.Now()
	r.send(frames.NewCancelFrame())
	r.down.waitFor(t, "CancelFrame", atLeast[*frames.CancelFrame](1))
	if d := time.Since(began); d > time.Second {
		t.Fatalf("cancel took %v", d)
	}
	if n := len(ofType[*frames.ErrorFrame](r.up.snapshot())); n != 0 {
		t.Fatal("cancelled request reported as error")
	}
}

func TestWAVHelpers(t *testing.T) {
	data := []byte{1, 2, 3, 4}
	w := WAV(data, 24000, 2)
	if !bytes.Equal(w[:44], WAVHeader(4, 24000, 2)) {
		t.Fatal("WAV header differs from WAVHeader")
	}
	if binary.LittleEndian.Uint32(w[28:]) != 24000*2*2 || binary.LittleEndian.Uint16(w[32:]) != 4 || binary.LittleEndian.Uint16(w[34:]) != 16 {
		t.Fatal("byte rate / block align / bits wrong")
	}
	if !bytes.Equal(w[44:], data) {
		t.Fatal("payload mismatch")
	}
}
