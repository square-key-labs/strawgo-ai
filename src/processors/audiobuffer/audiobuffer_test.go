package audiobuffer

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/wav"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type trackCall struct {
	user, bot []byte
	rate, ch  int
}

type mergedCall struct {
	audio    []byte
	rate, ch int
}

type recorder struct {
	mu      sync.Mutex
	tracks  []trackCall
	merged  []mergedCall
	users   []TurnAudio
	bots    []TurnAudio
	started int
	stopped int
}

func (r *recorder) handlers() Handlers {
	return Handlers{
		OnAudioData: func(a []byte, rate, ch int) {
			r.mu.Lock()
			r.merged = append(r.merged, mergedCall{a, rate, ch})
			r.mu.Unlock()
		},
		OnTrackAudioData: func(u, b []byte, rate, ch int) {
			r.mu.Lock()
			r.tracks = append(r.tracks, trackCall{u, b, rate, ch})
			r.mu.Unlock()
		},
		OnUserTurnAudio:    func(t TurnAudio) { r.mu.Lock(); r.users = append(r.users, t); r.mu.Unlock() },
		OnBotTurnAudio:     func(t TurnAudio) { r.mu.Lock(); r.bots = append(r.bots, t); r.mu.Unlock() },
		OnRecordingStarted: func() { r.mu.Lock(); r.started++; r.mu.Unlock() },
		OnRecordingStopped: func() { r.mu.Lock(); r.stopped++; r.mu.Unlock() },
	}
}

func constPCM(samples int, v int16) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out
}

func sine(samples, rate int, freq float64) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(8000*math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))))
	}
	return out
}

func send(t *testing.T, p *Processor, f frames.Frame) {
	t.Helper()
	if err := p.HandleFrame(context.Background(), f, frames.Downstream); err != nil {
		t.Fatal(err)
	}
}

func startFrame(rate int) *frames.StartFrame {
	f := frames.NewStartFrame()
	f.AudioInSampleRate = rate
	f.AudioOutSampleRate = rate
	return f
}

func userAudio(data []byte, rate int) *frames.AudioFrame {
	return frames.NewAudioFrame(data, rate, 1)
}

func botAudio(data []byte, rate int) *frames.TTSAudioFrame {
	return frames.NewTTSAudioFrame(data, rate, 1)
}

func TestTracksAlignedWithSilenceAndFlushedAtBufferSize(t *testing.T) {
	r := &recorder{}
	p := New(Config{BufferSize: 4800, AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(16000))
	user := constPCM(800, 1000)
	bot := constPCM(800, -2000)
	send(t, p, userAudio(user, 16000))
	send(t, p, userAudio(user, 16000))
	send(t, p, botAudio(bot, 16000))
	if len(r.tracks) != 0 {
		t.Fatal("flushed before buffer size")
	}
	send(t, p, botAudio(bot, 16000))
	if len(r.tracks) != 1 || len(r.merged) != 1 {
		t.Fatalf("tracks=%d merged=%d, want 1/1", len(r.tracks), len(r.merged))
	}
	tc := r.tracks[0]
	wantUser := append(append(append([]byte{}, user...), user...), make([]byte, 1600)...)
	wantBot := append(append(make([]byte, 1600), bot...), bot...)
	if !bytes.Equal(tc.user, wantUser) || !bytes.Equal(tc.bot, wantBot) {
		t.Fatalf("tracks not aligned: user %d bytes, bot %d bytes", len(tc.user), len(tc.bot))
	}
	if tc.rate != 16000 || tc.ch != 1 {
		t.Fatalf("rate/ch = %d/%d", tc.rate, tc.ch)
	}
	wantMix := append(append(append([]byte{}, user...), constPCM(800, -1000)...), bot...)
	if !bytes.Equal(r.merged[0].audio, wantMix) {
		t.Fatal("mono mix is not the sample-wise sum of the tracks")
	}
	if p.HasAudio() {
		t.Fatal("buffers not cleared after flush")
	}
}

func TestStereoInterleavesUserLeftBotRight(t *testing.T) {
	r := &recorder{}
	p := New(Config{NumChannels: 2, BufferSize: 2400, AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(8000))
	send(t, p, userAudio(constPCM(400, 111), 8000))
	send(t, p, userAudio(constPCM(400, 111), 8000))
	send(t, p, botAudio(constPCM(400, 222), 8000))
	send(t, p, botAudio(constPCM(400, 222), 8000))
	if len(r.merged) != 1 {
		t.Fatalf("merged = %d", len(r.merged))
	}
	m := r.merged[0]
	if m.ch != 2 || len(m.audio) != 4*1200 {
		t.Fatalf("stereo ch=%d len=%d", m.ch, len(m.audio))
	}
	l := int16(binary.LittleEndian.Uint16(m.audio[0:]))
	rr := int16(binary.LittleEndian.Uint16(m.audio[2:]))
	if l != 111 || rr != 0 {
		t.Fatalf("first frame L=%d R=%d, want 111/0", l, rr)
	}
	end := len(m.audio) - 4
	l = int16(binary.LittleEndian.Uint16(m.audio[end:]))
	rr = int16(binary.LittleEndian.Uint16(m.audio[end+2:]))
	if l != 0 || rr != 222 {
		t.Fatalf("last frame L=%d R=%d, want 0/222", l, rr)
	}
	if p.NumChannels() != 2 {
		t.Fatal("NumChannels != 2")
	}
}

func TestStopFlushesRemainderAndFiresCallbacks(t *testing.T) {
	r := &recorder{}
	p := New(Config{}, r.handlers())
	send(t, p, startFrame(16000))
	send(t, p, userAudio(constPCM(160, 5), 16000))
	if p.Recording() || p.HasAudio() || r.started != 0 {
		t.Fatal("recorded before start")
	}
	send(t, p, frames.NewAudioBufferStartRecordingFrame())
	if !p.Recording() || r.started != 1 {
		t.Fatal("start frame did not start recording")
	}
	send(t, p, userAudio(constPCM(160, 5), 16000))
	if len(r.tracks) != 0 {
		t.Fatal("flushed early")
	}
	send(t, p, frames.NewEndFrame())
	if p.Recording() || r.stopped != 1 || len(r.tracks) != 1 || len(r.tracks[0].user) != 320 {
		t.Fatalf("stop: recording=%v stopped=%d tracks=%d", p.Recording(), r.stopped, len(r.tracks))
	}
	send(t, p, frames.NewCancelFrame())
	if r.stopped != 1 {
		t.Fatal("stop fired twice")
	}
}

func TestStopRecordingAPI(t *testing.T) {
	r := &recorder{}
	p := New(Config{SampleRate: 8000}, r.handlers())
	p.StartRecording()
	p.StartRecording()
	send(t, p, botAudio(constPCM(80, 9), 8000))
	p.StopRecording()
	if r.started != 1 || r.stopped != 1 || len(r.tracks) != 1 || len(r.tracks[0].bot) != 160 || len(r.tracks[0].user) != 160 {
		t.Fatalf("started=%d stopped=%d tracks=%d", r.started, r.stopped, len(r.tracks))
	}
}

func TestSilenceInsertedForGaps(t *testing.T) {
	r := &recorder{}
	p := New(Config{AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(16000))
	send(t, p, userAudio(constPCM(320, 100), 16000))
	time.Sleep(350 * time.Millisecond)
	send(t, p, userAudio(constPCM(320, 100), 16000))
	send(t, p, frames.NewEndFrame())
	if len(r.tracks) != 1 {
		t.Fatalf("tracks = %d", len(r.tracks))
	}
	u := r.tracks[0].user
	gap := (len(u) - 2*640) / 2
	if gap < int(0.25*16000) || gap > int(1.0*16000) {
		t.Fatalf("inserted silence = %d samples, want ~0.33 s", gap)
	}
	if v := int16(binary.LittleEndian.Uint16(u[640:])); v != 0 {
		t.Fatalf("gap sample = %d, want 0", v)
	}
	if v := int16(binary.LittleEndian.Uint16(u[len(u)-2:])); v != 100 {
		t.Fatalf("last sample = %d, want 100", v)
	}
}

func TestResamplesToRecorderRate(t *testing.T) {
	r := &recorder{}
	p := New(Config{AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(16000))
	for i := 0; i < 10; i++ {
		send(t, p, userAudio(sine(160, 8000, 300), 8000))
	}
	send(t, p, frames.NewEndFrame())
	got := len(r.tracks[0].user) / 2
	if got < 3000 || got > 3300 {
		t.Fatalf("resampled samples = %d, want ~3200", got)
	}
	if r.tracks[0].rate != 16000 {
		t.Fatalf("rate = %d", r.tracks[0].rate)
	}
}

func TestStereoInputDownmixed(t *testing.T) {
	r := &recorder{}
	p := New(Config{AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(8000))
	stereo := make([]byte, 0, 80*4)
	for i := 0; i < 80; i++ {
		stereo = binary.LittleEndian.AppendUint16(stereo, uint16(int16(100)))
		stereo = binary.LittleEndian.AppendUint16(stereo, uint16(int16(300)))
	}
	send(t, p, frames.NewAudioFrame(stereo, 8000, 2))
	send(t, p, frames.NewEndFrame())
	u := r.tracks[0].user
	if len(u) != 160 || int16(binary.LittleEndian.Uint16(u)) != 200 {
		t.Fatalf("downmix len=%d first=%d", len(u), int16(binary.LittleEndian.Uint16(u)))
	}
}

func TestMemoryBoundedByMaxBufferSize(t *testing.T) {
	r := &recorder{}
	p := New(Config{MaxBufferSize: 8000, AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(8000))
	for i := 0; i < 100; i++ {
		send(t, p, userAudio(constPCM(160, 1), 8000))
		p.mu.Lock()
		n := len(p.user.buf) + len(p.bot.buf)
		p.mu.Unlock()
		if n > 2*8000 {
			t.Fatalf("buffered %d bytes, exceeds bound", n)
		}
	}
	if len(r.tracks) < 3 {
		t.Fatalf("flushes = %d, want periodic flushes", len(r.tracks))
	}
	for _, tc := range r.tracks {
		if len(tc.user) > 8000 || len(tc.user) != len(tc.bot) {
			t.Fatalf("flush user=%d bot=%d", len(tc.user), len(tc.bot))
		}
	}
}

func TestTurnAudio(t *testing.T) {
	r := &recorder{}
	p := New(Config{EnableTurnAudio: true, AutoStartRecording: true}, r.handlers())
	send(t, p, startFrame(8000))
	send(t, p, frames.NewUserStartedSpeakingFrame())
	send(t, p, userAudio(constPCM(160, 10), 8000))
	send(t, p, frames.NewUserStoppedSpeakingFrame())
	send(t, p, frames.NewBotStartedSpeakingFrame())
	send(t, p, botAudio(constPCM(240, 20), 8000))
	send(t, p, frames.NewBotStoppedSpeakingFrame())
	send(t, p, frames.NewUserStartedSpeakingFrame())
	if len(r.users) != 1 || len(r.bots) != 1 {
		t.Fatalf("turns user=%d bot=%d", len(r.users), len(r.bots))
	}
	if r.users[0].TurnNumber != 0 || len(r.users[0].Audio) != 320 || r.users[0].SampleRate != 8000 || r.users[0].NumChannels != 1 {
		t.Fatalf("user turn = %+v (len %d)", r.users[0].TurnNumber, len(r.users[0].Audio))
	}
	if r.bots[0].TurnNumber != 0 || len(r.bots[0].Audio) != 480 {
		t.Fatalf("bot turn len = %d", len(r.bots[0].Audio))
	}
	send(t, p, userAudio(constPCM(160, 10), 8000))
	send(t, p, frames.NewUserStoppedSpeakingFrame())
	send(t, p, frames.NewEndFrame())
	if len(r.users) != 2 || r.users[1].TurnNumber != 1 {
		t.Fatalf("final user turn not emitted at stop: %d", len(r.users))
	}
}

func TestTurnAudioCapped(t *testing.T) {
	r := &recorder{}
	p := New(Config{EnableTurnAudio: true, AutoStartRecording: true, MaxTurnAudioSize: 1000}, r.handlers())
	send(t, p, startFrame(8000))
	send(t, p, frames.NewUserStartedSpeakingFrame())
	for i := 0; i < 20; i++ {
		send(t, p, userAudio(constPCM(160, 10), 8000))
	}
	send(t, p, frames.NewUserStoppedSpeakingFrame())
	send(t, p, frames.NewEndFrame())
	if len(r.users) != 1 || len(r.users[0].Audio) > 1000 || len(r.users[0].Audio) == 0 {
		t.Fatalf("capped turn = %d bytes", len(r.users[0].Audio))
	}
}

func TestEncodeWAV(t *testing.T) {
	pcmData := constPCM(100, 7)
	b := EncodeWAV(pcmData, 16000, 1)
	a, err := wav.Decode(bytes.NewReader(b))
	if err != nil || a.SampleRate != 16000 || a.Channels != 1 || !bytes.Equal(a.Data, pcmData) {
		t.Fatalf("wav decode = %+v err=%v", a, err)
	}
}

type sink struct {
	mu  sync.Mutex
	got []frames.Frame
}

func (s *sink) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error { return nil }
func (s *sink) QueueFrame(f frames.Frame, _ frames.FrameDirection) error {
	s.mu.Lock()
	s.got = append(s.got, f)
	s.mu.Unlock()
	return nil
}
func (s *sink) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (s *sink) Link(processors.FrameProcessor)                      {}
func (s *sink) SetPrev(processors.FrameProcessor)                   {}
func (s *sink) Start(context.Context) error                         { return nil }
func (s *sink) Stop() error                                         { return nil }
func (s *sink) Name() string                                        { return "sink" }

func TestFramesForwarded(t *testing.T) {
	out := &sink{}
	p := New(Config{AutoStartRecording: true}, Handlers{})
	p.Link(out)
	in := []frames.Frame{startFrame(8000), userAudio(constPCM(10, 1), 8000), botAudio(constPCM(10, 1), 8000), frames.NewEndFrame()}
	for _, f := range in {
		send(t, p, f)
	}
	if len(out.got) != len(in) {
		t.Fatalf("forwarded %d frames, want %d", len(out.got), len(in))
	}
	for i := range in {
		if out.got[i] != in[i] {
			t.Fatalf("frame %d = %s, want %s", i, out.got[i].Name(), in[i].Name())
		}
	}
}
