package transports

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/audio/g711"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

func TestOutputPacesAudioInFixedChunks(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	queue(t, h.sess.Output(), ttsAudio(200, 8000))

	waitUntil(t, 2*time.Second, "10 audio chunks", func() bool { return len(h.conn.audioWrites()) == 10 })
	writes := h.conn.audioWrites()
	for _, w := range writes {
		if len(w.msg.Data) != 320 {
			t.Fatalf("chunk size = %d, want 320 (20 ms at 8 kHz)", len(w.msg.Data))
		}
	}
	if span := writes[len(writes)-1].at.Sub(writes[0].at); span < 160*time.Millisecond {
		t.Fatalf("200 ms of audio was sent in %v, not paced in real time", span)
	}
}

func TestOutputResamplesToWireRate(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	queue(t, h.sess.Output(), ttsAudio(100, 16000))
	queue(t, h.sess.Output(), frames.NewTTSStoppedFrame())

	waitUntil(t, 2*time.Second, "TTSStoppedFrame downstream", func() bool { return h.down.count("TTSStoppedFrame") == 1 })
	total := 0
	for _, w := range h.conn.audioWrites() {
		total += len(w.msg.Data)
	}
	if total != 1920 {
		t.Fatalf("wire audio = %d bytes, want 1920 (100 ms at 8 kHz plus the flushed filter tail, padded to a 20 ms chunk)", total)
	}
}

func TestOutputInterruptionClearsInstantly(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	out := h.sess.Output()
	queue(t, out, ttsAudio(1000, 8000))
	queue(t, out, ttsAudio(1000, 8000))

	waitUntil(t, time.Second, "bot started speaking", func() bool { return h.up.count("BotStartedSpeakingFrame") == 1 })
	time.Sleep(60 * time.Millisecond)
	queue(t, out, frames.NewInterruptionFrame())

	waitUntil(t, time.Second, "clear message", func() bool { return h.conn.textIndex("clear") >= 0 })
	waitUntil(t, time.Second, "bot stopped speaking", func() bool { return h.up.count("BotStoppedSpeakingFrame") == 1 })
	time.Sleep(100 * time.Millisecond)

	clearAt := h.conn.textIndex("clear")
	for i, w := range h.conn.snapshot() {
		if i > clearAt && w.msg.Type == serializers.BinaryMessage {
			t.Fatalf("audio leaked after clear at write %d", i)
		}
	}
	if n := len(h.conn.audioWrites()); n > 15 {
		t.Fatalf("%d chunks written for ~80 ms of playback", n)
	}

	queue(t, out, ttsAudio(40, 8000))
	waitUntil(t, time.Second, "audio after interruption", func() bool {
		return len(h.conn.snapshot()) > clearAt+2
	})
	if h.up.count("BotStartedSpeakingFrame") != 2 {
		t.Fatalf("expected a new BotStartedSpeakingFrame after the interruption")
	}
}

func TestOutputBotSpeakingFramesBothDirections(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	out := h.sess.Output()
	start := time.Now()
	queue(t, out, ttsAudio(300, 8000))
	queue(t, out, frames.NewTTSStoppedFrame())

	waitUntil(t, 2*time.Second, "bot stopped downstream", func() bool { return h.down.count("BotStoppedSpeakingFrame") == 1 })
	if elapsed := time.Since(start); elapsed < 250*time.Millisecond {
		t.Fatalf("BotStoppedSpeakingFrame after %v, before the audio finished playing", elapsed)
	}
	for _, c := range []*queuedFrameCapture{h.up, h.down} {
		for _, name := range []string{"BotStartedSpeakingFrame", "BotSpeakingFrame", "BotStoppedSpeakingFrame"} {
			if c.count(name) == 0 {
				t.Fatalf("missing %s in %s direction", name, map[bool]string{true: "upstream", false: "downstream"}[c == h.up])
			}
		}
	}
	if h.down.count("TTSAudioFrame") != 1 {
		t.Fatalf("played TTSAudioFrame was not pushed downstream")
	}
}

func TestOutputEndFrameForwardedAfterFlush(t *testing.T) {
	h := newOutputHarness(t, TransportParams{PlaybackAckTimeout: 50 * time.Millisecond})
	out := h.sess.Output()
	queue(t, out, ttsAudio(100, 8000))
	queue(t, out, frames.NewEndFrame())

	waitUntil(t, 2*time.Second, "EndFrame downstream", func() bool { return h.down.count("EndFrame") == 1 })
	if n := len(h.conn.audioWrites()); n != 5 {
		t.Fatalf("EndFrame forwarded after %d of 5 chunks", n)
	}
	if h.conn.textIndex("end") < 0 {
		t.Fatalf("serialized EndFrame not written before close")
	}
	if !h.conn.isClosed() {
		t.Fatalf("connection not closed after EndFrame")
	}
	if h.ser.hangups.Load() != 1 {
		t.Fatalf("hangups = %d, want 1", h.ser.hangups.Load())
	}
}

func TestOutputCancelFrameForwardedImmediately(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	out := h.sess.Output()
	queue(t, out, ttsAudio(2000, 8000))
	waitUntil(t, time.Second, "playback", func() bool { return len(h.conn.audioWrites()) > 0 })

	start := time.Now()
	queue(t, out, frames.NewCancelFrame())
	waitUntil(t, time.Second, "CancelFrame downstream", func() bool { return h.down.count("CancelFrame") == 1 })
	if time.Since(start) > 500*time.Millisecond {
		t.Fatalf("CancelFrame waited for queued audio")
	}
	if !h.conn.isClosed() {
		t.Fatalf("connection not closed after CancelFrame")
	}
}

func samplesOf(pcm []byte) []int {
	out := make([]int, len(pcm)/2)
	for i := range out {
		out[i] = int(int16(binary.LittleEndian.Uint16(pcm[2*i:])))
	}
	return out
}

func assertFadeEnvelope(t *testing.T, tail []int, full, maxSamples int) {
	t.Helper()
	if len(tail) == 0 {
		t.Fatal("no faded audio was sent after the interruption")
	}
	if len(tail) > maxSamples {
		t.Fatalf("faded tail = %d samples, want at most %d", len(tail), maxSamples)
	}
	prev := full
	for i, v := range tail {
		if v < 0 || v > prev {
			t.Fatalf("envelope not monotonically decreasing at sample %d: %d after %d", i, v, prev)
		}
		prev = v
	}
	if tail[0] < full/2 {
		t.Fatalf("fade starts at %d, a jump from %d", tail[0], full)
	}
	if prev != 0 {
		t.Fatalf("fade ends at %d, want 0", prev)
	}
}

func TestInterruptionFadeRampsQueuedAudioThenClears(t *testing.T) {
	const fade = 60 * time.Millisecond
	h := newOutputHarness(t, TransportParams{InterruptionFade: fade})
	out := h.sess.Output()
	queue(t, out, pcmTTS(constPCM(1000, 8000, 8000), 8000))
	queue(t, out, pcmTTS(constPCM(1000, 8000, 8000), 8000))

	waitUntil(t, time.Second, "bot started speaking", func() bool { return h.up.count("BotStartedSpeakingFrame") == 1 })
	time.Sleep(70 * time.Millisecond)
	interruptedAt := time.Now()
	queue(t, out, frames.NewInterruptionFrame())
	time.Sleep(fade / 2)
	if h.up.count("BotStoppedSpeakingFrame") != 0 {
		t.Fatal("BotStoppedSpeakingFrame emitted before the faded tail played")
	}
	if h.conn.textIndex("clear") >= 0 {
		t.Fatal("clear sent before the faded tail was handed to the wire")
	}
	waitUntil(t, time.Second, "clear message", func() bool { return h.conn.textIndex("clear") >= 0 })
	waitUntil(t, time.Second, "bot stopped speaking", func() bool { return h.up.count("BotStoppedSpeakingFrame") == 1 })
	time.Sleep(100 * time.Millisecond)

	writes := h.conn.snapshot()
	clearAt := h.conn.textIndex("clear")
	var tail []int
	var lastAudio time.Time
	for i, w := range writes {
		if w.msg.Type != serializers.BinaryMessage {
			continue
		}
		if i > clearAt {
			t.Fatalf("audio written after clear at write %d", i)
		}
		lastAudio = w.at
		smp := samplesOf(w.msg.Data)
		if len(tail) == 0 && smp[len(smp)-1] == 8000 {
			continue
		}
		tail = append(tail, smp...)
	}
	assertFadeEnvelope(t, tail, 8000, 480)
	if lat := lastAudio.Sub(interruptedAt); lat > fade+20*time.Millisecond+30*time.Millisecond {
		t.Fatalf("last audio byte %v after the interruption, want at most fade + one chunk", lat)
	}
	if writes[clearAt].at.Before(lastAudio) {
		t.Fatal("clear precedes the last faded chunk")
	}

	queue(t, out, ttsAudio(40, 8000))
	waitUntil(t, time.Second, "audio after the fade", func() bool { return len(h.conn.snapshot()) > clearAt+2 })
	if h.up.count("BotStartedSpeakingFrame") != 2 {
		t.Fatal("expected a new BotStartedSpeakingFrame after the faded interruption")
	}
}

func TestInterruptionFadeUsesQueuedItemsWhenCurrentIsShort(t *testing.T) {
	h := newOutputHarness(t, TransportParams{InterruptionFade: 60 * time.Millisecond})
	out := h.sess.Output()
	for i := 0; i < 40; i++ {
		queue(t, out, pcmTTS(constPCM(20, 8000, 8000), 8000))
	}
	waitUntil(t, time.Second, "playback", func() bool { return len(h.conn.audioWrites()) >= 3 })
	queue(t, out, frames.NewInterruptionFrame())
	waitUntil(t, time.Second, "clear message", func() bool { return h.conn.textIndex("clear") >= 0 })

	var tail []int
	for _, w := range h.conn.audioWrites() {
		smp := samplesOf(w.msg.Data)
		if len(tail) == 0 && smp[len(smp)-1] == 8000 {
			continue
		}
		tail = append(tail, smp...)
	}
	assertFadeEnvelope(t, tail, 8000, 480)
	if len(tail) < 320 {
		t.Fatalf("faded tail = %d samples, queued items were not used", len(tail))
	}
}

func TestInterruptionWithoutFadeNeverAttenuates(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	out := h.sess.Output()
	queue(t, out, pcmTTS(constPCM(1000, 8000, 8000), 8000))
	waitUntil(t, time.Second, "playback", func() bool { return len(h.conn.audioWrites()) >= 3 })
	interruptedAt := time.Now()
	queue(t, out, frames.NewInterruptionFrame())
	waitUntil(t, time.Second, "clear message", func() bool { return h.conn.textIndex("clear") >= 0 })
	time.Sleep(60 * time.Millisecond)

	writes := h.conn.snapshot()
	if d := writes[h.conn.textIndex("clear")].at.Sub(interruptedAt); d > 15*time.Millisecond {
		t.Fatalf("clear sent %v after the interruption with fade disabled", d)
	}
	for _, w := range h.conn.audioWrites() {
		for _, v := range samplesOf(w.msg.Data) {
			if v != 8000 {
				t.Fatalf("sample %d attenuated with fade disabled", v)
			}
		}
		if w.at.After(interruptedAt.Add(15 * time.Millisecond)) {
			t.Fatal("audio written after the interruption with fade disabled")
		}
	}
}

func TestInterruptionFadeNativeMulaw(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{AudioOutNative: true, InterruptionFade: 40 * time.Millisecond}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-f", "CA-f")
	fx := ts.next(t)
	loud := g711.UlawEncodeSample(8000)
	full := int(g711.UlawDecodeSample(loud))
	data := make([]byte, 16000)
	for i := range data {
		data[i] = loud
	}
	f := frames.NewTTSAudioFrame(data, 8000, 1)
	f.SetMetadata("codec", audio.CodecMulaw)
	if err := fx.task.QueueFrame(f); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "playback started", func() bool { return len(c.media()) >= 3 })
	if err := fx.task.QueueFrame(frames.NewInterruptionFrame()); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "clear event", func() bool { return c.index("clear") >= 0 })
	time.Sleep(100 * time.Millisecond)

	msgs := c.all()
	clearAt := c.index("clear")
	var tail []int
	for i, m := range msgs {
		if m.Event != "media" {
			continue
		}
		if i > clearAt {
			t.Fatalf("media sent at %d after clear at %d", i, clearAt)
		}
		if len(tail) == 0 && m.payload[len(m.payload)-1] == loud {
			continue
		}
		for _, b := range m.payload {
			tail = append(tail, int(g711.UlawDecodeSample(b)))
		}
	}
	assertFadeEnvelope(t, tail, full, 320)
}

func TestFadeOutCodecsReachSilenceWithoutAllocating(t *testing.T) {
	pcm := constPCM(40, 8000, 12000)
	ulaw := make([]byte, 320)
	alaw := make([]byte, 320)
	reset := func() {
		for i := 0; i < 320; i++ {
			binary.LittleEndian.PutUint16(pcm[2*i:], 12000)
			ulaw[i] = g711.UlawEncodeSample(12000)
			alaw[i] = g711.AlawEncodeSample(12000)
		}
	}
	if n := testing.AllocsPerRun(20, func() {
		reset()
		FadeOut(pcm, audio.CodecLinear16)
		FadeOut(ulaw, audio.CodecMulaw)
		FadeOut(alaw, audio.CodecAlaw)
	}); n != 0 {
		t.Fatalf("FadeOut allocates %.1f times per call", n)
	}
	decoded := make([]int, 320)
	for i := range decoded {
		decoded[i] = int(g711.UlawDecodeSample(ulaw[i]))
	}
	assertFadeEnvelope(t, decoded, int(g711.UlawDecodeSample(g711.UlawEncodeSample(12000))), 320)
	for i := range decoded {
		decoded[i] = int(g711.AlawDecodeSample(alaw[i]))
	}
	if decoded[319] > 16 {
		t.Fatalf("A-law fade ends at %d", decoded[319])
	}
	for i := 1; i < 320; i++ {
		if decoded[i] > decoded[i-1] {
			t.Fatalf("A-law envelope rises at %d", i)
		}
	}
	assertFadeEnvelope(t, samplesOf(pcm), 12000, 320)
	if FadeSamples(8000, 60*time.Millisecond) != 480 || FadeSamples(8000, 0) != 0 {
		t.Fatal("FadeSamples")
	}
}
