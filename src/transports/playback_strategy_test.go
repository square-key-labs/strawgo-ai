package transports

import (
	"context"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type inputHarness struct {
	sess *Session
	conn *fakeConn
	up   *queuedFrameCapture
	down *queuedFrameCapture
}

func newInputHarness(t *testing.T, params TransportParams, startRate int) *inputHarness {
	t.Helper()
	h := &inputHarness{conn: newFakeConn(), up: &queuedFrameCapture{}, down: &queuedFrameCapture{}}
	h.sess = newSession(h.conn, &testSerializer{}, params, serializers.CallInfo{}, nil)
	in := h.sess.Input()
	in.Link(h.down)
	in.SetPrev(h.up)
	if err := in.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = in.Stop()
		_ = h.sess.Close()
		h.sess.Wait()
	})
	h.sess.start()
	start := frames.NewStartFrame()
	start.AudioInSampleRate = startRate
	queue(t, in, start)
	return h
}

func TestInputDeliversAudioDTMFAndConnectedFrames(t *testing.T) {
	h := newInputHarness(t, TransportParams{}, 16000)
	h.conn.in <- serializers.Binary(make([]byte, 320))
	h.conn.in <- serializers.Text("dtmf:5")

	waitUntil(t, time.Second, "DTMF frame", func() bool { return h.down.count("InputDTMFFrame") == 1 })
	if h.down.count("ClientConnectedFrame") != 1 {
		t.Fatal("ClientConnectedFrame not pushed on start")
	}
	af, ok := h.down.find("AudioFrame").(*frames.AudioFrame)
	if !ok {
		t.Fatal("audio frame not delivered")
	}
	if af.SampleRate != 16000 || len(af.Data) < 636 || len(af.Data) > 640 {
		t.Fatalf("audio = %d bytes at %d Hz, want ~640 bytes at 16000 Hz", len(af.Data), af.SampleRate)
	}
	if d := h.down.find("InputDTMFFrame").(*frames.InputDTMFFrame); d.Button != frames.KeypadFive {
		t.Fatalf("DTMF button = %q", d.Button)
	}
}

func TestInputRemoteDisconnectCancelsTask(t *testing.T) {
	h := newInputHarness(t, TransportParams{}, 0)
	waitUntil(t, time.Second, "connected", func() bool { return h.down.count("ClientConnectedFrame") == 1 })
	h.conn.in <- serializers.Text("stop")

	waitUntil(t, time.Second, "disconnect handling", func() bool {
		return h.down.count("ClientDisconnectedFrame") == 1 && h.up.count("CancelTaskFrame") == 1
	})
}

func TestInputSessionTimeoutEndsTask(t *testing.T) {
	h := newInputHarness(t, TransportParams{SessionTimeout: 50 * time.Millisecond}, 0)
	waitUntil(t, time.Second, "session timeout", func() bool { return h.up.count("EndTaskFrame") == 1 })
}

func TestOutputDTMFFallsBackToInBandTones(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	f, err := frames.NewOutputDTMFFrameFromString("1")
	if err != nil {
		t.Fatal(err)
	}
	queue(t, h.sess.Output(), f)
	waitUntil(t, 2*time.Second, "DTMF frame downstream", func() bool { return h.down.count("OutputDTMFFrame") == 1 })
	if n := len(h.conn.audioWrites()); n != 25 {
		t.Fatalf("DTMF tone chunks = %d, want 25 (500 ms)", n)
	}
	if h.up.count("BotStartedSpeakingFrame") != 0 {
		t.Fatal("DTMF tones must not count as bot speech")
	}
}

func TestOutputTransportMessagesOrderedAndUrgent(t *testing.T) {
	h := newOutputHarness(t, TransportParams{})
	out := h.sess.Output()
	queue(t, out, ttsAudio(200, 8000))
	queue(t, out, frames.NewOutputTransportMessageFrame("after-audio"))
	waitUntil(t, time.Second, "playback", func() bool { return len(h.conn.audioWrites()) > 0 })
	queue(t, out, frames.NewOutputTransportMessageUrgentFrame("now"))

	waitUntil(t, 2*time.Second, "ordered message", func() bool { return h.conn.textIndex("msg:after-audio") >= 0 })
	urgent, ordered := h.conn.textIndex("urgent:now"), h.conn.textIndex("msg:after-audio")
	if urgent < 0 || urgent > 5 {
		t.Fatalf("urgent message at write %d, expected ahead of queued audio", urgent)
	}
	if ordered != len(h.conn.snapshot())-1 || len(h.conn.audioWrites()) != 10 {
		t.Fatalf("ordered message at %d was not sent after all queued audio", ordered)
	}
}
