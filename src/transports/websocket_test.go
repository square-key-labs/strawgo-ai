package transports

import (
	"bytes"
	"context"
	"errors"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type testSerializer struct {
	hangups atomic.Int32
	native  bool
}

func (s *testSerializer) WireSampleRates() (int, int) { return 8000, 8000 }

func (s *testSerializer) SerializeAudio(dst []serializers.Message, pcm []byte, sampleRate int) ([]serializers.Message, error) {
	return append(dst, serializers.Binary(pcm)), nil
}

func (s *testSerializer) Serialize(dst []serializers.Message, frame frames.Frame) ([]serializers.Message, error) {
	switch f := frame.(type) {
	case *frames.InterruptionFrame:
		return append(dst, serializers.Text("clear")), nil
	case *frames.EndFrame:
		return append(dst, serializers.Text("end")), nil
	case *frames.OutputTransportMessageFrame:
		return append(dst, serializers.Text("msg:"+f.Message.(string))), nil
	case *frames.OutputTransportMessageUrgentFrame:
		return append(dst, serializers.Text("urgent:"+f.Message.(string))), nil
	}
	return dst, nil
}

func (s *testSerializer) Deserialize(msg serializers.Message) (frames.Frame, error) {
	if msg.Type == serializers.BinaryMessage {
		f := frames.NewAudioFrame(msg.Data, 8000, 1)
		f.SetMetadata("codec", "linear16")
		return f, nil
	}
	text := string(msg.Data)
	switch {
	case strings.HasPrefix(text, "dtmf:"):
		return frames.NewInputDTMFFrame(frames.KeypadEntry(text[5:])), nil
	case strings.HasPrefix(text, "ack:"):
		f := frames.NewPlaybackCompleteFrame()
		f.SetMetadata("correlation_id", text[4:])
		return f, nil
	case text == "stop":
		return frames.NewEndFrame(), nil
	}
	return nil, nil
}

func (s *testSerializer) SerializePlaybackAck(id string) (serializers.Message, error) {
	return serializers.Text("mark:" + id), nil
}

func (s *testSerializer) Hangup(ctx context.Context) error {
	s.hangups.Add(1)
	return nil
}

type fakeWrite struct {
	msg serializers.Message
	at  time.Time
}

type fakeConn struct {
	in     chan serializers.Message
	closed chan struct{}
	once   sync.Once
	mu     sync.Mutex
	writes []fakeWrite
}

func newFakeConn() *fakeConn {
	return &fakeConn{in: make(chan serializers.Message, 64), closed: make(chan struct{})}
}

func (c *fakeConn) ReadMessage() (serializers.Message, error) {
	select {
	case m := <-c.in:
		return m, nil
	case <-c.closed:
		return serializers.Message{}, errConnClosed
	}
}

func (c *fakeConn) WriteMessage(m serializers.Message) error {
	select {
	case <-c.closed:
		return errConnClosed
	default:
	}
	c.mu.Lock()
	c.writes = append(c.writes, fakeWrite{msg: serializers.Message{Type: m.Type, Data: bytes.Clone(m.Data)}, at: time.Now()})
	c.mu.Unlock()
	return nil
}

func (c *fakeConn) Ping() error { return nil }

func (c *fakeConn) Close() error {
	c.once.Do(func() { close(c.closed) })
	return nil
}

func (c *fakeConn) isClosed() bool {
	select {
	case <-c.closed:
		return true
	default:
		return false
	}
}

func (c *fakeConn) snapshot() []fakeWrite {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]fakeWrite(nil), c.writes...)
}

func (c *fakeConn) audioWrites() []fakeWrite {
	var out []fakeWrite
	for _, w := range c.snapshot() {
		if w.msg.Type == serializers.BinaryMessage {
			out = append(out, w)
		}
	}
	return out
}

func (c *fakeConn) textIndex(text string) int {
	for i, w := range c.snapshot() {
		if w.msg.Type == serializers.TextMessage && string(w.msg.Data) == text {
			return i
		}
	}
	return -1
}

type harness struct {
	sess *Session
	conn *fakeConn
	ser  *testSerializer
	up   *queuedFrameCapture
	down *queuedFrameCapture
}

func newOutputHarness(t *testing.T, params TransportParams) *harness {
	t.Helper()
	h := &harness{conn: newFakeConn(), ser: &testSerializer{}, up: &queuedFrameCapture{}, down: &queuedFrameCapture{}}
	h.sess = newSession(h.conn, h.ser, params, serializers.CallInfo{}, nil)
	out := h.sess.Output()
	out.Link(h.down)
	out.SetPrev(h.up)
	if err := out.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = out.Stop()
		_ = h.sess.Close()
	})
	queue(t, out, frames.NewStartFrame())
	return h
}

func queue(t *testing.T, p processors.FrameProcessor, f frames.Frame) {
	t.Helper()
	if err := p.QueueFrame(f, frames.Downstream); err != nil {
		t.Fatalf("QueueFrame(%s): %v", f.Name(), err)
	}
}

func ttsAudio(ms, rate int) *frames.TTSAudioFrame {
	f := frames.NewTTSAudioFrame(bytes.Repeat([]byte{1, 2}, rate*ms/1000), rate, 1)
	f.SetMetadata("codec", "linear16")
	return f
}

func waitUntil(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

type echoHandler struct {
	p *processors.BaseProcessor
}

func (e *echoHandler) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if f, ok := frame.(*frames.AudioFrame); ok && direction == frames.Downstream {
		out := frames.NewTTSAudioFrame(f.Data, f.SampleRate, 1)
		out.SetMetadata("codec", "linear16")
		return e.p.PushFrame(out, direction)
	}
	return e.p.PushFrame(frame, direction)
}

func newEcho() *processors.BaseProcessor {
	h := &echoHandler{}
	h.p = processors.NewBaseProcessor("echo", h)
	return h.p
}

func TestServerSessionsAreIsolatedAndEndFrameFinishesPipeline(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	var finished atomic.Int32
	tasks := make(chan *pipeline.PipelineTask, 2)
	srv, err := NewServer(ServerConfig{
		Path:          "/ws",
		NewSerializer: func(*http.Request) serializers.FrameSerializer { return &testSerializer{} },
		Params:        TransportParams{AudioInSampleRate: 8000},
		Handler: func(ctx context.Context, s *Session) error {
			pipe := pipeline.NewPipeline([]processors.FrameProcessor{s.Input(), newEcho(), s.Output()})
			task := pipeline.NewPipelineTask(pipe)
			tasks <- task
			err := task.Run(ctx)
			finished.Add(1)
			return err
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() { _ = srv.Serve(ctx, ln) }()

	url := "ws://" + ln.Addr().String() + "/ws"
	dial := func() *websocket.Conn {
		c, _, err := websocket.DefaultDialer.Dial(url, nil)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	a, b := dial(), dial()
	defer a.Close()
	defer b.Close()

	payloadA := bytes.Repeat([]byte{0xA, 0}, 160)
	payloadB := bytes.Repeat([]byte{0xB, 0}, 160)
	if err := a.WriteMessage(websocket.BinaryMessage, payloadA); err != nil {
		t.Fatal(err)
	}
	if err := b.WriteMessage(websocket.BinaryMessage, payloadB); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		conn *websocket.Conn
		want []byte
	}{{a, payloadA}, {b, payloadB}} {
		_ = tc.conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		typ, data, err := tc.conn.ReadMessage()
		if err != nil {
			t.Fatal(err)
		}
		if typ != websocket.BinaryMessage || !bytes.Equal(data, tc.want) {
			t.Fatalf("connection received foreign or corrupt audio: %x", data[:4])
		}
	}

	first := <-tasks
	second := <-tasks
	if err := first.QueueFrame(frames.NewEndFrame()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-first.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("EndFrame did not finish the pipeline")
	}

	if err := b.WriteMessage(websocket.TextMessage, []byte("stop")); err != nil {
		t.Fatal(err)
	}
	select {
	case <-second.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("remote stop did not finish the pipeline")
	}
	waitUntil(t, 3*time.Second, "handlers to return", func() bool { return finished.Load() == 2 })
}

type transferSerializer struct {
	testSerializer
	err error
	mu  sync.Mutex
	req []serializers.TransferRequest
	at  time.Time
}

func (s *transferSerializer) CanTransfer() bool { return true }

func (s *transferSerializer) Transfer(ctx context.Context, req serializers.TransferRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.req = append(s.req, req)
	s.at = time.Now()
	return s.err
}

func (s *transferSerializer) requests() ([]serializers.TransferRequest, time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]serializers.TransferRequest(nil), s.req...), s.at
}

func startTransferOutput(t *testing.T, ser serializers.FrameSerializer) (*Session, *fakeConn, *queuedFrameCapture) {
	t.Helper()
	conn, up := newFakeConn(), &queuedFrameCapture{}
	sess := newSession(conn, ser, TransportParams{}, serializers.CallInfo{}, nil)
	out := sess.Output()
	out.Link(&queuedFrameCapture{})
	out.SetPrev(up)
	if err := out.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = out.Stop()
		_ = sess.Close()
	})
	queue(t, out, frames.NewStartFrame())
	return sess, conn, up
}

func transferFrame(dest string) (*frames.TransferCallFrame, chan error) {
	done := make(chan error, 1)
	f := frames.NewTransferCallFrame(dest)
	f.CallerID = "+14155550199"
	f.OnComplete = func(err error) { done <- err }
	return f, done
}

func awaitTransfer(t *testing.T, done chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("transfer did not complete")
		return nil
	}
}

func TestTransferCallFrameRunsAfterQueuedSpeechAndEndsTask(t *testing.T) {
	ser := &transferSerializer{}
	sess, conn, up := startTransferOutput(t, ser)
	queue(t, sess.Output(), ttsAudio(200, 8000))
	f, done := transferFrame("+14155550100")
	queue(t, sess.Output(), f)
	if err := awaitTransfer(t, done); err != nil {
		t.Fatal(err)
	}
	reqs, at := ser.requests()
	if len(reqs) != 1 || reqs[0].Destination != "+14155550100" || reqs[0].CallerID != "+14155550199" {
		t.Fatalf("requests = %+v", reqs)
	}
	audio := conn.audioWrites()
	if len(audio) == 0 || at.Before(audio[len(audio)-1].at) {
		t.Fatalf("transfer ran before queued speech was sent (%d audio writes)", len(audio))
	}
	waitUntil(t, 2*time.Second, "result and end task upstream", func() bool {
		return up.count("TransferCallResultFrame") == 1 && up.count("EndTaskFrame") == 1
	})
	if r := up.find("TransferCallResultFrame").(*frames.TransferCallResultFrame); !r.Succeeded() || r.Destination != "+14155550100" {
		t.Fatalf("result = %+v", r)
	}
}

func TestTransferCallFrameFailureIsReportedAndRedacted(t *testing.T) {
	ser := &transferSerializer{err: errors.New("carrier said Authorization: Bearer abcdefghijklmnop123")}
	sess, _, up := startTransferOutput(t, ser)
	f, done := transferFrame("+14155550100")
	queue(t, sess.Output(), f)
	err := awaitTransfer(t, done)
	if err == nil || strings.Contains(err.Error(), "abcdefghijklmnop123") {
		t.Fatalf("err = %v", err)
	}
	waitUntil(t, 2*time.Second, "result upstream", func() bool { return up.count("TransferCallResultFrame") == 1 })
	if r := up.find("TransferCallResultFrame").(*frames.TransferCallResultFrame); r.Succeeded() {
		t.Fatal("failed transfer reported success")
	}
	time.Sleep(50 * time.Millisecond)
	if up.count("EndTaskFrame") != 0 {
		t.Fatal("failed transfer ended the task")
	}
}

func TestTransferCallFrameWithoutTransfererIsUnsupported(t *testing.T) {
	sess, _, up := startTransferOutput(t, &testSerializer{})
	f, done := transferFrame("+14155550100")
	queue(t, sess.Output(), f)
	if err := awaitTransfer(t, done); !errors.Is(err, serializers.ErrTransferUnsupported) {
		t.Fatalf("err = %v", err)
	}
	waitUntil(t, 2*time.Second, "result upstream", func() bool { return up.count("TransferCallResultFrame") == 1 })
}
