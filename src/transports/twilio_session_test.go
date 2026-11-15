package transports

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/audio/filter"
	"github.com/square-key-labs/strawgo-ai/src/audio/mixer"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

type frameLog struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (l *frameLog) add(f frames.Frame) {
	l.mu.Lock()
	l.frames = append(l.frames, f)
	l.mu.Unlock()
}

func (l *frameLog) all() []frames.Frame {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]frames.Frame(nil), l.frames...)
}

func (l *frameLog) count(name string) int {
	n := 0
	for _, f := range l.all() {
		if f.Name() == name {
			n++
		}
	}
	return n
}

func logged[T frames.Frame](l *frameLog) []T {
	var out []T
	for _, f := range l.all() {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

type tapHandler struct {
	p    *processors.BaseProcessor
	down frameLog
	up   frameLog
}

func (h *tapHandler) HandleFrame(ctx context.Context, f frames.Frame, dir frames.FrameDirection) error {
	if dir == frames.Downstream {
		h.down.add(f)
	} else {
		h.up.add(f)
	}
	return h.p.PushFrame(f, dir)
}

func newTap() *tapHandler {
	h := &tapHandler{}
	h.p = processors.NewBaseProcessor("tap", h)
	return h
}

type callFixture struct {
	sess        *Session
	task        *pipeline.PipelineTask
	tap         *tapHandler
	sink        *frameLog
	handlerDone chan struct{}
}

type twServer struct {
	url   string
	calls chan *callFixture
}

func startTwilioServer(t *testing.T, params TransportParams, cfg serializers.TwilioConfig) *twServer {
	t.Helper()
	ts := &twServer{calls: make(chan *callFixture, 8)}
	var mu sync.Mutex
	var fixtures []*callFixture
	srv, err := NewServer(ServerConfig{
		NewSerializer: func(*http.Request) serializers.FrameSerializer { return serializers.NewTwilioFrameSerializer(cfg) },
		Params:        params,
		Handler: func(ctx context.Context, s *Session) error {
			tap := newTap()
			pipe := pipeline.NewPipeline([]processors.FrameProcessor{s.Input(), tap.p, s.Output()})
			tc := pipeline.DefaultPipelineTaskConfig()
			tc.AudioInSampleRate = 16000
			tc.AudioOutSampleRate = 24000
			task := pipeline.NewPipelineTaskWithConfig(pipe, tc)
			fx := &callFixture{sess: s, task: task, tap: tap, sink: &frameLog{}, handlerDone: make(chan struct{})}
			task.OnFrameReachedDownstream(fx.sink.add)
			mu.Lock()
			fixtures = append(fixtures, fx)
			mu.Unlock()
			defer close(fx.handlerDone)
			ts.calls <- fx
			return task.Run(ctx)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	hs := httptest.NewServer(srv)
	ts.url = "ws" + strings.TrimPrefix(hs.URL, "http") + "/ws"
	t.Cleanup(func() {
		srv.Shutdown()
		mu.Lock()
		list := append([]*callFixture(nil), fixtures...)
		mu.Unlock()
		for _, fx := range list {
			select {
			case <-fx.handlerDone:
			case <-time.After(5 * time.Second):
				t.Errorf("session handler did not return after shutdown")
			}
		}
		hs.Close()
	})
	return ts
}

func (ts *twServer) next(t *testing.T) *callFixture {
	t.Helper()
	select {
	case fx := <-ts.calls:
		return fx
	case <-time.After(3 * time.Second):
		t.Fatal("no session started")
		return nil
	}
}

type twMsg struct {
	Event     string `json:"event"`
	StreamSid string `json:"streamSid"`
	Media     *struct {
		Payload string `json:"payload"`
	} `json:"media"`
	payload []byte
	at      time.Time
}

type twClient struct {
	ws     *websocket.Conn
	wmu    sync.Mutex
	mu     sync.Mutex
	msgs   []twMsg
	closed chan struct{}
}

func dialTwilio(t *testing.T, url, streamSid, callSid string) *twClient {
	t.Helper()
	ws, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatal(err)
	}
	c := &twClient{ws: ws, closed: make(chan struct{})}
	t.Cleanup(func() { _ = ws.Close() })
	go c.readLoop()
	c.sendJSON(t, map[string]any{"event": "connected", "protocol": "Call", "version": "1.0.0"})
	c.sendJSON(t, map[string]any{
		"event":     "start",
		"streamSid": streamSid,
		"start": map[string]any{
			"streamSid":        streamSid,
			"callSid":          callSid,
			"accountSid":       "AC1",
			"customParameters": map[string]string{"lang": "en", "caller": streamSid},
			"mediaFormat":      map[string]any{"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
		},
	})
	return c
}

func (c *twClient) readLoop() {
	defer close(c.closed)
	for {
		_, data, err := c.ws.ReadMessage()
		if err != nil {
			return
		}
		var m twMsg
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		m.at = time.Now()
		if m.Media != nil {
			m.payload, _ = base64.StdEncoding.DecodeString(m.Media.Payload)
		}
		c.mu.Lock()
		c.msgs = append(c.msgs, m)
		c.mu.Unlock()
	}
}

func (c *twClient) sendJSON(t *testing.T, v any) {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	c.wmu.Lock()
	defer c.wmu.Unlock()
	if err := c.ws.WriteMessage(websocket.TextMessage, data); err != nil {
		t.Fatal(err)
	}
}

func (c *twClient) sendMedia(t *testing.T, streamSid string, ulaw []byte) {
	t.Helper()
	c.sendJSON(t, map[string]any{"event": "media", "streamSid": streamSid, "media": map[string]any{"track": "inbound", "payload": base64.StdEncoding.EncodeToString(ulaw)}})
}

func (c *twClient) all() []twMsg {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]twMsg(nil), c.msgs...)
}

func (c *twClient) media() []twMsg {
	var out []twMsg
	for _, m := range c.all() {
		if m.Event == "media" {
			out = append(out, m)
		}
	}
	return out
}

func (c *twClient) index(event string) int {
	for i, m := range c.all() {
		if m.Event == event {
			return i
		}
	}
	return -1
}

func (c *twClient) waitClosed(t *testing.T, what string) {
	t.Helper()
	select {
	case <-c.closed:
	case <-time.After(3 * time.Second):
		t.Fatalf("connection not closed: %s", what)
	}
}

func waitDone(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func constPCM(ms, rate int, v int16) []byte {
	n := rate * ms / 1000
	out := make([]byte, 0, 2*n)
	for i := 0; i < n; i++ {
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}
	return out
}

func pcmTTS(pcm []byte, rate int) *frames.TTSAudioFrame {
	f := frames.NewTTSAudioFrame(pcm, rate, 1)
	f.SetMetadata("codec", audio.CodecLinear16)
	return f
}

func ulawOf(t *testing.T, pcm []byte) []byte {
	t.Helper()
	out, err := audio.EncodePCM16(nil, pcm, audio.CodecMulaw)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func meanSample(t *testing.T, ulaw []byte) float64 {
	t.Helper()
	pcm, err := audio.DecodePCM16(nil, ulaw, audio.CodecMulaw)
	if err != nil {
		t.Fatal(err)
	}
	if len(pcm) < 2 {
		return 0
	}
	sum := 0.0
	for i := 0; i+1 < len(pcm); i += 2 {
		sum += float64(int16(binary.LittleEndian.Uint16(pcm[i:])))
	}
	return sum / float64(len(pcm)/2)
}

func TestTwilioCallInfoParsedFromStartEvent(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	dialTwilio(t, ts.url, "MZ-info", "CA-info")
	fx := ts.next(t)
	info := fx.sess.CallInfo()
	if info.Provider != "twilio" || info.StreamID != "MZ-info" || info.CallID != "CA-info" || info.AccountID != "AC1" {
		t.Fatalf("call info = %+v", info)
	}
	if info.Encoding != audio.CodecMulaw || info.SampleRate != 8000 {
		t.Fatalf("media format = %s/%d", info.Encoding, info.SampleRate)
	}
	if info.CustomParameters["lang"] != "en" || info.CustomParameters["caller"] != "MZ-info" {
		t.Fatalf("custom parameters = %v", info.CustomParameters)
	}
	if fx.sess.Request() == nil || fx.sess.Request().URL.Path != "/ws" {
		t.Fatalf("upgrade request not exposed on session")
	}
}

func TestTwilioInboundMulawBecomesPipelineRateAudio(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-in", "CA-in")
	fx := ts.next(t)
	waitUntil(t, 2*time.Second, "ClientConnectedFrame", func() bool { return fx.tap.down.count("ClientConnectedFrame") == 1 })

	chunk := ulawOf(t, constPCM(20, 8000, 3000))
	const n = 10
	for i := 0; i < n; i++ {
		c.sendMedia(t, "MZ-in", chunk)
	}
	total := func() int {
		sum := 0
		for _, f := range logged[*frames.AudioFrame](&fx.tap.down) {
			sum += len(f.Data)
		}
		return sum
	}
	waitUntil(t, 2*time.Second, "inbound audio frames", func() bool { return len(logged[*frames.AudioFrame](&fx.tap.down)) == n })
	for _, f := range logged[*frames.AudioFrame](&fx.tap.down) {
		if f.SampleRate != 16000 || f.Channels != 1 {
			t.Fatalf("inbound frame %d Hz/%d ch, want 16000 Hz mono", f.SampleRate, f.Channels)
		}
		if codec, _ := f.Metadata()["codec"].(string); codec != audio.CodecLinear16 {
			t.Fatalf("inbound codec = %q", codec)
		}
	}
	if got, want := total(), n*640; got < want-64 || got > want {
		t.Fatalf("inbound bytes = %d, want ~%d (200 ms at 16 kHz)", got, want)
	}
	last := logged[*frames.AudioFrame](&fx.tap.down)[n-1]
	mean := 0.0
	for i := 0; i+1 < len(last.Data); i += 2 {
		mean += float64(int16(binary.LittleEndian.Uint16(last.Data[i:])))
	}
	mean /= float64(len(last.Data) / 2)
	if mean < 2700 || mean > 3300 {
		t.Fatalf("decoded level = %.0f, want ~3000", mean)
	}
}

func TestTwilioDTMFBecomesInputDTMFFrame(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-d", "CA-d")
	fx := ts.next(t)
	c.sendJSON(t, map[string]any{"event": "dtmf", "streamSid": "MZ-d", "dtmf": map[string]any{"track": "inbound_track", "digit": "7"}})
	c.sendJSON(t, map[string]any{"event": "dtmf", "streamSid": "MZ-d", "dtmf": map[string]any{"track": "inbound_track", "digit": "#"}})
	waitUntil(t, 2*time.Second, "two DTMF frames", func() bool { return len(logged[*frames.InputDTMFFrame](&fx.tap.down)) == 2 })
	got := logged[*frames.InputDTMFFrame](&fx.tap.down)
	if got[0].Button != frames.KeypadEntry("7") || got[1].Button != frames.KeypadEntry("#") {
		t.Fatalf("DTMF buttons = %q, %q", got[0].Button, got[1].Button)
	}
}

func TestTwilioConcurrentCallsAreIsolated(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	a := dialTwilio(t, ts.url, "MZ-A", "CA-A")
	fa := ts.next(t)
	b := dialTwilio(t, ts.url, "MZ-B", "CA-B")
	fb := ts.next(t)
	if fa.sess.ID() == fb.sess.ID() {
		t.Fatal("sessions share an id")
	}
	if fa.sess.CallInfo().StreamID != "MZ-A" || fb.sess.CallInfo().StreamID != "MZ-B" {
		t.Fatalf("call info crossed: %s / %s", fa.sess.CallInfo().StreamID, fb.sess.CallInfo().StreamID)
	}

	if err := fa.task.QueueFrame(pcmTTS(constPCM(200, 8000, 6000), 8000)); err != nil {
		t.Fatal(err)
	}
	if err := fb.task.QueueFrame(pcmTTS(constPCM(200, 8000, -6000), 8000)); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		a.sendMedia(t, "MZ-A", ulawOf(t, constPCM(20, 8000, 1000)))
	}
	waitUntil(t, 3*time.Second, "both calls receive their audio", func() bool {
		return len(a.media()) >= 10 && len(b.media()) >= 10
	})
	for _, tc := range []struct {
		c    *twClient
		sid  string
		sign float64
	}{{a, "MZ-A", 1}, {b, "MZ-B", -1}} {
		for _, m := range tc.c.media() {
			if m.StreamSid != tc.sid {
				t.Fatalf("%s received media for stream %s", tc.sid, m.StreamSid)
			}
			if v := meanSample(t, m.payload) * tc.sign; v < 5000 {
				t.Fatalf("%s received foreign audio (mean %.0f)", tc.sid, v*tc.sign)
			}
		}
	}
	waitUntil(t, 2*time.Second, "call A inbound audio", func() bool { return len(logged[*frames.AudioFrame](&fa.tap.down)) == 5 })
	if n := len(logged[*frames.AudioFrame](&fb.tap.down)); n != 0 {
		t.Fatalf("call B pipeline received %d of call A's inbound frames", n)
	}
}

func TestTwilioOutboundAudioPacedInRealTime(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-p", "CA-p")
	fx := ts.next(t)
	start := time.Now()
	if err := fx.task.QueueFrame(pcmTTS(constPCM(400, 24000, 2000), 24000)); err != nil {
		t.Fatal(err)
	}
	if err := fx.task.QueueFrame(frames.NewTTSStoppedFrame()); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 3*time.Second, "TTSStoppedFrame at sink", func() bool { return fx.sink.count("TTSStoppedFrame") == 1 })
	waitUntil(t, time.Second, "21 media messages", func() bool { return len(c.media()) >= 21 })
	media := c.media()
	if len(media) != 21 {
		t.Fatalf("media messages = %d, want 21 (400 ms plus the flushed resampler tail in 20 ms chunks)", len(media))
	}
	for i, m := range media {
		if len(m.payload) != 160 {
			t.Fatalf("chunk %d = %d bytes, want 160 (20 ms of 8 kHz mu-law)", i, len(m.payload))
		}
	}
	span := media[len(media)-1].at.Sub(media[0].at)
	if span < 330*time.Millisecond || span > 1500*time.Millisecond {
		t.Fatalf("400 ms of audio spread over %v, want ~380 ms of real-time pacing", span)
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("playback took %v", elapsed)
	}
	if v := meanSample(t, media[10].payload); v < 1700 || v > 2300 {
		t.Fatalf("resampled level = %.0f, want ~2000", v)
	}
}

func TestTwilioInterruptionSendsClearAndStopsAudio(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-i", "CA-i")
	fx := ts.next(t)
	if err := fx.task.QueueFrame(pcmTTS(constPCM(2000, 8000, 2000), 8000)); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "playback started", func() bool { return len(c.media()) >= 3 })
	if err := fx.task.QueueFrame(frames.NewInterruptionFrame()); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "clear event", func() bool { return c.index("clear") >= 0 })
	waitUntil(t, 2*time.Second, "BotStoppedSpeakingFrame upstream", func() bool { return fx.tap.up.count("BotStoppedSpeakingFrame") == 1 })
	time.Sleep(150 * time.Millisecond)
	msgs := c.all()
	clearAt := c.index("clear")
	if msgs[clearAt].StreamSid != "MZ-i" {
		t.Fatalf("clear for stream %q", msgs[clearAt].StreamSid)
	}
	for i := clearAt + 1; i < len(msgs); i++ {
		if msgs[i].Event == "media" {
			t.Fatalf("media sent at %d after clear at %d", i, clearAt)
		}
	}
	if n := len(c.media()); n > 30 {
		t.Fatalf("%d chunks sent before interruption took effect", n)
	}

	if err := fx.task.QueueFrame(pcmTTS(constPCM(60, 8000, 2000), 8000)); err != nil {
		t.Fatal(err)
	}
	before := len(c.media())
	waitUntil(t, 2*time.Second, "audio resumes after interruption", func() bool { return len(c.media()) >= before+3 })
}

func TestTwilioBotSpeakingFramesBothDirections(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-b", "CA-b")
	fx := ts.next(t)
	if err := fx.task.QueueFrame(pcmTTS(constPCM(300, 8000, 2000), 8000)); err != nil {
		t.Fatal(err)
	}
	if err := fx.task.QueueFrame(frames.NewTTSStoppedFrame()); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 3*time.Second, "BotStoppedSpeakingFrame in both directions", func() bool {
		return fx.sink.count("BotStoppedSpeakingFrame") == 1 && fx.tap.up.count("BotStoppedSpeakingFrame") == 1
	})
	waitUntil(t, time.Second, "all 15 chunks", func() bool { return len(c.media()) >= 15 })
	if n := len(c.media()); n != 15 {
		t.Fatalf("got %d chunks, want 15", n)
	}
	for name, l := range map[string]*frameLog{"upstream": &fx.tap.up, "downstream": fx.sink} {
		order := []string{}
		for _, f := range l.all() {
			switch f.Name() {
			case "BotStartedSpeakingFrame", "BotStoppedSpeakingFrame":
				order = append(order, f.Name())
			}
		}
		if len(order) != 2 || order[0] != "BotStartedSpeakingFrame" || order[1] != "BotStoppedSpeakingFrame" {
			t.Fatalf("%s bot speaking order = %v", name, order)
		}
		if l.count("BotSpeakingFrame") == 0 {
			t.Fatalf("no BotSpeakingFrame %s", name)
		}
	}
}

func TestTwilioEndFrameFlushesHangsUpClosesAndForwards(t *testing.T) {
	type hangup struct {
		path, user, pass, status string
	}
	hangups := make(chan hangup, 4)
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, pass, _ := r.BasicAuth()
		_ = r.ParseForm()
		hangups <- hangup{path: r.URL.Path, user: user, pass: pass, status: r.PostForm.Get("Status")}
		_, _ = w.Write([]byte(`{}`))
	}))
	defer api.Close()

	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{AuthToken: "tok", BaseURL: api.URL})
	c := dialTwilio(t, ts.url, "MZ-e", "CA-e")
	fx := ts.next(t)
	if err := fx.task.QueueFrame(pcmTTS(constPCM(200, 8000, 2000), 8000)); err != nil {
		t.Fatal(err)
	}
	if err := fx.task.QueueFrame(frames.NewEndFrame()); err != nil {
		t.Fatal(err)
	}
	waitDone(t, fx.task.Done(), "pipeline finished by EndFrame")
	if fx.sink.count("EndFrame") != 1 {
		t.Fatal("EndFrame not forwarded to the pipeline sink")
	}
	c.waitClosed(t, "after EndFrame")
	if n := len(c.media()); n != 10 {
		t.Fatalf("connection closed after %d of 10 queued chunks", n)
	}
	select {
	case h := <-hangups:
		if h.path != "/2010-04-01/Accounts/AC1/Calls/CA-e.json" || h.user != "AC1" || h.pass != "tok" || h.status != "completed" {
			t.Fatalf("hangup request = %+v", h)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no hangup REST call after EndFrame")
	}
	waitDone(t, fx.handlerDone, "handler return")
	select {
	case h := <-hangups:
		t.Fatalf("second hangup request %+v", h)
	default:
	}
}

func TestTwilioClientStopEndsSession(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-s", "CA-s")
	fx := ts.next(t)
	waitUntil(t, 2*time.Second, "ClientConnectedFrame", func() bool { return fx.tap.down.count("ClientConnectedFrame") == 1 })
	c.sendJSON(t, map[string]any{"event": "stop", "streamSid": "MZ-s", "stop": map[string]any{"callSid": "CA-s"}})
	waitDone(t, fx.task.Done(), "pipeline finished after client stop")
	waitDone(t, fx.handlerDone, "handler return")
	waitDone(t, fx.sess.Done(), "session closed")
	if fx.tap.down.count("ClientDisconnectedFrame") != 1 {
		t.Fatal("ClientDisconnectedFrame not pushed")
	}
	c.waitClosed(t, "after stop")
}

func TestTwilioClientDropEndsSession(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-x", "CA-x")
	fx := ts.next(t)
	waitUntil(t, 2*time.Second, "ClientConnectedFrame", func() bool { return fx.tap.down.count("ClientConnectedFrame") == 1 })
	_ = c.ws.Close()
	waitDone(t, fx.task.Done(), "pipeline finished after socket drop")
	waitDone(t, fx.handlerDone, "handler return")
	if fx.tap.down.count("ClientDisconnectedFrame") != 1 {
		t.Fatal("ClientDisconnectedFrame not pushed")
	}
}

func TestTwilioDisconnectCallbackReplacesCancel(t *testing.T) {
	var called atomic.Int32
	ts := startTwilioServer(t, TransportParams{OnClientDisconnected: func(s *Session) {
		called.Add(1)
		_ = s.Close()
	}}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-cb", "CA-cb")
	fx := ts.next(t)
	waitUntil(t, 2*time.Second, "ClientConnectedFrame", func() bool { return fx.tap.down.count("ClientConnectedFrame") == 1 })
	_ = c.ws.Close()
	waitUntil(t, 2*time.Second, "disconnect callback", func() bool { return called.Load() == 1 })
	waitDone(t, fx.sess.Done(), "session closed by callback")
	select {
	case <-fx.task.Done():
		t.Fatal("pipeline cancelled although OnClientDisconnected was set")
	case <-time.After(100 * time.Millisecond):
	}
	if fx.tap.down.count("ClientDisconnectedFrame") != 1 {
		t.Fatal("ClientDisconnectedFrame not pushed")
	}
}

type idAnalyzer struct {
	id     string
	frames atomic.Int32
}

func (a *idAnalyzer) AnalyzeAudio(f *frames.AudioFrame, emit func(frames.Frame)) {
	if a.frames.Add(1) == 1 {
		emit(frames.NewInputTransportMessageFrame(a.id))
	}
}

type countingFilter struct {
	starts atomic.Int32
	calls  atomic.Int32
}

func (f *countingFilter) Start(int) error                 { f.starts.Add(1); return nil }
func (f *countingFilter) Stop() error                     { return nil }
func (f *countingFilter) ProcessFrame(frames.Frame) error { return nil }
func (f *countingFilter) Filter(dst, in []byte) []byte {
	f.calls.Add(1)
	return append(dst, in...)
}

type offsetMixer struct {
	offset  int16
	starts  atomic.Int32
	rate    atomic.Int32
	mixes   atomic.Int32
	enabled atomic.Bool
	updates atomic.Int32
}

func newOffsetMixer(offset int16) *offsetMixer {
	m := &offsetMixer{offset: offset}
	m.enabled.Store(true)
	return m
}

func (m *offsetMixer) Start(rate int) error {
	m.starts.Add(1)
	m.rate.Store(int32(rate))
	return nil
}

func (m *offsetMixer) Stop() error { return nil }

func (m *offsetMixer) ProcessFrame(f frames.Frame) error {
	if e, ok := f.(*frames.MixerEnableFrame); ok {
		m.enabled.Store(e.Enable)
	}
	m.updates.Add(1)
	return nil
}

func (m *offsetMixer) Mix(dst, in []byte) []byte {
	m.mixes.Add(1)
	off := m.offset
	if !m.enabled.Load() {
		off = 0
	}
	for i := 0; i+1 < len(in); i += 2 {
		dst = binary.LittleEndian.AppendUint16(dst, uint16(int16(binary.LittleEndian.Uint16(in[i:]))+off))
	}
	return dst
}

var _ mixer.Mixer = (*offsetMixer)(nil)
var _ filter.Filter = (*countingFilter)(nil)

func TestPerSessionFactoriesInvokedOncePerSession(t *testing.T) {
	var analyzers, mixers, filters atomic.Int32
	var mu sync.Mutex
	var mixList []*offsetMixer
	var filterList []*countingFilter
	params := TransportParams{
		NewAudioInAnalyzer: func() AudioInputAnalyzer {
			return &idAnalyzer{id: fmt.Sprintf("analyzer-%d", analyzers.Add(1))}
		},
		NewOutputMixer: func() mixer.Mixer {
			mixers.Add(1)
			m := newOffsetMixer(0)
			mu.Lock()
			mixList = append(mixList, m)
			mu.Unlock()
			return m
		},
		NewInputFilter: func() filter.Filter {
			filters.Add(1)
			f := &countingFilter{}
			mu.Lock()
			filterList = append(filterList, f)
			mu.Unlock()
			return f
		},
	}
	ts := startTwilioServer(t, params, serializers.TwilioConfig{})
	a := dialTwilio(t, ts.url, "MZ-fa", "CA-fa")
	fa := ts.next(t)
	b := dialTwilio(t, ts.url, "MZ-fb", "CA-fb")
	fb := ts.next(t)
	for i := 0; i < 3; i++ {
		a.sendMedia(t, "MZ-fa", ulawOf(t, constPCM(20, 8000, 500)))
		b.sendMedia(t, "MZ-fb", ulawOf(t, constPCM(20, 8000, 500)))
	}
	waitUntil(t, 2*time.Second, "audio in both sessions", func() bool {
		return len(logged[*frames.AudioFrame](&fa.tap.down)) == 3 && len(logged[*frames.AudioFrame](&fb.tap.down)) == 3
	})
	if analyzers.Load() != 2 || mixers.Load() != 2 || filters.Load() != 2 {
		t.Fatalf("factory calls analyzer=%d mixer=%d filter=%d, want 2 each", analyzers.Load(), mixers.Load(), filters.Load())
	}
	ids := map[string]bool{}
	for _, fx := range []*callFixture{fa, fb} {
		msgs := logged[*frames.InputTransportMessageFrame](&fx.tap.down)
		if len(msgs) != 1 {
			t.Fatalf("session saw %d analyzer markers, want exactly its own", len(msgs))
		}
		ids[msgs[0].Message.(string)] = true
	}
	if len(ids) != 2 {
		t.Fatalf("both sessions were fed by the same analyzer: %v", ids)
	}
	mu.Lock()
	defer mu.Unlock()
	for i, m := range mixList {
		if m.starts.Load() != 1 || m.rate.Load() != 8000 {
			t.Fatalf("mixer %d started %d times at %d Hz", i, m.starts.Load(), m.rate.Load())
		}
	}
	for i, f := range filterList {
		if f.starts.Load() != 1 || f.calls.Load() != 3 {
			t.Fatalf("filter %d starts=%d calls=%d, want 1 and 3", i, f.starts.Load(), f.calls.Load())
		}
	}
}

func TestMixerKeepsSendingDuringSilence(t *testing.T) {
	m := newOffsetMixer(4000)
	ts := startTwilioServer(t, TransportParams{NewOutputMixer: func() mixer.Mixer { return m }}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-m", "CA-m")
	fx := ts.next(t)
	waitUntil(t, 2*time.Second, "background audio", func() bool { return len(c.media()) >= 3 })
	first := len(c.media())
	time.Sleep(300 * time.Millisecond)
	got := len(c.media()) - first
	if got < 8 || got > 25 {
		t.Fatalf("%d chunks in 300 ms of silence, want ~15 paced chunks", got)
	}
	for _, msg := range c.media()[:5] {
		if v := meanSample(t, msg.payload); v < 3500 || v > 4500 {
			t.Fatalf("silence chunk level = %.0f, want mixer output ~4000", v)
		}
	}
	if fx.tap.up.count("BotStartedSpeakingFrame") != 0 {
		t.Fatal("background mixer audio counted as bot speech")
	}

	if err := fx.task.QueueFrame(frames.NewMixerEnableFrame(false)); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "mixer disabled", func() bool { return !m.enabled.Load() })
	time.Sleep(60 * time.Millisecond)
	before := len(c.media())
	waitUntil(t, 2*time.Second, "more chunks", func() bool { return len(c.media()) >= before+3 })
	if v := meanSample(t, c.media()[before+1].payload); v < -100 || v > 100 {
		t.Fatalf("disabled mixer still adds audio (%.0f)", v)
	}
}

func TestNativeMulawPassthrough(t *testing.T) {
	ts := startTwilioServer(t, TransportParams{AudioInNative: true, AudioOutNative: true}, serializers.TwilioConfig{})
	c := dialTwilio(t, ts.url, "MZ-n", "CA-n")
	fx := ts.next(t)

	raw := make([]byte, 160)
	for i := range raw {
		raw[i] = byte(i)
	}
	c.sendMedia(t, "MZ-n", raw)
	waitUntil(t, 2*time.Second, "native inbound frame", func() bool { return len(logged[*frames.AudioFrame](&fx.tap.down)) == 1 })
	in := logged[*frames.AudioFrame](&fx.tap.down)[0]
	if codec, _ := in.Metadata()["codec"].(string); audio.NormalizeCodec(codec) != audio.CodecMulaw {
		t.Fatalf("inbound codec = %q, want mulaw", codec)
	}
	if in.SampleRate != 8000 || !bytes.Equal(in.Data, raw) {
		t.Fatalf("inbound native audio altered: %d Hz, equal=%v", in.SampleRate, bytes.Equal(in.Data, raw))
	}

	out := make([]byte, 320)
	for i := range out {
		out[i] = byte(255 - i%256)
	}
	f := frames.NewTTSAudioFrame(out, 8000, 1)
	f.SetMetadata("codec", audio.CodecMulaw)
	if err := fx.task.QueueFrame(f); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "native outbound chunks", func() bool { return len(c.media()) >= 2 })
	media := c.media()
	if !bytes.Equal(media[0].payload, out[:160]) || !bytes.Equal(media[1].payload, out[160:]) {
		t.Fatal("outbound mu-law was transcoded instead of passed through")
	}
}

func TestDialClientSessionExchangesAudio(t *testing.T) {
	received := make(chan []byte, 64)
	closed := make(chan struct{})
	up := websocket.Upgrader{}
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := up.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer close(closed)
		defer ws.Close()
		if err := ws.WriteMessage(websocket.BinaryMessage, constPCM(20, 8000, 1234)); err != nil {
			return
		}
		for {
			typ, data, err := ws.ReadMessage()
			if err != nil {
				return
			}
			if typ == websocket.BinaryMessage {
				received <- data
			}
		}
	}))
	defer remote.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	sess, err := Dial(ctx, ClientConfig{
		URL:        "ws" + strings.TrimPrefix(remote.URL, "http"),
		Serializer: &testSerializer{},
		Params:     TransportParams{AudioInSampleRate: 8000},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sess.Wait()
	defer sess.Close()
	tap := newTap()
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{sess.Input(), tap.p, sess.Output()}))
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		_ = task.Run(ctx)
	}()

	waitUntil(t, 2*time.Second, "audio from remote", func() bool { return len(logged[*frames.AudioFrame](&tap.down)) == 1 })
	if f := logged[*frames.AudioFrame](&tap.down)[0]; f.SampleRate != 8000 || !bytes.Equal(f.Data, constPCM(20, 8000, 1234)) {
		t.Fatalf("remote audio = %d bytes at %d Hz", len(f.Data), f.SampleRate)
	}
	if err := task.QueueFrame(pcmTTS(constPCM(100, 8000, 777), 8000)); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		select {
		case d := <-received:
			if !bytes.Equal(d, constPCM(20, 8000, 777)) {
				t.Fatalf("chunk %d = %d bytes, not the bot audio", i, len(d))
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("remote got %d of 5 chunks", i)
		}
	}
	if err := task.QueueFrame(frames.NewEndFrame()); err != nil {
		t.Fatal(err)
	}
	waitDone(t, runDone, "client pipeline end")
	waitDone(t, closed, "remote sees the close")
}

func TestDialRequiresSerializerAndReportsDialErrors(t *testing.T) {
	if _, err := Dial(context.Background(), ClientConfig{URL: "ws://127.0.0.1:1"}); err == nil {
		t.Fatal("Dial without serializer succeeded")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := Dial(ctx, ClientConfig{URL: "ws://127.0.0.1:1/none", Serializer: &testSerializer{}}); err == nil {
		t.Fatal("Dial to a closed port succeeded")
	}
}

func TestNewServerValidatesConfig(t *testing.T) {
	if _, err := NewServer(ServerConfig{Handler: func(context.Context, *Session) error { return nil }}); err == nil {
		t.Fatal("missing serializer accepted")
	}
	if _, err := NewServer(ServerConfig{NewSerializer: func(*http.Request) serializers.FrameSerializer { return &testSerializer{} }}); err == nil {
		t.Fatal("missing handler accepted")
	}
}

func TestServerRejectsUpgradesAfterShutdown(t *testing.T) {
	srv, err := NewServer(ServerConfig{
		NewSerializer: func(*http.Request) serializers.FrameSerializer { return &testSerializer{} },
		Handler:       func(context.Context, *Session) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	hs := httptest.NewServer(srv)
	defer hs.Close()
	srv.Shutdown()
	_, resp, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(hs.URL, "http"), nil)
	if err == nil {
		t.Fatal("upgrade accepted after shutdown")
	}
	if resp == nil || resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %v, want 503", resp)
	}
}

func TestServerHandlerPanicClosesSession(t *testing.T) {
	srv, err := NewServer(ServerConfig{
		NewSerializer: func(*http.Request) serializers.FrameSerializer { return &testSerializer{} },
		Handler:       func(context.Context, *Session) error { panic("boom") },
	})
	if err != nil {
		t.Fatal(err)
	}
	hs := httptest.NewServer(srv)
	defer hs.Close()
	defer srv.Shutdown()
	ws, _, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(hs.URL, "http"), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	_ = ws.SetReadDeadline(time.Now().Add(3 * time.Second))
	if _, _, err := ws.ReadMessage(); err == nil || !websocket.IsCloseError(err, websocket.CloseNormalClosure) {
		t.Fatalf("expected a normal close after handler panic, got %v", err)
	}
}

func TestDialDroppedConnectionEndsSessionAndTask(t *testing.T) {
	drop := make(chan struct{})
	up := websocket.Upgrader{}
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := up.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		go func() {
			for {
				if _, _, err := ws.ReadMessage(); err != nil {
					return
				}
			}
		}()
		<-drop
		_ = ws.UnderlyingConn().Close()
	}))
	defer remote.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	sess, err := Dial(ctx, ClientConfig{
		URL:        "ws" + strings.TrimPrefix(remote.URL, "http"),
		Serializer: &testSerializer{},
		Params:     TransportParams{AudioInSampleRate: 8000},
	})
	if err != nil {
		t.Fatal(err)
	}
	tap := newTap()
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{sess.Input(), tap.p, sess.Output()}))
	runDone := make(chan struct{})
	go func() {
		defer close(runDone)
		_ = task.Run(ctx)
	}()
	if err := task.QueueFrame(pcmTTS(constPCM(2000, 8000, 777), 8000)); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, 2*time.Second, "bot audio flowing", func() bool { return tap.up.count("BotStartedSpeakingFrame") == 1 })

	close(drop)
	waitDone(t, runDone, "pipeline task to finish after the connection dropped")
	waitDone(t, sess.Done(), "session to close after the connection dropped")
	waited := make(chan struct{})
	go func() {
		sess.Wait()
		close(waited)
	}()
	waitDone(t, waited, "session goroutines to exit")
	if tap.down.count("ClientDisconnectedFrame") != 1 {
		t.Fatalf("ClientDisconnectedFrame count = %d, want 1", tap.down.count("ClientDisconnectedFrame"))
	}
	if ctx.Err() != nil {
		t.Fatal("task only finished because the test context expired")
	}
}

func dialPlain(t *testing.T, hs *httptest.Server) (*websocket.Conn, *http.Response, error) {
	t.Helper()
	ws, resp, err := websocket.DefaultDialer.Dial("ws"+strings.TrimPrefix(hs.URL, "http"), nil)
	if err == nil {
		t.Cleanup(func() { _ = ws.Close() })
	}
	return ws, resp, err
}

func runSessionPipeline(ctx context.Context, s *Session) error {
	return pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{s.Input(), s.Output()})).Run(ctx)
}

func TestServerShutdownContextDrainsActiveSessions(t *testing.T) {
	started := make(chan struct{}, 1)
	var cancelledWhileDraining atomic.Bool
	srv, err := NewServer(ServerConfig{
		NewSerializer: func(*http.Request) serializers.FrameSerializer { return &testSerializer{} },
		Handler: func(ctx context.Context, s *Session) error {
			started <- struct{}{}
			err := runSessionPipeline(ctx, s)
			if ctx.Err() != nil {
				cancelledWhileDraining.Store(true)
			}
			return err
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	hs := httptest.NewServer(srv)
	defer hs.Close()
	ws, _, err := dialPlain(t, hs)
	if err != nil {
		t.Fatal(err)
	}
	waitDone(t, started, "session handler")

	result := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		result <- srv.ShutdownContext(ctx)
	}()
	waitUntil(t, time.Second, "new upgrades rejected while draining", func() bool {
		_, resp, err := dialPlain(t, hs)
		return err != nil && resp != nil && resp.StatusCode == http.StatusServiceUnavailable
	})
	select {
	case err := <-result:
		t.Fatalf("ShutdownContext returned %v while a session was still active", err)
	case <-time.After(100 * time.Millisecond):
	}
	if srv.ActiveSessions() != 1 {
		t.Fatalf("active sessions = %d while draining, want 1", srv.ActiveSessions())
	}

	_ = ws.Close()
	select {
	case err := <-result:
		if err != nil {
			t.Fatalf("ShutdownContext = %v after the session ended on its own", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ShutdownContext did not return after the last session ended")
	}
	if srv.ActiveSessions() != 0 {
		t.Fatalf("active sessions = %d after drain", srv.ActiveSessions())
	}
	if cancelledWhileDraining.Load() {
		t.Fatal("draining cancelled the handler context instead of letting the call finish")
	}
}

func TestServerShutdownContextForcesSessionsAtDeadline(t *testing.T) {
	started := make(chan struct{}, 2)
	stuck := make(chan struct{})
	defer close(stuck)
	var calls atomic.Int32
	srv, err := NewServer(ServerConfig{
		NewSerializer: func(*http.Request) serializers.FrameSerializer { return &testSerializer{} },
		Params:        TransportParams{CloseTimeout: 100 * time.Millisecond},
		Handler: func(ctx context.Context, s *Session) error {
			n := calls.Add(1)
			started <- struct{}{}
			if n == 1 {
				return runSessionPipeline(ctx, s)
			}
			<-stuck
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	hs := httptest.NewServer(srv)
	defer hs.Close()
	first, _, err := dialPlain(t, hs)
	if err != nil {
		t.Fatal(err)
	}
	waitDone(t, started, "first session handler")
	if _, _, err := dialPlain(t, hs); err != nil {
		t.Fatal(err)
	}
	waitDone(t, started, "second session handler")

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	err = srv.ShutdownContext(ctx)
	elapsed := time.Since(start)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("ShutdownContext = %v, want deadline exceeded", err)
	}
	if elapsed < 140*time.Millisecond || elapsed > time.Second {
		t.Fatalf("ShutdownContext took %v, want the 150 ms drain bound plus at most CloseTimeout", elapsed)
	}
	_ = first.SetReadDeadline(time.Now().Add(2 * time.Second))
	if _, _, err := first.ReadMessage(); err == nil {
		t.Fatal("active session was not closed at the drain deadline")
	}
	if n := srv.ActiveSessions(); n != 1 {
		t.Fatalf("active sessions = %d, want only the stuck handler", n)
	}
}
