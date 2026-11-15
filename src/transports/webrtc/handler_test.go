package webrtc

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	pwebrtc "github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

func loopbackSettings() *pwebrtc.SettingEngine {
	se := &pwebrtc.SettingEngine{}
	se.SetNetworkTypes([]pwebrtc.NetworkType{pwebrtc.NetworkTypeUDP4})
	se.SetIncludeLoopbackCandidate(true)
	se.SetIPFilter(func(ip net.IP) bool { return ip.IsLoopback() })
	return se
}

type offerer struct {
	pc    *pwebrtc.PeerConnection
	track *pwebrtc.TrackLocalStaticSample
	dc    *pwebrtc.DataChannel
}

func newOfferer(t *testing.T, caps ...pwebrtc.RTPCodecCapability) *offerer {
	t.Helper()
	m := &pwebrtc.MediaEngine{}
	for i, c := range caps {
		pt := pwebrtc.PayloadType(0)
		switch c.MimeType {
		case pwebrtc.MimeTypePCMA:
			pt = 8
		case pwebrtc.MimeTypePCMU:
			pt = 0
		default:
			pt = pwebrtc.PayloadType(111 + i)
		}
		if err := m.RegisterCodec(pwebrtc.RTPCodecParameters{RTPCodecCapability: c, PayloadType: pt}, pwebrtc.RTPCodecTypeAudio); err != nil {
			t.Fatal(err)
		}
	}
	pc, err := pwebrtc.NewAPI(pwebrtc.WithMediaEngine(m), pwebrtc.WithSettingEngine(*loopbackSettings())).NewPeerConnection(pwebrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pc.Close() })
	track, err := pwebrtc.NewTrackLocalStaticSample(caps[0], "audio", "client")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pc.AddTrack(track); err != nil {
		t.Fatal(err)
	}
	dc, err := pc.CreateDataChannel("chat", nil)
	if err != nil {
		t.Fatal(err)
	}
	return &offerer{pc: pc, track: track, dc: dc}
}

func (o *offerer) offer(t *testing.T, restart bool) string {
	t.Helper()
	var opts *pwebrtc.OfferOptions
	if restart {
		opts = &pwebrtc.OfferOptions{ICERestart: true}
	}
	offer, err := o.pc.CreateOffer(opts)
	if err != nil {
		t.Fatal(err)
	}
	gathered := pwebrtc.GatheringCompletePromise(o.pc)
	if err := o.pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	select {
	case <-gathered:
	case <-time.After(5 * time.Second):
		t.Fatal("client ICE gathering timed out")
	}
	return o.pc.LocalDescription().SDP
}

type sigServer struct {
	h   *Handler
	url string
}

func newSigServer(t *testing.T, cfg Config) *sigServer {
	t.Helper()
	if cfg.Handler == nil {
		cfg.Handler = func(ctx context.Context, s *Session) error {
			select {
			case <-ctx.Done():
			case <-s.Done():
			}
			return nil
		}
	}
	if cfg.SettingEngine == nil {
		cfg.SettingEngine = loopbackSettings()
	}
	cfg.GatherTimeout = 3 * time.Second
	h, err := NewHandler(cfg)
	if err != nil {
		t.Fatal(err)
	}
	hs := httptest.NewServer(h)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := h.Shutdown(ctx); err != nil {
			t.Errorf("handler shutdown: %v", err)
		}
		hs.Close()
	})
	return &sigServer{h: h, url: hs.URL}
}

func (s *sigServer) do(t *testing.T, method string, body any) (int, []byte) {
	t.Helper()
	var buf bytes.Buffer
	switch v := body.(type) {
	case string:
		buf.WriteString(v)
	default:
		if err := json.NewEncoder(&buf).Encode(v); err != nil {
			t.Fatal(err)
		}
	}
	req, err := http.NewRequest(method, s.url, &buf)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out bytes.Buffer
	_, _ = out.ReadFrom(resp.Body)
	return resp.StatusCode, out.Bytes()
}

func (s *sigServer) postOffer(t *testing.T, req map[string]any) Answer {
	t.Helper()
	status, body := s.do(t, http.MethodPost, req)
	if status != http.StatusOK {
		t.Fatalf("POST offer status %d: %s", status, body)
	}
	var ans Answer
	if err := json.Unmarshal(body, &ans); err != nil {
		t.Fatalf("answer is not JSON: %v (%s)", err, body)
	}
	return ans
}

var pcmuCap = pwebrtc.RTPCodecCapability{MimeType: pwebrtc.MimeTypePCMU, ClockRate: 8000, Channels: 1}
var pcmaCap = pwebrtc.RTPCodecCapability{MimeType: pwebrtc.MimeTypePCMA, ClockRate: 8000, Channels: 1}

func TestOfferReturnsAnswerWithPCID(t *testing.T) {
	sessions := make(chan *Session, 1)
	srv := newSigServer(t, Config{Handler: func(ctx context.Context, s *Session) error {
		sessions <- s
		<-ctx.Done()
		return nil
	}})
	o := newOfferer(t, pcmuCap)
	ans := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer", "request_data": map[string]string{"user": "u1"}})
	if ans.Type != "answer" || !strings.HasPrefix(ans.PCID, "pc-") || !strings.Contains(ans.SDP, "m=audio") {
		t.Fatalf("answer = type %q pc_id %q", ans.Type, ans.PCID)
	}
	if !strings.Contains(ans.SDP, "PCMU/8000") {
		t.Fatal("answer does not select PCMU")
	}
	if err := o.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: ans.SDP}); err != nil {
		t.Fatalf("client rejects answer: %v", err)
	}
	select {
	case s := <-sessions:
		if s.ID() != ans.PCID || srv.h.Session(ans.PCID) != s {
			t.Fatalf("session id %q, pc_id %q", s.ID(), ans.PCID)
		}
		if s.Codec() != PCMU {
			t.Fatalf("session codec = %v", s.Codec().Capability().MimeType)
		}
		var data map[string]string
		if err := json.Unmarshal(s.RequestData(), &data); err != nil || data["user"] != "u1" {
			t.Fatalf("request data = %s", s.RequestData())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("session handler not invoked")
	}
}

func TestOfferPrefersConfiguredCodecOrder(t *testing.T) {
	srv := newSigServer(t, Config{Codecs: []Codec{PCMA, PCMU}})
	o := newOfferer(t, pcmuCap, pcmaCap)
	ans := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	if got := srv.h.Session(ans.PCID).Codec(); got != PCMA {
		t.Fatalf("codec = %s, want PCMA", got.Capability().MimeType)
	}
}

func TestOfferWithoutSupportedCodecIs400(t *testing.T) {
	srv := newSigServer(t, Config{Codecs: []Codec{PCMA}})
	o := newOfferer(t, pcmuCap)
	status, body := srv.do(t, http.MethodPost, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	if status != http.StatusBadRequest || !strings.Contains(string(body), "codec") {
		t.Fatalf("status %d body %s", status, body)
	}
}

func TestBadRequestsRejected(t *testing.T) {
	srv := newSigServer(t, Config{})
	for _, tc := range []struct {
		name   string
		method string
		body   any
		want   int
	}{
		{"wrong type", http.MethodPost, map[string]any{"sdp": "v=0", "type": "answer"}, http.StatusBadRequest},
		{"invalid json", http.MethodPost, "{", http.StatusBadRequest},
		{"invalid sdp", http.MethodPost, map[string]any{"sdp": "garbage", "type": "offer"}, http.StatusBadRequest},
		{"patch unknown pc", http.MethodPatch, map[string]any{"pc_id": "pc-missing", "candidates": []any{}}, http.StatusNotFound},
		{"patch invalid json", http.MethodPatch, "nope", http.StatusBadRequest},
		{"get", http.MethodGet, "", http.StatusMethodNotAllowed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if status, body := srv.do(t, tc.method, tc.body); status != tc.want {
				t.Fatalf("status %d, want %d (%s)", status, tc.want, body)
			}
		})
	}
}

func TestPatchCandidatesForKnownPC(t *testing.T) {
	srv := newSigServer(t, Config{})
	o := newOfferer(t, pcmuCap)
	ans := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	cand := map[string]any{"candidate": "candidate:1 1 udp 2130706431 127.0.0.1 50000 typ host", "sdp_mid": "0", "sdp_mline_index": 0}
	status, body := srv.do(t, http.MethodPatch, map[string]any{"pc_id": ans.PCID, "candidates": []any{cand}})
	if status != http.StatusOK || !strings.Contains(string(body), `"success"`) {
		t.Fatalf("PATCH status %d: %s", status, body)
	}
}

func TestRenegotiateAndRestartKeepPCID(t *testing.T) {
	var mu sync.Mutex
	starts := 0
	srv := newSigServer(t, Config{Handler: func(ctx context.Context, s *Session) error {
		mu.Lock()
		starts++
		mu.Unlock()
		select {
		case <-ctx.Done():
		case <-s.Done():
		}
		return nil
	}})
	o := newOfferer(t, pcmuCap)
	first := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	if err := o.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: first.SDP}); err != nil {
		t.Fatal(err)
	}
	sess := srv.h.Session(first.PCID)

	re := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer", "pc_id": first.PCID})
	if re.PCID != first.PCID || re.Type != "answer" {
		t.Fatalf("renegotiation answer pc_id %q type %q", re.PCID, re.Type)
	}
	if err := o.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: re.SDP}); err != nil {
		t.Fatal(err)
	}

	o2 := newOfferer(t, pcmuCap)
	restarted := srv.postOffer(t, map[string]any{"sdp": o2.offer(t, false), "type": "offer", "pc_id": first.PCID, "restart_pc": true})
	if restarted.PCID != first.PCID || srv.h.Session(first.PCID) != sess {
		t.Fatalf("restart created a new session: %q", restarted.PCID)
	}
	if err := o2.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: restarted.SDP}); err != nil {
		t.Fatalf("restart answer rejected: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if starts != 1 {
		t.Fatalf("session handler ran %d times, want 1", starts)
	}
}

func TestSingleConnectionModeRejectsSecondPeer(t *testing.T) {
	srv := newSigServer(t, Config{ConnectionMode: SingleConnection})
	o := newOfferer(t, pcmuCap)
	first := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	if err := o.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: first.SDP}); err != nil {
		t.Fatal(err)
	}
	otherSDP := newOfferer(t, pcmuCap).offer(t, false)
	status, _ := srv.do(t, http.MethodPost, map[string]any{"sdp": otherSDP, "type": "offer"})
	if status != http.StatusBadRequest {
		t.Fatalf("second connection status %d, want 400", status)
	}
	status, _ = srv.do(t, http.MethodPost, map[string]any{"sdp": otherSDP, "type": "offer", "pc_id": "pc-other"})
	if status != http.StatusBadRequest {
		t.Fatalf("mismatched pc_id status %d, want 400", status)
	}
	again := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer", "pc_id": first.PCID})
	if again.PCID != first.PCID {
		t.Fatalf("pc_id changed to %q", again.PCID)
	}
}

func TestShutdownRejectsNewOffers(t *testing.T) {
	srv := newSigServer(t, Config{})
	o := newOfferer(t, pcmuCap)
	ans := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.h.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if srv.h.Session(ans.PCID) != nil {
		t.Fatal("session still registered after shutdown")
	}
	status, _ := srv.do(t, http.MethodPost, map[string]any{"sdp": newOfferer(t, pcmuCap).offer(t, false), "type": "offer"})
	if status != http.StatusServiceUnavailable {
		t.Fatalf("offer after shutdown status %d, want 503", status)
	}
}

func TestNewHandlerRequiresSessionHandler(t *testing.T) {
	if _, err := NewHandler(Config{}); err == nil {
		t.Fatal("handler without SessionHandler accepted")
	}
}

func sine(n int, freq, rate, amp float64) []byte {
	out := make([]byte, 0, 2*n)
	for i := 0; i < n; i++ {
		v := int16(amp * math.Sin(2*math.Pi*freq*float64(i)/rate))
		out = binary.LittleEndian.AppendUint16(out, uint16(v))
	}
	return out
}

func TestG711CodecsRoundTrip(t *testing.T) {
	pcm := sine(160, 440, 8000, 12000)
	for _, tc := range []struct {
		name  string
		codec Codec
		mime  string
		pt    pwebrtc.PayloadType
	}{{"PCMU", PCMU, pwebrtc.MimeTypePCMU, 0}, {"PCMA", PCMA, pwebrtc.MimeTypePCMA, 8}} {
		t.Run(tc.name, func(t *testing.T) {
			if c := tc.codec.Capability(); c.MimeType != tc.mime || c.ClockRate != 8000 || c.Channels != 1 {
				t.Fatalf("capability = %+v", c)
			}
			if tc.codec.PayloadType() != tc.pt || tc.codec.SampleRate() != 8000 {
				t.Fatalf("payload type %d rate %d", tc.codec.PayloadType(), tc.codec.SampleRate())
			}
			enc, err := tc.codec.NewEncoder()
			if err != nil {
				t.Fatal(err)
			}
			dec, err := tc.codec.NewDecoder()
			if err != nil {
				t.Fatal(err)
			}
			payload, err := enc.Encode(nil, pcm)
			if err != nil {
				t.Fatal(err)
			}
			if len(payload) != 160 {
				t.Fatalf("payload = %d bytes, want 1 byte per sample", len(payload))
			}
			back, err := dec.Decode(nil, payload)
			if err != nil {
				t.Fatal(err)
			}
			if len(back) != len(pcm) {
				t.Fatalf("decoded %d bytes, want %d", len(back), len(pcm))
			}
			for i := 0; i < len(pcm); i += 2 {
				a := float64(int16(binary.LittleEndian.Uint16(pcm[i:])))
				b := float64(int16(binary.LittleEndian.Uint16(back[i:])))
				if math.Abs(a-b) > math.Max(16, math.Abs(a)*0.07) {
					t.Fatalf("sample %d: %v -> %v exceeds G.711 quantisation error", i/2, a, b)
				}
			}
			silence, _ := enc.Encode(nil, make([]byte, 320))
			zero, _ := dec.Decode(nil, silence)
			for i := 0; i < len(zero); i += 2 {
				if v := int16(binary.LittleEndian.Uint16(zero[i:])); v > 8 || v < -8 {
					t.Fatalf("silence decoded to %d", v)
				}
			}
		})
	}
}

type tap struct {
	p  *processors.BaseProcessor
	mu sync.Mutex
	in []*frames.AudioFrame
	cc int
}

func (h *tap) HandleFrame(ctx context.Context, f frames.Frame, dir frames.FrameDirection) error {
	h.mu.Lock()
	switch v := f.(type) {
	case *frames.AudioFrame:
		h.in = append(h.in, v)
	case *frames.ClientConnectedFrame:
		h.cc++
	}
	h.mu.Unlock()
	return h.p.PushFrame(f, dir)
}

func (h *tap) counts() (int, int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.in), h.cc
}

func TestLoopbackAudioBothWays(t *testing.T) {
	type call struct {
		tap  *tap
		task *pipeline.PipelineTask
	}
	calls := make(chan call, 1)
	srv := newSigServer(t, Config{Params: Params{AudioInSampleRate: 16000}, Handler: func(ctx context.Context, s *Session) error {
		tp := &tap{}
		tp.p = processors.NewBaseProcessor("tap", tp)
		task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{s.Input(), tp.p, s.Output()}))
		calls <- call{tap: tp, task: task}
		return task.Run(ctx)
	}})
	o := newOfferer(t, pcmuCap)
	gotRTP := make(chan float64, 256)
	o.pc.OnTrack(func(tr *pwebrtc.TrackRemote, _ *pwebrtc.RTPReceiver) {
		for {
			pkt, _, err := tr.ReadRTP()
			if err != nil {
				return
			}
			pcm, _ := PCMU.(Decoder).Decode(nil, pkt.Payload)
			select {
			case gotRTP <- rms(pcm):
			default:
			}
		}
	})
	connected := make(chan struct{})
	var once sync.Once
	o.pc.OnConnectionStateChange(func(s pwebrtc.PeerConnectionState) {
		if s == pwebrtc.PeerConnectionStateConnected {
			once.Do(func() { close(connected) })
		}
	})
	ans := srv.postOffer(t, map[string]any{"sdp": o.offer(t, false), "type": "offer"})
	if err := o.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeAnswer, SDP: ans.SDP}); err != nil {
		t.Fatal(err)
	}
	var c call
	select {
	case c = <-calls:
	case <-time.After(3 * time.Second):
		t.Fatal("no session")
	}
	select {
	case <-connected:
	case <-time.After(5 * time.Second):
		t.Fatal("ICE did not connect over loopback")
	}
	deadline := time.Now().Add(4 * time.Second)
	for {
		if _, cc := c.tap.counts(); cc == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("ClientConnectedFrame not pushed")
		}
		time.Sleep(10 * time.Millisecond)
	}

	payload, _ := PCMU.(Encoder).Encode(nil, sine(160, 440, 8000, 8000))
	for {
		if err := o.track.WriteSample(mediaSample(payload)); err != nil {
			t.Fatal(err)
		}
		if n, _ := c.tap.counts(); n >= 3 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("no inbound audio frames reached the pipeline")
		}
		time.Sleep(20 * time.Millisecond)
	}
	c.tap.mu.Lock()
	first := c.tap.in[0]
	c.tap.mu.Unlock()
	if first.SampleRate != 16000 {
		t.Fatalf("inbound rate %d, want 16000", first.SampleRate)
	}

	tts := frames.NewTTSAudioFrame(sine(1600, 300, 8000, 8000), 8000, 1)
	if err := c.task.QueueFrame(tts); err != nil {
		t.Fatal(err)
	}
	got := 0
	for got < 3 {
		select {
		case level := <-gotRTP:
			if level > 3000 {
				got++
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("client received %d voiced outbound RTP packets", got)
		}
	}
}

func mediaSample(data []byte) media.Sample {
	return media.Sample{Data: data, Duration: 20 * time.Millisecond}
}

func rms(pcm []byte) float64 {
	if len(pcm) < 2 {
		return 0
	}
	sum := 0.0
	for i := 0; i+1 < len(pcm); i += 2 {
		v := float64(int16(binary.LittleEndian.Uint16(pcm[i:])))
		sum += v * v
	}
	return math.Sqrt(sum / float64(len(pcm)/2))
}

type recordedChunk struct {
	pcm []byte
	at  time.Time
}

type recordingCodec struct {
	mu     sync.Mutex
	chunks []recordedChunk
}

func (c *recordingCodec) Capability() pwebrtc.RTPCodecCapability { return PCMU.Capability() }
func (c *recordingCodec) PayloadType() pwebrtc.PayloadType       { return 0 }
func (c *recordingCodec) SampleRate() int                        { return 8000 }
func (c *recordingCodec) NewEncoder() (Encoder, error)           { return c, nil }
func (c *recordingCodec) NewDecoder() (Decoder, error)           { return PCMU.NewDecoder() }

func (c *recordingCodec) Encode(dst, pcm []byte) ([]byte, error) {
	c.mu.Lock()
	c.chunks = append(c.chunks, recordedChunk{pcm: bytes.Clone(pcm), at: time.Now()})
	c.mu.Unlock()
	return PCMU.(Encoder).Encode(dst, pcm)
}

func (c *recordingCodec) loud() []recordedChunk {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []recordedChunk
	for _, ch := range c.chunks {
		if !bytes.Equal(ch.pcm, make([]byte, len(ch.pcm))) {
			out = append(out, ch)
		}
	}
	return out
}

type upstreamNames struct {
	mu    sync.Mutex
	names []string
}

func (u *upstreamNames) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}
func (u *upstreamNames) QueueFrame(f frames.Frame, _ frames.FrameDirection) error {
	u.mu.Lock()
	u.names = append(u.names, f.Name())
	u.mu.Unlock()
	return nil
}
func (u *upstreamNames) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (u *upstreamNames) Link(processors.FrameProcessor)                      {}
func (u *upstreamNames) SetPrev(processors.FrameProcessor)                   {}
func (u *upstreamNames) Start(context.Context) error                         { return nil }
func (u *upstreamNames) Stop() error                                         { return nil }
func (u *upstreamNames) Name() string                                        { return "upstream-names" }

func (u *upstreamNames) count(name string) int {
	u.mu.Lock()
	defer u.mu.Unlock()
	n := 0
	for _, s := range u.names {
		if s == name {
			n++
		}
	}
	return n
}

func runFadeOutput(t *testing.T, fade time.Duration) (*recordingCodec, *upstreamNames, time.Time) {
	t.Helper()
	codec := &recordingCodec{}
	track, err := pwebrtc.NewTrackLocalStaticSample(codec.Capability(), "audio", "fade")
	if err != nil {
		t.Fatal(err)
	}
	s := &Session{id: "fade", params: Params{InterruptionFade: fade, DisableAudioOutSilence: true}.withDefaults(), codec: codec, track: track, log: logger.WithPrefix("fade-test")}
	out := newOutputProcessor(s)
	up := &upstreamNames{}
	out.SetPrev(up)
	if err := out.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = out.Stop() })
	if err := out.QueueFrame(frames.NewStartFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	pcm := make([]byte, 0, 16000)
	for i := 0; i < 8000; i++ {
		pcm = binary.LittleEndian.AppendUint16(pcm, 8000)
	}
	for i := 0; i < 2; i++ {
		f := frames.NewTTSAudioFrame(pcm, 8000, 1)
		f.SetMetadata("codec", "linear16")
		if err := out.QueueFrame(f, frames.Downstream); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(2 * time.Second)
	for len(codec.loud()) < 4 {
		if time.Now().After(deadline) {
			t.Fatal("playback never started")
		}
		time.Sleep(2 * time.Millisecond)
	}
	at := time.Now()
	if err := out.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	return codec, up, at
}

func TestWebRTCInterruptionFadeRampsToSilence(t *testing.T) {
	const fade = 60 * time.Millisecond
	codec, up, at := runFadeOutput(t, fade)
	time.Sleep(fade / 2)
	if up.count("BotStoppedSpeakingFrame") != 0 {
		t.Fatal("BotStoppedSpeakingFrame emitted before the faded tail played")
	}
	deadline := time.Now().Add(time.Second)
	for up.count("BotStoppedSpeakingFrame") == 0 {
		if time.Now().After(deadline) {
			t.Fatal("no BotStoppedSpeakingFrame after the fade")
		}
		time.Sleep(2 * time.Millisecond)
	}
	time.Sleep(80 * time.Millisecond)

	var tail []int
	var last time.Time
	for _, ch := range codec.loud() {
		last = ch.at
		if len(tail) == 0 && int16(binary.LittleEndian.Uint16(ch.pcm[len(ch.pcm)-2:])) == 8000 {
			continue
		}
		for i := 0; i+1 < len(ch.pcm); i += 2 {
			tail = append(tail, int(int16(binary.LittleEndian.Uint16(ch.pcm[i:]))))
		}
	}
	if len(tail) == 0 || len(tail) > 480 {
		t.Fatalf("faded tail = %d samples, want 1..480", len(tail))
	}
	prev := 8000
	for i, v := range tail {
		if v < 0 || v > prev {
			t.Fatalf("envelope not monotonically decreasing at sample %d: %d after %d", i, v, prev)
		}
		prev = v
	}
	if prev != 0 || tail[0] < 4000 {
		t.Fatalf("fade runs from %d to %d, want full level down to 0", tail[0], prev)
	}
	if lat := last.Sub(at); lat > fade+20*time.Millisecond+30*time.Millisecond {
		t.Fatalf("last audio %v after the interruption, want at most fade + one chunk", lat)
	}
}

func TestWebRTCInterruptionWithoutFadeCutsInstantly(t *testing.T) {
	codec, up, at := runFadeOutput(t, 0)
	deadline := time.Now().Add(time.Second)
	for up.count("BotStoppedSpeakingFrame") == 0 {
		if time.Now().After(deadline) {
			t.Fatal("no BotStoppedSpeakingFrame after the interruption")
		}
		time.Sleep(2 * time.Millisecond)
	}
	time.Sleep(80 * time.Millisecond)
	for _, ch := range codec.loud() {
		if ch.at.After(at.Add(15 * time.Millisecond)) {
			t.Fatal("audio written after the interruption with fade disabled")
		}
		for i := 0; i+1 < len(ch.pcm); i += 2 {
			if int16(binary.LittleEndian.Uint16(ch.pcm[i:])) != 8000 {
				t.Fatal("audio attenuated with fade disabled")
			}
		}
	}
}
