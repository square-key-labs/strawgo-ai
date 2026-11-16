package webrtc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	pwebrtc "github.com/pion/webrtc/v4"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

var ErrSessionClosed = errors.New("webrtc: session closed")

type eventKind uint8

const (
	evPeer eventKind = iota
	evState
	evDataChannel
	evDCOpen
	evDCClose
	evMessage
)

type event struct {
	kind eventKind
	peer *peer
	dc   *pwebrtc.DataChannel
	msg  pwebrtc.DataChannelMessage
}

type outMsg struct {
	data   []byte
	binary bool
}

type peer struct {
	pc     *pwebrtc.PeerConnection
	ctx    context.Context
	cancel context.CancelFunc
}

func (p *peer) close() {
	p.cancel()
	_ = p.pc.Close()
}

type Session struct {
	id      string
	handler *Handler
	params  Params
	codec   Codec
	servers []pwebrtc.ICEServer
	request *http.Request
	data    json.RawMessage
	log     *logger.Logger
	track   *pwebrtc.TrackLocalStaticSample

	input  *InputProcessor
	output *OutputProcessor

	ctx    context.Context
	cancel context.CancelFunc

	goMu     sync.Mutex
	goClosed bool
	wg       sync.WaitGroup

	events    chan event
	outbox    chan outMsg
	sem       chan struct{}
	closeCh   chan struct{}
	closeOnce sync.Once
	loopDone  chan struct{}

	startedCh      chan struct{}
	disconnectedCh chan struct{}
	ready          atomic.Bool
	closing        atomic.Bool

	mu   sync.Mutex
	peer *peer
}

func newSession(h *Handler, id string, codec Codec, servers []pwebrtc.ICEServer, r *http.Request, data json.RawMessage) (*Session, error) {
	ctx, cancel := context.WithCancel(context.Background())
	s := &Session{
		id:             id,
		handler:        h,
		params:         h.params.forSession(),
		codec:          codec,
		servers:        servers,
		request:        r,
		data:           data,
		log:            logger.WithPrefix("WebRTCSession " + id),
		ctx:            ctx,
		cancel:         cancel,
		events:         make(chan event, 64),
		outbox:         make(chan outMsg, 64),
		sem:            make(chan struct{}, 1),
		closeCh:        make(chan struct{}),
		loopDone:       make(chan struct{}),
		startedCh:      make(chan struct{}),
		disconnectedCh: make(chan struct{}),
	}
	if codec != nil {
		track, err := pwebrtc.NewTrackLocalStaticSample(codec.Capability(), "audio", "strawgo-"+id)
		if err != nil {
			cancel()
			return nil, err
		}
		s.track = track
	}
	s.input = newInputProcessor(s)
	s.output = newOutputProcessor(s)
	s.wg.Add(1)
	go s.loop()
	return s, nil
}

func (s *Session) ID() string { return s.id }

func (s *Session) Input() *InputProcessor { return s.input }

func (s *Session) Output() *OutputProcessor { return s.output }

func (s *Session) Request() *http.Request { return s.request }

func (s *Session) RequestData() json.RawMessage { return s.data }

func (s *Session) Codec() Codec { return s.codec }

func (s *Session) Done() <-chan struct{} { return s.ctx.Done() }

func (s *Session) Disconnected() <-chan struct{} { return s.disconnectedCh }

func (s *Session) Close() error {
	s.closeOnce.Do(func() {
		s.closing.Store(true)
		close(s.closeCh)
	})
	return nil
}

func (s *Session) Wait() {
	<-s.loopDone
	s.goMu.Lock()
	s.goClosed = true
	s.goMu.Unlock()
	s.wg.Wait()
}

func (s *Session) SendMessage(ctx context.Context, msg any) error {
	m, err := encodeMessage(msg)
	if err != nil {
		return err
	}
	return s.enqueue(ctx, m)
}

func (s *Session) RequestRenegotiation(ctx context.Context) error {
	return s.enqueue(ctx, outMsg{data: renegotiateMessage})
}

func (s *Session) codecName() string {
	if s.codec == nil {
		return "none"
	}
	return s.codec.Capability().MimeType
}

func encodeMessage(msg any) (outMsg, error) {
	switch v := msg.(type) {
	case string:
		return outMsg{data: []byte(v)}, nil
	case []byte:
		return outMsg{data: v, binary: true}, nil
	case json.RawMessage:
		return outMsg{data: v}, nil
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return outMsg{}, err
	}
	return outMsg{data: data}, nil
}

func (s *Session) enqueue(ctx context.Context, m outMsg) error {
	select {
	case s.outbox <- m:
		return nil
	case <-s.ctx.Done():
		return ErrSessionClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Session) enter() bool {
	s.goMu.Lock()
	defer s.goMu.Unlock()
	if s.goClosed || s.ctx.Err() != nil {
		return false
	}
	s.wg.Add(1)
	return true
}

func (s *Session) currentPeer() *peer {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.peer
}

func (s *Session) post(p *peer, e event) {
	e.peer = p
	select {
	case s.events <- e:
	case <-p.ctx.Done():
	}
}

func (s *Session) acquire(ctx context.Context) error {
	select {
	case s.sem <- struct{}{}:
		return nil
	case <-s.ctx.Done():
		return ErrSessionClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Session) release() { <-s.sem }

func (s *Session) negotiate(ctx context.Context, offer string, restart bool) (pwebrtc.SessionDescription, error) {
	if err := s.acquire(ctx); err != nil {
		return pwebrtc.SessionDescription{}, err
	}
	defer s.release()
	p := s.currentPeer()
	fresh := p == nil || restart
	if fresh {
		np, err := s.newPeer()
		if err != nil {
			return pwebrtc.SessionDescription{}, err
		}
		s.mu.Lock()
		if s.ctx.Err() != nil {
			s.mu.Unlock()
			np.close()
			return pwebrtc.SessionDescription{}, ErrSessionClosed
		}
		old := s.peer
		s.peer = np
		s.mu.Unlock()
		if old != nil {
			s.log.Info("restarting peer connection")
			old.close()
		}
		s.post(np, event{kind: evPeer})
		p = np
	}
	if err := p.pc.SetRemoteDescription(pwebrtc.SessionDescription{Type: pwebrtc.SDPTypeOffer, SDP: offer}); err != nil {
		if p.ctx.Err() != nil {
			return pwebrtc.SessionDescription{}, ErrSessionClosed
		}
		return pwebrtc.SessionDescription{}, fmt.Errorf("%w: set remote description: %v", ErrBadRequest, err)
	}
	if fresh && s.track != nil {
		sender, err := p.pc.AddTrack(s.track)
		if err != nil {
			return pwebrtc.SessionDescription{}, fmt.Errorf("add track: %w", err)
		}
		if s.enter() {
			go func() {
				defer s.wg.Done()
				drainRTCP(sender)
			}()
		}
	}
	answer, err := p.pc.CreateAnswer(nil)
	if err != nil {
		return pwebrtc.SessionDescription{}, fmt.Errorf("create answer: %w", err)
	}
	gathered := pwebrtc.GatheringCompletePromise(p.pc)
	if err := p.pc.SetLocalDescription(answer); err != nil {
		return pwebrtc.SessionDescription{}, fmt.Errorf("set local description: %w", err)
	}
	timer := time.NewTimer(s.handler.cfg.GatherTimeout)
	defer timer.Stop()
	select {
	case <-gathered:
	case <-timer.C:
		s.log.Warn("ICE gathering timed out; answering with candidates gathered so far")
	case <-ctx.Done():
		return pwebrtc.SessionDescription{}, ctx.Err()
	case <-p.ctx.Done():
		return pwebrtc.SessionDescription{}, ErrSessionClosed
	}
	desc := p.pc.LocalDescription()
	if desc == nil {
		return pwebrtc.SessionDescription{}, errors.New("webrtc: no local description")
	}
	return *desc, nil
}

func (s *Session) addCandidates(ctx context.Context, cands []ICECandidate) error {
	if err := s.acquire(ctx); err != nil {
		return err
	}
	defer s.release()
	p := s.currentPeer()
	if p == nil {
		return ErrNotFound
	}
	for _, c := range cands {
		mid, idx := c.SDPMid, c.SDPMLineIndex
		if err := p.pc.AddICECandidate(pwebrtc.ICECandidateInit{Candidate: c.Candidate, SDPMid: &mid, SDPMLineIndex: &idx}); err != nil {
			return fmt.Errorf("%w: add ice candidate: %v", ErrBadRequest, err)
		}
	}
	return nil
}

func (s *Session) newPeer() (*peer, error) {
	m := &pwebrtc.MediaEngine{}
	if s.codec != nil {
		if err := m.RegisterCodec(pwebrtc.RTPCodecParameters{RTPCodecCapability: s.codec.Capability(), PayloadType: s.codec.PayloadType()}, pwebrtc.RTPCodecTypeAudio); err != nil {
			return nil, err
		}
	}
	for _, c := range videoCodecs {
		if err := m.RegisterCodec(c, pwebrtc.RTPCodecTypeVideo); err != nil {
			return nil, err
		}
	}
	reg := &interceptor.Registry{}
	if err := pwebrtc.RegisterDefaultInterceptors(m, reg); err != nil {
		return nil, err
	}
	opts := []func(*pwebrtc.API){pwebrtc.WithMediaEngine(m), pwebrtc.WithInterceptorRegistry(reg)}
	if se := s.handler.cfg.SettingEngine; se != nil {
		opts = append(opts, pwebrtc.WithSettingEngine(*se))
	}
	pc, err := pwebrtc.NewAPI(opts...).NewPeerConnection(pwebrtc.Configuration{
		ICEServers:   s.servers,
		BundlePolicy: pwebrtc.BundlePolicyMaxBundle,
	})
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(s.ctx)
	p := &peer{pc: pc, ctx: ctx, cancel: cancel}
	pc.OnConnectionStateChange(func(pwebrtc.PeerConnectionState) {
		s.post(p, event{kind: evState})
	})
	pc.OnDataChannel(func(dc *pwebrtc.DataChannel) {
		dc.OnOpen(func() { s.post(p, event{kind: evDCOpen, dc: dc}) })
		dc.OnClose(func() { s.post(p, event{kind: evDCClose, dc: dc}) })
		dc.OnMessage(func(m pwebrtc.DataChannelMessage) { s.post(p, event{kind: evMessage, msg: m}) })
		s.post(p, event{kind: evDataChannel, dc: dc})
	})
	pc.OnTrack(func(t *pwebrtc.TrackRemote, _ *pwebrtc.RTPReceiver) {
		if !s.enter() {
			return
		}
		defer s.wg.Done()
		if t.Kind() == pwebrtc.RTPCodecTypeAudio {
			s.readAudio(t)
			return
		}
		discardTrack(t)
	})
	return p, nil
}

var videoCodecs = []pwebrtc.RTPCodecParameters{
	{RTPCodecCapability: pwebrtc.RTPCodecCapability{MimeType: pwebrtc.MimeTypeVP8, ClockRate: 90000}, PayloadType: 96},
	{RTPCodecCapability: pwebrtc.RTPCodecCapability{MimeType: pwebrtc.MimeTypeH264, ClockRate: 90000, SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"}, PayloadType: 102},
}

func drainRTCP(sender *pwebrtc.RTPSender) {
	buf := make([]byte, 1500)
	for {
		if _, _, err := sender.Read(buf); err != nil {
			return
		}
	}
}

func discardTrack(t *pwebrtc.TrackRemote) {
	buf := make([]byte, 1500)
	for {
		if _, _, err := t.Read(buf); err != nil {
			return
		}
	}
}

func (s *Session) readAudio(t *pwebrtc.TrackRemote) {
	codec := codecForMime(s.handler.cfg.Codecs, t.Codec().RTPCodecCapability)
	if codec == nil {
		s.log.Warn("unsupported inbound audio codec %s", t.Codec().MimeType)
		discardTrack(t)
		return
	}
	dec, err := codec.NewDecoder()
	if err != nil {
		s.log.Warn("audio decoder: %v", err)
		discardTrack(t)
		return
	}
	wire := codec.SampleRate()
	var (
		buf     = make([]byte, 1500)
		pkt     rtp.Packet
		scratch []byte
		rs      *resample.Resampler
		lastSeq uint16
		haveSeq bool
	)
	for {
		n, _, err := t.Read(buf)
		if err != nil {
			return
		}
		if pkt.Unmarshal(buf[:n]) != nil || len(pkt.Payload) == 0 {
			continue
		}
		if haveSeq && int16(pkt.SequenceNumber-lastSeq) <= 0 {
			continue
		}
		lastSeq, haveSeq = pkt.SequenceNumber, true
		if !s.ready.Load() {
			continue
		}
		target := s.input.targetRate
		if target <= 0 {
			target = wire
		}
		var data []byte
		if target == wire {
			if data, err = dec.Decode(nil, pkt.Payload); err != nil {
				continue
			}
		} else {
			if scratch, err = dec.Decode(scratch[:0], pkt.Payload); err != nil {
				continue
			}
			if rs == nil || rs.OutRate() != target {
				if rs, err = resample.New(wire, target); err != nil {
					s.log.Warn("input resampler: %v", err)
					discardTrack(t)
					return
				}
			}
			data = rs.ProcessBytes(make([]byte, 0, 2*rs.MaxOutput(len(scratch)/2)), scratch)
		}
		s.input.deliverAudio(frames.NewAudioFrame(data, target, 1))
	}
}

type loopState struct {
	cur             *peer
	dc              *pwebrtc.DataChannel
	dcOpen          bool
	dcTimerStarted  bool
	queueing        bool
	pending         []outMsg
	pendingIn       []any
	pcConnected     bool
	started         bool
	clientConnected bool
	disconnected    bool

	connectT    *time.Timer
	disconnectT *time.Timer
	dcT         *time.Timer
	keepT       *time.Timer
	sessionT    *time.Timer
}

func newStoppedTimer() *time.Timer {
	t := time.NewTimer(time.Hour)
	t.Stop()
	return t
}

func (s *Session) loop() {
	defer s.wg.Done()
	defer close(s.loopDone)
	st := &loopState{
		queueing:    true,
		connectT:    newStoppedTimer(),
		disconnectT: newStoppedTimer(),
		dcT:         newStoppedTimer(),
		keepT:       newStoppedTimer(),
		sessionT:    newStoppedTimer(),
	}
	defer func() {
		for _, t := range []*time.Timer{st.connectT, st.disconnectT, st.dcT, st.keepT, st.sessionT} {
			t.Stop()
		}
	}()
	startedCh := s.startedCh
	for {
		select {
		case <-s.closeCh:
			s.teardown(st)
			return
		case <-startedCh:
			startedCh = nil
			st.started = true
			if st.disconnected {
				s.handleDisconnect()
			} else {
				s.maybeConnected(st)
			}
		case e := <-s.events:
			s.handleEvent(st, e)
		case m := <-s.outbox:
			s.sendOut(st, m)
		case <-st.connectT.C:
			s.markDisconnected(st, "connect timeout")
		case <-st.disconnectT.C:
			s.markDisconnected(st, "peer disconnected")
		case <-st.keepT.C:
			s.markDisconnected(st, "keep-alive timeout")
		case <-st.dcT.C:
			if !st.dcOpen {
				s.log.Warn("data channel not open within %v; discarding queued messages", s.params.DataChannelTimeout)
				st.pending = nil
				st.queueing = false
			}
		case <-st.sessionT.C:
			s.onSessionTimeout()
		}
	}
}

func (s *Session) handleEvent(st *loopState, e event) {
	if e.kind == evPeer {
		st.cur = e.peer
		st.dc, st.dcOpen, st.dcTimerStarted, st.pcConnected = nil, false, false, false
		st.queueing = true
		st.disconnectT.Stop()
		st.dcT.Stop()
		if !st.disconnected {
			st.connectT.Reset(s.params.ConnectTimeout)
		}
		return
	}
	if e.peer != st.cur || e.peer != s.currentPeer() {
		return
	}
	switch e.kind {
	case evState:
		s.onState(st)
	case evDataChannel:
		st.dc, st.dcOpen = e.dc, false
		if e.dc.ReadyState() == pwebrtc.DataChannelStateOpen {
			s.onDCOpen(st)
		}
	case evDCOpen:
		if e.dc == st.dc && !st.dcOpen {
			s.onDCOpen(st)
		}
	case evDCClose:
		if e.dc == st.dc {
			st.dc, st.dcOpen = nil, false
			if !s.closing.Load() {
				s.markDisconnected(st, "data channel closed")
			}
		}
	case evMessage:
		s.onMessage(st, e.msg)
	}
}

func (s *Session) onState(st *loopState) {
	state := st.cur.pc.ConnectionState()
	switch state {
	case pwebrtc.PeerConnectionStateConnected:
		st.connectT.Stop()
		st.disconnectT.Stop()
		st.pcConnected = true
		if !st.dcOpen && !st.dcTimerStarted {
			st.dcTimerStarted = true
			st.dcT.Reset(s.params.DataChannelTimeout)
		}
		s.maybeConnected(st)
	case pwebrtc.PeerConnectionStateDisconnected:
		if st.pcConnected {
			st.disconnectT.Reset(s.params.DisconnectTimeout)
		}
	case pwebrtc.PeerConnectionStateFailed, pwebrtc.PeerConnectionStateClosed:
		if !s.closing.Load() {
			s.markDisconnected(st, "connection "+state.String())
		}
	}
}

func (s *Session) onDCOpen(st *loopState) {
	st.dcOpen = true
	st.dcT.Stop()
	for _, m := range st.pending {
		s.write(st.dc, m)
	}
	st.pending = nil
}

func (s *Session) write(dc *pwebrtc.DataChannel, m outMsg) {
	var err error
	if m.binary {
		err = dc.Send(m.data)
	} else {
		err = dc.SendText(string(m.data))
	}
	if err != nil && !s.closing.Load() {
		s.log.Debug("data channel send: %v", err)
	}
}

func (s *Session) sendOut(st *loopState, m outMsg) {
	switch {
	case st.dcOpen:
		s.write(st.dc, m)
	case st.queueing && len(st.pending) < maxPendingOut:
		st.pending = append(st.pending, m)
	case st.queueing:
		s.log.Warn("data channel queue full; discarding message")
	}
}

func (s *Session) onMessage(st *loopState, m pwebrtc.DataChannelMessage) {
	if m.IsString && bytes.HasPrefix(m.Data, []byte("ping")) {
		if s.params.KeepAliveTimeout > 0 && !st.disconnected {
			st.keepT.Reset(s.params.KeepAliveTimeout)
		}
		return
	}
	var v any
	if err := json.Unmarshal(m.Data, &v); err != nil {
		s.log.Debug("ignoring non-JSON data channel message: %v", err)
		return
	}
	if obj, ok := v.(map[string]any); ok && obj["type"] == signallingType && obj["message"] != nil {
		s.log.Debug("signalling message: %v", obj["message"])
		return
	}
	if !st.clientConnected {
		if len(st.pendingIn) < maxPendingIn {
			st.pendingIn = append(st.pendingIn, v)
		}
		return
	}
	s.deliverMessage(v)
}

func (s *Session) deliverMessage(v any) {
	_ = s.input.BroadcastFrame(s.ctx, func() frames.Frame { return frames.NewInputTransportMessageFrame(v) })
}

func (s *Session) maybeConnected(st *loopState) {
	if !st.started || !st.pcConnected || st.clientConnected || st.disconnected {
		return
	}
	st.clientConnected = true
	s.ready.Store(true)
	s.log.Info("client connected")
	_ = s.input.PushFrame(frames.NewClientConnectedFrame(), frames.Downstream)
	if cb := s.params.OnClientConnected; cb != nil {
		cb(s)
	}
	for _, v := range st.pendingIn {
		s.deliverMessage(v)
	}
	st.pendingIn = nil
	if s.params.SessionTimeout > 0 {
		st.sessionT.Reset(s.params.SessionTimeout)
	}
}

func (s *Session) markDisconnected(st *loopState, reason string) {
	if st.disconnected {
		return
	}
	st.disconnected = true
	s.ready.Store(false)
	st.connectT.Stop()
	st.disconnectT.Stop()
	st.keepT.Stop()
	st.sessionT.Stop()
	close(s.disconnectedCh)
	s.log.Info("client disconnected: %s", reason)
	if st.started {
		s.handleDisconnect()
	}
}

func (s *Session) handleDisconnect() {
	if s.closing.Load() || s.input.ending.Load() {
		return
	}
	_ = s.input.PushFrame(frames.NewClientDisconnectedFrame(), frames.Downstream)
	if cb := s.params.OnClientDisconnected; cb != nil {
		cb(s)
		return
	}
	_ = s.input.PushFrame(frames.NewCancelTaskFrame("client disconnected"), frames.Upstream)
}

func (s *Session) onSessionTimeout() {
	if s.closing.Load() || s.input.ending.Load() {
		return
	}
	s.log.Info("session timeout after %v", s.params.SessionTimeout)
	if cb := s.params.OnSessionTimeout; cb != nil {
		cb(s)
		return
	}
	_ = s.input.PushFrame(frames.NewEndTaskFrame("session timeout"), frames.Upstream)
}

func (s *Session) teardown(st *loopState) {
	if st.dcOpen && st.dc != nil {
		s.sendPeerLeft(st.dc)
	}
	s.cancel()
	s.input.stopFilter()
	s.mu.Lock()
	p := s.peer
	s.peer = nil
	s.mu.Unlock()
	if p != nil {
		p.close()
	}
}

func (s *Session) sendPeerLeft(dc *pwebrtc.DataChannel) {
	flushed := make(chan struct{})
	var once sync.Once
	dc.SetBufferedAmountLowThreshold(0)
	dc.OnBufferedAmountLow(func() { once.Do(func() { close(flushed) }) })
	if err := dc.SendText(string(peerLeftMessage)); err != nil {
		return
	}
	if dc.BufferedAmount() == 0 {
		return
	}
	t := time.NewTimer(s.params.CloseTimeout)
	defer t.Stop()
	select {
	case <-flushed:
	case <-t.C:
	}
}
