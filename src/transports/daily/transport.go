package daily

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/pion/opus"
	"github.com/pion/webrtc/v3"
	"github.com/pion/webrtc/v3/pkg/media"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	defaultDailyAPIBase   = "https://api.daily.co/v1"
	defaultSampleRate     = 48000
	defaultChannels       = 1
	defaultReconnectDelay = 2 * time.Second
)

type DailyConfig struct {
	APIKey             string
	APIBaseURL         string
	RoomName           string
	CreateRoomIfAbsent bool
	SampleRate         int
	Channels           int
	ReconnectDelay     time.Duration
	HTTPClient         *http.Client
	WebRTCConfig       webrtc.Configuration
	OnParticipantJoin  func(participantID string)
	OnParticipantLeave func(participantID string)
}

type Participant struct {
	ID       string
	JoinedAt time.Time
}

type DailyTransport struct {
	config     DailyConfig
	httpClient *http.Client

	peerFactory func(cfg webrtc.Configuration) (peerConnection, error)
	peerMu      sync.RWMutex
	peer        peerConnection

	codec audioCodec

	inputProc  *InputProcessor
	outputProc *OutputProcessor

	participantsMu sync.RWMutex
	participants   map[string]Participant

	lifecycleMu sync.Mutex
	running     bool
	ctx         context.Context
	cancel      context.CancelFunc

	roomURL      string
	meetingToken string
	joinRoomFn   func(ctx context.Context) error

	reconnectCh chan struct{}
	reconnectWg sync.WaitGroup
}

type peerConnection interface {
	OnRemoteTrack(func(remoteAudioTrack))
	OnConnectionStateChange(func(connectionState))
	CreateOutgoingAudioTrack() (outgoingAudioTrack, error)
	Close() error
}

type connectionState string

const (
	connectionStateNew          connectionState = "new"
	connectionStateConnecting   connectionState = "connecting"
	connectionStateConnected    connectionState = "connected"
	connectionStateDisconnected connectionState = "disconnected"
	connectionStateFailed       connectionState = "failed"
	connectionStateClosed       connectionState = "closed"
)

type remoteAudioTrack interface {
	ReadOpusPacket() ([]byte, error)
	CodecMimeType() string
	ID() string
	StreamID() string
}

type outgoingAudioTrack interface {
	WriteOpus(payload []byte, duration time.Duration) error
	Close() error
}

type audioCodec interface {
	DecodeOpusToPCM(payload []byte, sampleRate, channels int) ([]byte, error)
	EncodePCMToOpus(pcm []byte, sampleRate, channels int) ([]byte, error)
}

type pionAudioCodec struct {
	mu      sync.Mutex
	decoder opus.Decoder
}

func newPionAudioCodec() audioCodec {
	return &pionAudioCodec{decoder: opus.NewDecoder()}
}

func (c *pionAudioCodec) DecodeOpusToPCM(payload []byte, sampleRate, channels int) ([]byte, error) {
	if len(payload) == 0 {
		return nil, nil
	}

	if channels <= 0 {
		channels = defaultChannels
	}

	pcm := make([]byte, 5760*2*channels)
	c.mu.Lock()
	_, _, err := c.decoder.Decode(payload, pcm)
	c.mu.Unlock()
	if err != nil {
		return nil, err
	}

	return pcm, nil
}

func (c *pionAudioCodec) EncodePCMToOpus(pcm []byte, sampleRate, channels int) ([]byte, error) {
	return nil, errors.New("opus encoding is not available in pure Go; send TTSAudioFrame with metadata codec=opus")
}

func NewDailyTransport(config DailyConfig) *DailyTransport {
	if config.APIBaseURL == "" {
		config.APIBaseURL = defaultDailyAPIBase
	}
	config.APIBaseURL = strings.TrimRight(config.APIBaseURL, "/")
	if config.SampleRate == 0 {
		config.SampleRate = defaultSampleRate
	}
	if config.Channels == 0 {
		config.Channels = defaultChannels
	}
	if config.ReconnectDelay <= 0 {
		config.ReconnectDelay = defaultReconnectDelay
	}

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}

	t := &DailyTransport{
		config:       config,
		httpClient:   httpClient,
		codec:        newPionAudioCodec(),
		participants: make(map[string]Participant),
		reconnectCh:  make(chan struct{}, 1),
	}
	t.joinRoomFn = t.joinRoom

	t.peerFactory = func(cfg webrtc.Configuration) (peerConnection, error) {
		return newPionPeer(cfg)
	}
	t.inputProc = newInputProcessor(t)
	t.outputProc = newOutputProcessor(t)

	return t
}

func (t *DailyTransport) Input() processors.FrameProcessor {
	return t.inputProc
}

func (t *DailyTransport) Output() processors.FrameProcessor {
	return t.outputProc
}

func (t *DailyTransport) Start(ctx context.Context) error {
	if err := t.Connect(ctx); err != nil {
		return err
	}

	<-ctx.Done()
	return t.Disconnect()
}

func (t *DailyTransport) Connect(ctx context.Context) error {
	t.lifecycleMu.Lock()
	defer t.lifecycleMu.Unlock()

	if t.running {
		return nil
	}

	runCtx, cancel := context.WithCancel(ctx)
	t.ctx = runCtx
	t.cancel = cancel

	if err := t.joinRoomFn(runCtx); err != nil {
		cancel()
		return err
	}

	peer, err := t.peerFactory(t.config.WebRTCConfig)
	if err != nil {
		cancel()
		return fmt.Errorf("create peer: %w", err)
	}

	if err := t.configurePeer(peer); err != nil {
		_ = peer.Close()
		cancel()
		return err
	}

	t.peerMu.Lock()
	t.peer = peer
	t.peerMu.Unlock()

	t.running = true
	t.reconnectWg.Add(1)
	go t.reconnectLoop()

	return nil
}

func (t *DailyTransport) Disconnect() error {
	t.lifecycleMu.Lock()
	defer t.lifecycleMu.Unlock()

	if !t.running {
		return nil
	}

	t.running = false
	if t.cancel != nil {
		t.cancel()
	}

	t.peerMu.Lock()
	peer := t.peer
	t.peer = nil
	t.peerMu.Unlock()

	if peer != nil {
		_ = peer.Close()
	}

	t.outputProc.clearTrack()
	t.reconnectWg.Wait()

	return nil
}

func (t *DailyTransport) requestReconnect() {
	select {
	case t.reconnectCh <- struct{}{}:
	default:
	}
}

func (t *DailyTransport) reconnectLoop() {
	defer t.reconnectWg.Done()

	for {
		select {
		case <-t.ctx.Done():
			return
		case <-t.reconnectCh:
			t.tryReconnect()
		}
	}
}

func (t *DailyTransport) tryReconnect() {
	time.Sleep(t.config.ReconnectDelay)

	t.lifecycleMu.Lock()
	if !t.running {
		t.lifecycleMu.Unlock()
		return
	}
	t.lifecycleMu.Unlock()

	peer, err := t.peerFactory(t.config.WebRTCConfig)
	if err != nil {
		return
	}

	if err := t.configurePeer(peer); err != nil {
		_ = peer.Close()
		return
	}

	t.peerMu.Lock()
	oldPeer := t.peer
	t.peer = peer
	t.peerMu.Unlock()

	if oldPeer != nil {
		_ = oldPeer.Close()
	}
}

func (t *DailyTransport) configurePeer(peer peerConnection) error {
	peer.OnRemoteTrack(func(track remoteAudioTrack) {
		t.inputProc.consumeRemoteTrack(track)
	})

	peer.OnConnectionStateChange(func(state connectionState) {
		if state == connectionStateFailed || state == connectionStateDisconnected {
			t.requestReconnect()
		}
	})

	outgoing, err := peer.CreateOutgoingAudioTrack()
	if err != nil {
		return fmt.Errorf("create outgoing track: %w", err)
	}
	t.outputProc.setTrack(outgoing)

	// Emit BotConnectedFrame when bot's peer connection is set up
	if err := t.outputProc.PushFrame(frames.NewBotConnectedFrame(), frames.Downstream); err != nil {
		return fmt.Errorf("emit bot connected frame: %w", err)
	}

	return nil
}

func (t *DailyTransport) Participants() []Participant {
	t.participantsMu.RLock()
	defer t.participantsMu.RUnlock()

	out := make([]Participant, 0, len(t.participants))
	for _, p := range t.participants {
		out = append(out, p)
	}

	return out
}

func (t *DailyTransport) participantJoined(participantID string) {
	if participantID == "" {
		return
	}

	t.participantsMu.Lock()
	_, exists := t.participants[participantID]
	if !exists {
		t.participants[participantID] = Participant{ID: participantID, JoinedAt: time.Now()}
	}
	t.participantsMu.Unlock()

	if !exists && t.config.OnParticipantJoin != nil {
		t.config.OnParticipantJoin(participantID)
	}
}

func (t *DailyTransport) participantLeft(participantID string) {
	if participantID == "" {
		return
	}

	t.participantsMu.Lock()
	_, exists := t.participants[participantID]
	if exists {
		delete(t.participants, participantID)
	}
	t.participantsMu.Unlock()

	if exists && t.config.OnParticipantLeave != nil {
		t.config.OnParticipantLeave(participantID)
	}
}

func (t *DailyTransport) joinRoom(ctx context.Context) error {
	if t.config.APIKey == "" {
		return errors.New("daily api key is required")
	}
	if t.config.RoomName == "" {
		return errors.New("daily room name is required")
	}

	if t.config.CreateRoomIfAbsent {
		if _, err := t.dailyPOST(ctx, "/rooms", map[string]interface{}{"name": t.config.RoomName}); err != nil {
			return fmt.Errorf("create room: %w", err)
		}
	}

	resBody, err := t.dailyPOST(ctx, "/meeting-tokens", map[string]interface{}{
		"properties": map[string]interface{}{
			"room_name": t.config.RoomName,
		},
	})
	if err != nil {
		return fmt.Errorf("create meeting token: %w", err)
	}

	var tokenResp struct {
		Token string `json:"token"`
	}
	if err := json.Unmarshal(resBody, &tokenResp); err != nil {
		return fmt.Errorf("decode meeting token response: %w", err)
	}
	if tokenResp.Token == "" {
		return errors.New("meeting token response missing token")
	}

	t.meetingToken = tokenResp.Token
	t.roomURL = fmt.Sprintf("https://%s.daily.co/%s", "api", t.config.RoomName)

	return nil
}

func (t *DailyTransport) dailyPOST(ctx context.Context, path string, payload interface{}) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, t.config.APIBaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+t.config.APIKey)
	request.Header.Set("Content-Type", "application/json")

	response, err := t.httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}

	if response.StatusCode >= 400 {
		return nil, fmt.Errorf("daily api %s failed: %s", path, strings.TrimSpace(string(responseBody)))
	}

	return responseBody, nil
}

func (t *DailyTransport) sendOpus(payload []byte, duration time.Duration) error {
	if len(payload) == 0 {
		return nil
	}

	track := t.outputProc.track()
	if track == nil {
		return errors.New("daily transport not connected")
	}

	return track.WriteOpus(payload, duration)
}

type pionPeer struct {
	pc *webrtc.PeerConnection
}

func newPionPeer(config webrtc.Configuration) (peerConnection, error) {
	pc, err := webrtc.NewPeerConnection(config)
	if err != nil {
		return nil, err
	}
	return &pionPeer{pc: pc}, nil
}

func (p *pionPeer) OnRemoteTrack(handler func(remoteAudioTrack)) {
	p.pc.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		_ = receiver
		handler(&pionRemoteTrack{track: track})
	})
}

func (p *pionPeer) OnConnectionStateChange(handler func(connectionState)) {
	p.pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		handler(mapConnectionState(state))
	})
}

func (p *pionPeer) CreateOutgoingAudioTrack() (outgoingAudioTrack, error) {
	track, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 1},
		"audio",
		"strawgo",
	)
	if err != nil {
		return nil, err
	}

	sender, err := p.pc.AddTrack(track)
	if err != nil {
		return nil, err
	}

	go func() {
		rtcpBuf := make([]byte, 1500)
		for {
			if _, _, readErr := sender.Read(rtcpBuf); readErr != nil {
				return
			}
		}
	}()

	return &pionOutgoingTrack{track: track}, nil
}

func (p *pionPeer) Close() error {
	return p.pc.Close()
}

type pionRemoteTrack struct {
	track *webrtc.TrackRemote
}

func (t *pionRemoteTrack) ReadOpusPacket() ([]byte, error) {
	packet, _, err := t.track.ReadRTP()
	if err != nil {
		return nil, err
	}
	return packet.Payload, nil
}

func (t *pionRemoteTrack) CodecMimeType() string {
	return t.track.Codec().MimeType
}

func (t *pionRemoteTrack) ID() string {
	return t.track.ID()
}

func (t *pionRemoteTrack) StreamID() string {
	return t.track.StreamID()
}

type pionOutgoingTrack struct {
	track *webrtc.TrackLocalStaticSample
}

func (t *pionOutgoingTrack) WriteOpus(payload []byte, duration time.Duration) error {
	if duration <= 0 {
		duration = 20 * time.Millisecond
	}
	return t.track.WriteSample(media.Sample{Data: payload, Duration: duration})
}

func (t *pionOutgoingTrack) Close() error {
	return nil
}

func mapConnectionState(state webrtc.PeerConnectionState) connectionState {
	switch state {
	case webrtc.PeerConnectionStateNew:
		return connectionStateNew
	case webrtc.PeerConnectionStateConnecting:
		return connectionStateConnecting
	case webrtc.PeerConnectionStateConnected:
		return connectionStateConnected
	case webrtc.PeerConnectionStateDisconnected:
		return connectionStateDisconnected
	case webrtc.PeerConnectionStateFailed:
		return connectionStateFailed
	case webrtc.PeerConnectionStateClosed:
		return connectionStateClosed
	default:
		return connectionStateNew
	}
}
