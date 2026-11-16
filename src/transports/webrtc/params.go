package webrtc

import (
	"encoding/json"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/filter"
	"github.com/square-key-labs/strawgo-ai/src/audio/mixer"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

const (
	DefaultConnectTimeout     = 60 * time.Second
	DefaultDisconnectTimeout  = 5 * time.Second
	DefaultDataChannelTimeout = 10 * time.Second
	DefaultGatherTimeout      = 10 * time.Second
	DefaultCloseTimeout       = 250 * time.Millisecond

	audioChunk        = 20 * time.Millisecond
	botSpeakingPeriod = 200 * time.Millisecond
	outputQueueSize   = 64
	maxPendingOut     = 50
	maxPendingIn      = 50
	maxBodyBytes      = 1 << 20
	signallingType    = "signalling"
)

type Params struct {
	AudioInSampleRate         int
	DisableAudioInPassthrough bool
	AudioInAnalyzer           transports.AudioInputAnalyzer
	NewAudioInAnalyzer        func() transports.AudioInputAnalyzer
	NewOutputMixer            func() mixer.Mixer
	NewInputFilter            func() filter.Filter
	InputFilter               filter.Filter
	AudioOutMixer             transports.AudioMixer
	DisableAudioOutSilence    bool
	InterruptionFade          time.Duration
	BotStopTimeout            time.Duration
	ConnectTimeout            time.Duration
	DisconnectTimeout         time.Duration
	DataChannelTimeout        time.Duration
	KeepAliveTimeout          time.Duration
	SessionTimeout            time.Duration
	CloseTimeout              time.Duration
	OnClientConnected         func(*Session)
	OnClientDisconnected      func(*Session)
	OnSessionTimeout          func(*Session)
}

func (p Params) withDefaults() Params {
	def := func(v *time.Duration, d time.Duration) {
		if *v == 0 {
			*v = d
		}
	}
	def(&p.BotStopTimeout, transports.DefaultBotStopTimeout)
	def(&p.ConnectTimeout, DefaultConnectTimeout)
	def(&p.DisconnectTimeout, DefaultDisconnectTimeout)
	def(&p.DataChannelTimeout, DefaultDataChannelTimeout)
	def(&p.CloseTimeout, DefaultCloseTimeout)
	return p
}

func (p Params) forSession() Params {
	if p.NewAudioInAnalyzer != nil {
		p.AudioInAnalyzer = p.NewAudioInAnalyzer()
	}
	if p.NewOutputMixer != nil {
		p.AudioOutMixer = p.NewOutputMixer()
	}
	if p.NewInputFilter != nil {
		p.InputFilter = p.NewInputFilter()
	}
	return p
}

type OfferRequest struct {
	SDP              string          `json:"sdp"`
	Type             string          `json:"type"`
	PCID             string          `json:"pc_id,omitempty"`
	RestartPC        bool            `json:"restart_pc,omitempty"`
	RequestData      json.RawMessage `json:"request_data,omitempty"`
	RequestDataCamel json.RawMessage `json:"requestData,omitempty"`
}

type Answer struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
	PCID string `json:"pc_id"`
}

type ICECandidate struct {
	Candidate     string `json:"candidate"`
	SDPMid        string `json:"sdp_mid"`
	SDPMLineIndex uint16 `json:"sdp_mline_index"`
}

type PatchRequest struct {
	PCID       string         `json:"pc_id"`
	Candidates []ICECandidate `json:"candidates"`
}

var (
	peerLeftMessage    = []byte(`{"type":"signalling","message":{"type":"peerLeft"}}`)
	renegotiateMessage = []byte(`{"type":"signalling","message":{"type":"renegotiate"}}`)
)
