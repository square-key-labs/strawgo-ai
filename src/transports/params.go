package transports

import (
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/audio/filter"
	"github.com/square-key-labs/strawgo-ai/src/audio/mixer"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

const (
	DefaultAudioOutChunk      = 20 * time.Millisecond
	DefaultWriteTimeout       = 10 * time.Second
	DefaultPingInterval       = 20 * time.Second
	DefaultHandshakeTimeout   = 10 * time.Second
	DefaultCloseTimeout       = 2 * time.Second
	DefaultPlaybackAckTimeout = time.Second
	DefaultBotStopTimeout     = 3 * time.Second
	DefaultHangupTimeout      = 10 * time.Second
	botSpeakingPeriod         = 200 * time.Millisecond
	outputQueueSize           = 64
)

type AudioInputAnalyzer interface {
	AnalyzeAudio(frame *frames.AudioFrame, emit func(frames.Frame))
}

type AudioMixer = mixer.Mixer

type TransportParams struct {
	AudioInSampleRate         int
	AudioOutSampleRate        int
	AudioOutChunk             time.Duration
	DisableAudioInPassthrough bool
	AudioInAnalyzer           AudioInputAnalyzer
	NewAudioInAnalyzer        func() AudioInputAnalyzer
	NewOutputMixer            func() mixer.Mixer
	NewInputFilter            func() filter.Filter
	AudioOutMixer             AudioMixer
	OutputMixer               mixer.Mixer
	InputFilter               filter.Filter
	AudioInNative             bool
	AudioOutNative            bool
	AudioOutEndSilence        time.Duration
	InterruptionFade          time.Duration
	BotStopTimeout            time.Duration
	PlaybackAck               bool
	PlaybackAckTimeout        time.Duration
	SessionTimeout            time.Duration
	WriteTimeout              time.Duration
	PingInterval              time.Duration
	HandshakeTimeout          time.Duration
	CloseTimeout              time.Duration
	HangupTimeout             time.Duration
	OnClientConnected         func(*Session)
	OnClientDisconnected      func(*Session)
	OnSessionTimeout          func(*Session)
}

func (p TransportParams) withDefaults() TransportParams {
	def := func(v *time.Duration, d time.Duration) {
		if *v == 0 {
			*v = d
		}
	}
	def(&p.AudioOutChunk, DefaultAudioOutChunk)
	def(&p.WriteTimeout, DefaultWriteTimeout)
	def(&p.PingInterval, DefaultPingInterval)
	def(&p.HandshakeTimeout, DefaultHandshakeTimeout)
	def(&p.CloseTimeout, DefaultCloseTimeout)
	def(&p.PlaybackAckTimeout, DefaultPlaybackAckTimeout)
	def(&p.BotStopTimeout, DefaultBotStopTimeout)
	def(&p.HangupTimeout, DefaultHangupTimeout)
	if p.AudioOutChunk < time.Millisecond {
		p.AudioOutChunk = DefaultAudioOutChunk
	}
	return p
}

func (p TransportParams) outputMixer() mixer.Mixer {
	if p.OutputMixer != nil {
		return p.OutputMixer
	}
	return p.AudioOutMixer
}

func frameCodec(f frames.Frame) string {
	if h, ok := f.(interface{ HasMetadata() bool }); ok && !h.HasMetadata() {
		return audio.CodecLinear16
	}
	codec, _ := f.Metadata()["codec"].(string)
	return audio.NormalizeCodec(codec)
}
