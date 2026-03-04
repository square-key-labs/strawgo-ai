package daily

import (
	"context"
	"errors"
	"io"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type InputProcessor struct {
	*processors.BaseProcessor
	transport *DailyTransport
	codec     audioCodec
}

func newInputProcessor(transport *DailyTransport) *InputProcessor {
	p := &InputProcessor{
		transport: transport,
		codec:     transport.codec,
	}
	p.BaseProcessor = processors.NewBaseProcessor("DailyInput", p)
	return p
}

func (p *InputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
	}
	return p.PushFrame(frame, direction)
}

func (p *InputProcessor) consumeRemoteTrack(track remoteAudioTrack) {
	participantID := track.StreamID()
	if participantID == "" {
		participantID = track.ID()
	}
	p.transport.participantJoined(participantID)

	// Emit ClientConnectedFrame when a remote participant joins
	if err := p.PushFrame(frames.NewClientConnectedFrame(), frames.Downstream); err != nil {
		_ = p.PushError("emit client connected frame", err, false)
	}

	go func() {
		defer p.transport.participantLeft(participantID)

		for {
			payload, err := track.ReadOpusPacket()
			if err != nil {
				if errors.Is(err, io.EOF) {
					return
				}
				_ = p.PushError("read daily audio track", err, false)
				return
			}

			if len(payload) == 0 {
				continue
			}

			codecName := "opus"
			audioData := payload
			if strings.Contains(strings.ToLower(track.CodecMimeType()), "opus") {
				decoded, decodeErr := p.codec.DecodeOpusToPCM(payload, p.transport.config.SampleRate, p.transport.config.Channels)
				if decodeErr == nil && len(decoded) > 0 {
					audioData = decoded
					codecName = "linear16"
				}
			}

			frame := frames.NewAudioFrame(audioData, p.transport.config.SampleRate, p.transport.config.Channels)
			frame.SetMetadata("codec", codecName)
			frame.SetMetadata("participant_id", participantID)
			if err := p.BaseProcessor.PushFrame(frame, frames.Downstream); err != nil {
				_ = p.PushError("emit daily audio frame", err, false)
				return
			}
		}
	}()
}
