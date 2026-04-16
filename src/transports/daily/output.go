package daily

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type OutputProcessor struct {
	*processors.BaseProcessor
	transport *DailyTransport
	codec     audioCodec

	trackMu sync.RWMutex
	writer  outgoingAudioTrack
}

func newOutputProcessor(transport *DailyTransport) *OutputProcessor {
	p := &OutputProcessor{
		transport: transport,
		codec:     transport.codec,
	}
	p.BaseProcessor = processors.NewBaseProcessor("DailyOutput", p)
	return p
}

func (p *OutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
		return p.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.EndFrame); ok {
		p.clearTrack()
		return nil
	}

	if ttsFrame, ok := frame.(*frames.TTSAudioFrame); ok {
		return p.sendAudioFrame(ttsFrame)
	}

	if _, ok := frame.(*frames.AudioFrame); ok {
		return nil
	}

	return p.PushFrame(frame, direction)
}

func (p *OutputProcessor) setTrack(track outgoingAudioTrack) {
	p.trackMu.Lock()
	defer p.trackMu.Unlock()

	if p.writer != nil {
		_ = p.writer.Close()
	}
	p.writer = track
}

func (p *OutputProcessor) clearTrack() {
	p.trackMu.Lock()
	defer p.trackMu.Unlock()

	if p.writer != nil {
		_ = p.writer.Close()
	}
	p.writer = nil
}

func (p *OutputProcessor) track() outgoingAudioTrack {
	p.trackMu.RLock()
	defer p.trackMu.RUnlock()
	return p.writer
}

func (p *OutputProcessor) sendAudioFrame(frame *frames.TTSAudioFrame) error {
	codecName, _ := frame.Metadata()["codec"].(string)

	// Pre-encoded Opus: send directly (caller guarantees one frame per call).
	if codecName == "opus" {
		return p.transport.sendOpus(frame.Data, 20*time.Millisecond)
	}

	// Reject non-PCM codecs we cannot encode.
	if codecName != "linear16" && codecName != "" {
		return fmt.Errorf("daily transport: cannot encode codec %q; use linear16 or pre-encode to opus", codecName)
	}

	// Chunk linear16 PCM into 20ms Opus frames.
	// pion/webrtc's OpusPayloader is a blind byte copy — each WriteSample must
	// carry exactly one Opus frame, or the remote decoder will error.
	frameBytes := (frame.SampleRate / 50) * 2 // 20ms at sampleRate, int16 LE
	if frameBytes <= 0 {
		frameBytes = 1920 // fallback: 48kHz 20ms
	}

	data := frame.Data
	for len(data) >= frameBytes {
		encoded, err := p.codec.EncodePCMToOpus(data[:frameBytes], frame.SampleRate, frame.Channels)
		if err != nil {
			return fmt.Errorf("encode pcm to opus: %w", err)
		}
		if err := p.transport.sendOpus(encoded, 20*time.Millisecond); err != nil {
			return fmt.Errorf("send opus packet: %w", err)
		}
		data = data[frameBytes:]
	}

	// Zero-pad tail frame (if any).
	if len(data) > 0 {
		padded := make([]byte, frameBytes)
		copy(padded, data)
		encoded, err := p.codec.EncodePCMToOpus(padded, frame.SampleRate, frame.Channels)
		if err != nil {
			return fmt.Errorf("encode pcm to opus (tail): %w", err)
		}
		if err := p.transport.sendOpus(encoded, 20*time.Millisecond); err != nil {
			return fmt.Errorf("send opus packet (tail): %w", err)
		}
	}

	return nil
}
