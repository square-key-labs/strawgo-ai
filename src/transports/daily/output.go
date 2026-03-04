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
	payload := frame.Data

	if codecName != "opus" {
		encoded, err := p.codec.EncodePCMToOpus(frame.Data, frame.SampleRate, frame.Channels)
		if err != nil {
			return fmt.Errorf("encode tts pcm to opus: %w", err)
		}
		payload = encoded
	}

	duration := 20 * time.Millisecond
	if durationMS, ok := frame.Metadata()["duration_ms"]; ok {
		switch v := durationMS.(type) {
		case int:
			duration = time.Duration(v) * time.Millisecond
		case int64:
			duration = time.Duration(v) * time.Millisecond
		case float64:
			duration = time.Duration(v * float64(time.Millisecond))
		}
	}

	if err := p.transport.sendOpus(payload, duration); err != nil {
		return fmt.Errorf("send daily opus packet: %w", err)
	}

	return nil
}
