package stt

import (
	"context"
	"encoding/binary"
	"errors"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

const (
	DefaultPreRoll         = 500 * time.Millisecond
	DefaultTrailingSilence = 500 * time.Millisecond
	DefaultMaxSegment      = 30 * time.Second
	segmentDrainTimeout    = 30 * time.Second
)

type segmenter struct {
	b        *Base
	provider SegmentedProvider

	preroll []byte
	buf     []byte
	active  bool

	queue  chan []byte
	closed bool
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func newSegmenter(b *Base, provider SegmentedProvider) *segmenter {
	size := b.opts.SegmentQueue
	if size <= 0 {
		size = 8
	}
	return &segmenter{b: b, provider: provider, queue: make(chan []byte, size)}
}

func (s *segmenter) bytesFor(d time.Duration) int {
	n := int(d.Seconds() * float64(s.b.SampleRate()))
	return n * s.b.opts.BytesPerSample
}

func (s *segmenter) prerollBytes() int {
	d := s.b.opts.PreRoll
	if d <= 0 {
		d = DefaultPreRoll
	}
	return s.bytesFor(d)
}

func (s *segmenter) maxBytes() int {
	d := s.b.opts.MaxSegment
	if d <= 0 {
		d = DefaultMaxSegment
	}
	return s.bytesFor(d)
}

func (s *segmenter) start(ctx context.Context) {
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.wg.Add(1)
	go s.worker()
}

func (s *segmenter) append(data []byte) {
	if s.active {
		s.buf = append(s.buf, data...)
		if len(s.buf) >= s.maxBytes() {
			s.flush()
			s.active = true
			s.buf = make([]byte, 0, len(s.buf))
		}
		return
	}
	limit := s.prerollBytes()
	s.preroll = append(s.preroll, data...)
	if len(s.preroll) > 2*limit {
		copy(s.preroll, s.preroll[len(s.preroll)-limit:])
		s.preroll = s.preroll[:limit]
	}
}

func (s *segmenter) begin() {
	if s.active {
		return
	}
	s.active = true
	limit := s.prerollBytes()
	pre := s.preroll
	if len(pre) > limit {
		pre = pre[len(pre)-limit:]
	}
	s.buf = make([]byte, 0, len(pre)+s.bytesFor(5*time.Second))
	s.buf = append(s.buf, pre...)
	s.preroll = s.preroll[:0]
}

func (s *segmenter) end() {
	if !s.active {
		return
	}
	s.flush()
}

func (s *segmenter) flush() {
	s.active = false
	pcm := s.buf
	s.buf = nil
	if len(pcm) == 0 || s.closed {
		return
	}
	trailing := s.b.opts.TrailingSilence
	if trailing == 0 {
		trailing = DefaultTrailingSilence
	}
	if trailing > 0 {
		pcm = append(pcm, make([]byte, s.bytesFor(trailing))...)
	}
	wav := WAV(pcm, s.b.SampleRate(), 1)
	select {
	case s.queue <- wav:
	default:
		_ = s.b.PushError("transcription backlog full, dropping speech segment", errors.New("segment queue full"), false)
	}
}

func (s *segmenter) worker() {
	defer s.wg.Done()
	for wav := range s.queue {
		if s.ctx.Err() != nil {
			continue
		}
		started := time.Now()
		t, err := s.provider.Transcribe(s.ctx, wav)
		if s.b.StartConfig().EnableMetrics {
			s.b.emitMetrics(frames.ProcessingMetricsData{MetricsBase: s.b.metricsBase(), Value: time.Since(started)})
		}
		if err != nil {
			if s.ctx.Err() == nil {
				s.b.pushError("transcription failed", err, false)
			}
			continue
		}
		t.Final = true
		t.Finalized = true
		sink{s.b}.Transcript(t)
	}
}

func (s *segmenter) stop(graceful bool) {
	if s.closed {
		return
	}
	s.closed = true
	if graceful && s.active {
		s.closed = false
		s.flush()
		s.closed = true
	}
	if !graceful {
		s.cancel()
	}
	close(s.queue)
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	timer := time.NewTimer(segmentDrainTimeout)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
		s.cancel()
		<-done
	}
	s.cancel()
}

func WAVHeader(dataLen, sampleRate, channels int) []byte {
	h := make([]byte, 44)
	writeWAVHeader(h, dataLen, sampleRate, channels)
	return h
}

func WAV(pcm []byte, sampleRate, channels int) []byte {
	out := make([]byte, 44+len(pcm))
	writeWAVHeader(out, len(pcm), sampleRate, channels)
	copy(out[44:], pcm)
	return out
}

func writeWAVHeader(h []byte, dataLen, sampleRate, channels int) {
	if channels <= 0 {
		channels = 1
	}
	copy(h[0:], "RIFF")
	binary.LittleEndian.PutUint32(h[4:], uint32(36+dataLen))
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], uint16(channels))
	binary.LittleEndian.PutUint32(h[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(h[28:], uint32(sampleRate*channels*2))
	binary.LittleEndian.PutUint16(h[32:], uint16(channels*2))
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], uint32(dataLen))
}
