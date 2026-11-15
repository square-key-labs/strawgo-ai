package stt

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultSampleRate  = 16000
	DefaultTTFBTimeout = 2 * time.Second
	DefaultTTFSP99     = time.Second
)

type Transcript struct {
	Text      string
	Final     bool
	Finalized bool
	Language  string
}

type Sink interface {
	Transcript(t Transcript)
	Push(frame frames.Frame, direction frames.FrameDirection)
	Error(err error, fatal bool)
}

type StreamingProvider interface {
	Connect(ctx context.Context, sink Sink) error
	SendAudio(audio []byte)
	Disconnect(graceful bool)
}

type SegmentedProvider interface {
	Transcribe(ctx context.Context, wav []byte) (Transcript, error)
}

type Finalizer interface {
	Finalize()
}

type SettingsApplier interface {
	ApplySettings(settings frames.STTSettings) error
}

type Options struct {
	Provider        string
	Settings        frames.STTSettings
	SampleRate      int
	BytesPerSample  int
	TTFSP99         time.Duration
	TTFBTimeout     time.Duration
	DropAudio       bool
	PreRoll         time.Duration
	TrailingSilence time.Duration
	MaxSegment      time.Duration
	SegmentQueue    int
}

type Base struct {
	*processors.BaseProcessor

	opts      Options
	stream    StreamingProvider
	segmented *segmenter
	log       *logger.Logger

	settingsMu sync.RWMutex
	settings   frames.STTSettings
	sampleRate atomic.Int64

	started      bool
	stopped      bool
	userSpeaking bool
	muted        atomic.Bool

	metricsMu    sync.Mutex
	ttfbStart    time.Time
	ttfbPending  bool
	ttfbReported bool
	lastFinal    time.Time
	ttfbTimer    *time.Timer

	pendingAudioBytes atomic.Int64
}

func NewStreaming(name string, provider StreamingProvider, opts Options) *Base {
	b := newBase(name, opts)
	b.stream = provider
	return b
}

func NewSegmented(name string, provider SegmentedProvider, opts Options) *Base {
	b := newBase(name, opts)
	b.segmented = newSegmenter(b, provider)
	return b
}

func newBase(name string, opts Options) *Base {
	if opts.BytesPerSample <= 0 {
		opts.BytesPerSample = 2
	}
	if opts.TTFBTimeout <= 0 {
		opts.TTFBTimeout = DefaultTTFBTimeout
	}
	if opts.TTFSP99 <= 0 {
		opts.TTFSP99 = DefaultTTFSP99
	}
	b := &Base{
		opts:     opts,
		settings: opts.Settings,
		log:      logger.WithPrefix(name),
	}
	b.sampleRate.Store(int64(opts.SampleRate))
	b.BaseProcessor = processors.NewBaseProcessor(name, b)
	return b
}

func (b *Base) Settings() frames.STTSettings {
	b.settingsMu.RLock()
	defer b.settingsMu.RUnlock()
	return b.settings
}

func (b *Base) UpdateSettings(delta frames.STTSettings) error {
	b.settingsMu.Lock()
	b.settings = b.settings.Merge(delta)
	current := b.settings
	b.settingsMu.Unlock()
	if applier, ok := b.provider().(SettingsApplier); ok {
		return applier.ApplySettings(current)
	}
	return nil
}

func (b *Base) SetLanguage(language string) {
	if err := b.UpdateSettings(frames.STTSettings{Language: language}); err != nil {
		b.log.Warn("language update failed: %v", err)
	}
}

func (b *Base) SetModel(model string) {
	if err := b.UpdateSettings(frames.STTSettings{Model: model}); err != nil {
		b.log.Warn("model update failed: %v", err)
	}
}

func (b *Base) SampleRate() int {
	if rate := int(b.sampleRate.Load()); rate > 0 {
		return rate
	}
	return DefaultSampleRate
}

func (b *Base) Muted() bool {
	return b.muted.Load()
}

func (b *Base) provider() any {
	if b.stream != nil {
		return b.stream
	}
	if b.segmented != nil {
		return b.segmented.provider
	}
	return nil
}

func (b *Base) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.AudioFrame:
		if direction == frames.Downstream {
			b.processAudio(f)
			if b.opts.DropAudio {
				return nil
			}
		}
	case *frames.StartFrame:
		if err := b.PushFrame(f, direction); err != nil {
			return err
		}
		b.start()
		if b.opts.Provider != "" {
			return b.PushFrame(frames.NewSTTMetadataFrame(b.opts.Provider, b.opts.TTFSP99), frames.Downstream)
		}
		return nil
	case *frames.EndFrame:
		b.stop(true)
	case *frames.CancelFrame:
		b.stop(false)
	case *frames.VADUserStartedSpeakingFrame:
		b.userStarted()
	case *frames.UserStartedSpeakingFrame:
		if direction == frames.Downstream {
			b.userStarted()
		}
	case *frames.VADUserStoppedSpeakingFrame:
		end := f.Timestamp
		if end.IsZero() {
			end = time.Now()
		}
		b.userStopped(end.Add(-time.Duration(f.StopSecs * float64(time.Second))))
	case *frames.UserStoppedSpeakingFrame:
		if direction == frames.Downstream {
			b.userStopped(time.Now())
		}
	case *frames.STTMuteFrame:
		b.muted.Store(f.Mute)
		return nil
	case *frames.STTUpdateSettingsFrame:
		if f.Service != nil && f.Service.Name() != b.Name() {
			break
		}
		if err := b.UpdateSettings(f.Delta); err != nil {
			_ = b.PushError("settings update failed", err, false)
		}
		return nil
	case *frames.InterruptionFrame:
		b.cancelTTFB()
	}
	return b.PushFrame(frame, direction)
}

func (b *Base) start() {
	if b.started {
		return
	}
	b.started = true
	if b.opts.SampleRate <= 0 {
		rate := b.StartConfig().AudioInSampleRate
		if rate <= 0 {
			rate = DefaultSampleRate
		}
		b.sampleRate.Store(int64(rate))
	}
	ctx := b.Context()
	if b.stream != nil {
		if err := b.stream.Connect(ctx, sink{b}); err != nil {
			_ = b.PushError("STT connect failed", err, false)
		}
	}
	if b.segmented != nil {
		b.segmented.start(ctx)
	}
}

func (b *Base) stop(graceful bool) {
	if b.stopped {
		return
	}
	b.stopped = true
	b.cancelTTFB()
	if !b.started {
		return
	}
	if b.stream != nil {
		b.stream.Disconnect(graceful)
	}
	if b.segmented != nil {
		b.segmented.stop(graceful)
	}
}

func (b *Base) processAudio(f *frames.AudioFrame) {
	if b.stopped || len(f.Data) == 0 || b.muted.Load() {
		return
	}
	if !b.started {
		b.start()
	}
	b.pendingAudioBytes.Add(int64(len(f.Data)))
	if b.stream != nil {
		b.stream.SendAudio(f.Data)
	}
	if b.segmented != nil {
		b.segmented.append(f.Data)
	}
}

func (b *Base) userStarted() {
	if b.userSpeaking {
		return
	}
	b.userSpeaking = true
	b.cancelTTFB()
	if b.segmented != nil {
		b.segmented.begin()
	}
}

func (b *Base) userStopped(speechEnd time.Time) {
	if !b.userSpeaking {
		return
	}
	b.userSpeaking = false
	b.armTTFB(speechEnd)
	if finalizer, ok := b.stream.(Finalizer); ok {
		finalizer.Finalize()
	}
	if b.segmented != nil {
		b.segmented.end()
	}
}

func (b *Base) metricsBase() frames.MetricsBase {
	return frames.MetricsBase{Processor: b.Name(), Model: b.Settings().Model}
}

func (b *Base) armTTFB(start time.Time) {
	if !b.StartConfig().EnableMetrics {
		return
	}
	b.metricsMu.Lock()
	defer b.metricsMu.Unlock()
	if b.ttfbReported && b.StartConfig().ReportOnlyInitialTTFB {
		return
	}
	b.ttfbStart = start
	b.ttfbPending = true
	b.lastFinal = time.Time{}
	if b.ttfbTimer == nil {
		b.ttfbTimer = time.AfterFunc(b.opts.TTFBTimeout, b.ttfbTimeout)
	} else {
		b.ttfbTimer.Reset(b.opts.TTFBTimeout)
	}
}

func (b *Base) cancelTTFB() {
	b.metricsMu.Lock()
	b.ttfbPending = false
	if b.ttfbTimer != nil {
		b.ttfbTimer.Stop()
	}
	b.metricsMu.Unlock()
}

func (b *Base) ttfbTimeout() {
	b.metricsMu.Lock()
	if !b.ttfbPending {
		b.metricsMu.Unlock()
		return
	}
	b.ttfbPending = false
	start, last := b.ttfbStart, b.lastFinal
	if last.After(start) {
		b.ttfbReported = true
	}
	b.metricsMu.Unlock()
	if last.After(start) {
		b.emitMetrics(frames.TTFBMetricsData{MetricsBase: b.metricsBase(), Value: last.Sub(start)})
	}
}

func (b *Base) onFinal(t Transcript) {
	now := time.Now()
	b.metricsMu.Lock()
	if !b.ttfbPending {
		b.metricsMu.Unlock()
		return
	}
	if !t.Finalized {
		b.lastFinal = now
		b.metricsMu.Unlock()
		return
	}
	b.ttfbPending = false
	b.ttfbReported = true
	if b.ttfbTimer != nil {
		b.ttfbTimer.Stop()
	}
	start := b.ttfbStart
	b.metricsMu.Unlock()
	if now.After(start) {
		b.emitMetrics(frames.TTFBMetricsData{MetricsBase: b.metricsBase(), Value: now.Sub(start)})
	}
}

func (b *Base) emitUsage() {
	if !b.StartConfig().EnableUsageMetrics {
		return
	}
	n := b.pendingAudioBytes.Swap(0)
	if n <= 0 {
		return
	}
	seconds := float64(n) / float64(b.SampleRate()*b.opts.BytesPerSample)
	b.emitMetrics(frames.STTUsageMetricsData{MetricsBase: b.metricsBase(), Value: frames.STTUsage{AudioSeconds: seconds}})
}

func (b *Base) emitMetrics(data frames.MetricsData) {
	_ = b.PushFrame(frames.NewMetricsFrame(data), frames.Downstream)
}

type sink struct {
	b *Base
}

func (s sink) Transcript(t Transcript) {
	if t.Text == "" {
		return
	}
	if !t.Final {
		f := frames.NewInterimTranscriptionFrame(t.Text, "")
		f.Language = t.Language
		_ = s.b.PushFrame(f, frames.Downstream)
		return
	}
	f := frames.NewTranscriptionFrame(t.Text, true)
	f.Language = t.Language
	s.b.onFinal(t)
	s.b.emitUsage()
	_ = s.b.PushFrame(f, frames.Downstream)
}

func (s sink) Push(frame frames.Frame, direction frames.FrameDirection) {
	_ = s.b.PushFrame(frame, direction)
}

func (s sink) Error(err error, fatal bool) {
	_ = s.b.PushError(s.b.Name()+" error", err, fatal)
}
