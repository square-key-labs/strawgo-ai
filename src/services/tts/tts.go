package tts

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultSampleRate    = 24000
	DefaultStopTimeout   = 2 * time.Second
	DefaultChunkDuration = 40 * time.Millisecond
	endDrainTimeout      = 10 * time.Second
)

type Word struct {
	Text  string
	Start time.Duration
}

type Sink interface {
	Audio(contextID string, audio []byte)
	Words(contextID string, words []Word)
	Done(contextID string)
	Error(err error, fatal bool)
}

type StreamingProvider interface {
	Connect(ctx context.Context, sink Sink) error
	Disconnect(graceful bool)
	Speak(contextID, text string) error
	Flush(contextID string) error
	Cancel(contextID string) error
}

type Request struct {
	ContextID  string
	Text       string
	Settings   frames.TTSSettings
	SampleRate int
}

type HTTPProvider interface {
	Synthesize(ctx context.Context, req Request) (io.ReadCloser, error)
}

type SettingsApplier interface {
	ApplySettings(settings frames.TTSSettings) error
}

type Options struct {
	Settings       frames.TTSSettings
	SampleRate     int
	Codec          string
	BytesPerSample int
	Aggregation    AggregationMode
	WordTimestamps bool
	Filters        []TextFilter
	StopTimeout    time.Duration
	ChunkDuration  time.Duration
}

type Base struct {
	*processors.BaseProcessor

	opts   Options
	stream StreamingProvider
	http   HTTPProvider
	log    *logger.Logger
	pb     *playback

	settingsMu sync.RWMutex
	settings   frames.TTSSettings

	formatMu   sync.RWMutex
	sampleRate int
	codec      string

	agg      sentenceAggregator
	aggStart time.Time
	turnID   string
	started  bool
	stopped  bool

	ttfbReported atomic.Bool
}

func NewStreaming(name string, provider StreamingProvider, opts Options) *Base {
	b := newBase(name, opts)
	b.stream = provider
	return b
}

func NewHTTP(name string, provider HTTPProvider, opts Options) *Base {
	b := newBase(name, opts)
	b.http = provider
	return b
}

func newBase(name string, opts Options) *Base {
	if opts.Codec == "" {
		opts.Codec = "linear16"
	}
	if opts.BytesPerSample <= 0 {
		opts.BytesPerSample = 2
	}
	if opts.StopTimeout <= 0 {
		opts.StopTimeout = DefaultStopTimeout
	}
	if opts.ChunkDuration <= 0 {
		opts.ChunkDuration = DefaultChunkDuration
	}
	b := &Base{
		opts:       opts,
		settings:   opts.Settings,
		sampleRate: opts.SampleRate,
		codec:      opts.Codec,
		log:        logger.WithPrefix(name),
		agg:        sentenceAggregator{mode: opts.Aggregation},
	}
	b.pb = newPlayback(b, opts.StopTimeout)
	b.BaseProcessor = processors.NewBaseProcessor(name, b)
	return b
}

func NewContextID() string {
	return uuid.NewString()
}

func (b *Base) Settings() frames.TTSSettings {
	b.settingsMu.RLock()
	defer b.settingsMu.RUnlock()
	return b.settings
}

func (b *Base) UpdateSettings(delta frames.TTSSettings) error {
	b.settingsMu.Lock()
	b.settings = b.settings.Merge(delta)
	current := b.settings
	b.settingsMu.Unlock()
	var provider any = b.stream
	if b.http != nil {
		provider = b.http
	}
	if applier, ok := provider.(SettingsApplier); ok {
		return applier.ApplySettings(current)
	}
	return nil
}

func (b *Base) SetVoice(voice string) {
	b.applyDelta(frames.TTSSettings{Voice: voice})
}

func (b *Base) SetModel(model string) {
	b.applyDelta(frames.TTSSettings{Model: model})
}

func (b *Base) SetLanguage(language string) {
	b.applyDelta(frames.TTSSettings{Language: language})
}

func (b *Base) applyDelta(delta frames.TTSSettings) {
	if err := b.UpdateSettings(delta); err != nil {
		b.log.Warn("settings update failed: %v", err)
	}
}

func (b *Base) SetAudioFormat(sampleRate int, codec string) {
	b.formatMu.Lock()
	if sampleRate > 0 {
		b.sampleRate = sampleRate
	}
	if codec != "" {
		b.codec = codec
	}
	b.formatMu.Unlock()
}

func (b *Base) SampleRate() int {
	b.formatMu.RLock()
	defer b.formatMu.RUnlock()
	if b.sampleRate > 0 {
		return b.sampleRate
	}
	return DefaultSampleRate
}

func (b *Base) Codec() string {
	b.formatMu.RLock()
	defer b.formatMu.RUnlock()
	return b.codec
}

func (b *Base) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Upstream {
		return b.PushFrame(frame, direction)
	}
	switch f := frame.(type) {
	case *frames.StartFrame:
		if err := b.PushFrame(f, direction); err != nil {
			return err
		}
		b.start()
		return nil
	case *frames.InterruptionFrame:
		b.interrupt()
		return b.PushFrame(f, direction)
	case *frames.CancelFrame:
		b.stop(false)
		return b.PushFrame(f, direction)
	case *frames.EndFrame:
		b.finishTurn(ctx)
		b.pb.waitIdle(ctx, endDrainTimeout)
		b.stop(true)
		return b.pb.pass(f)
	case *frames.LLMFullResponseStartFrame:
		b.finishTurn(ctx)
		b.turnID = NewContextID()
		return b.pb.pass(f)
	case *frames.LLMFullResponseEndFrame:
		b.finishTurn(ctx)
		return b.pb.pass(f)
	case *frames.LLMTextFrame:
		if f.SkipTTS {
			return b.pb.pass(f)
		}
		b.aggregate(ctx, f.Text)
		return nil
	case *frames.TextFrame:
		if f.SkipTTS {
			return b.pb.pass(f)
		}
		if b.turnID == "" {
			b.speakAlone(ctx, f.Text)
			return nil
		}
		b.aggregate(ctx, f.Text)
		return nil
	case *frames.AggregatedTextFrame:
		if f.SkipTTS {
			return b.pb.pass(f)
		}
		if b.turnID == "" {
			b.speakAlone(ctx, f.Text)
			return nil
		}
		b.speak(ctx, b.turnID, f.Text)
		return nil
	case *frames.TTSSpeakFrame:
		b.speakAlone(ctx, f.Text)
		return nil
	case *frames.TTSUpdateSettingsFrame:
		if f.Service != nil && f.Service.Name() != b.Name() {
			return b.pb.pass(f)
		}
		if err := b.UpdateSettings(f.Delta); err != nil {
			b.pushError("settings update failed", err, false)
		}
		return nil
	}
	if frames.IsSystem(frame) {
		return b.PushFrame(frame, direction)
	}
	return b.pb.pass(frame)
}

func (b *Base) start() {
	if b.started || b.stopped {
		return
	}
	b.started = true
	if b.opts.SampleRate <= 0 {
		b.formatMu.Lock()
		if b.sampleRate <= 0 {
			if rate := b.StartConfig().AudioOutSampleRate; rate > 0 {
				b.sampleRate = rate
			} else {
				b.sampleRate = DefaultSampleRate
			}
		}
		b.formatMu.Unlock()
	}
	ctx := b.Context()
	go b.pb.run(ctx)
	if b.stream != nil {
		if err := b.stream.Connect(ctx, sink{b}); err != nil {
			b.pushError("TTS connect failed", err, false)
		}
	}
}

func (b *Base) stop(graceful bool) {
	if b.stopped {
		return
	}
	b.stopped = true
	if !graceful {
		b.pb.interrupt()
	}
	if b.started && b.stream != nil {
		b.stream.Disconnect(graceful)
	}
}

func (b *Base) interrupt() {
	cancelled, active := b.pb.interrupt()
	b.agg.reset()
	b.aggStart = time.Time{}
	b.turnID = ""
	for _, f := range b.opts.Filters {
		f.Reset()
	}
	if b.stream != nil {
		for _, id := range cancelled {
			if err := b.stream.Cancel(id); err != nil {
				b.log.Debug("cancel %s: %v", id, err)
			}
		}
	}
	if active {
		_ = b.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
	}
}

func (b *Base) aggregate(ctx context.Context, text string) {
	if b.turnID == "" {
		b.turnID = NewContextID()
	}
	id := b.turnID
	if b.aggStart.IsZero() && b.opts.Aggregation != AggregateTokens {
		b.aggStart = time.Now()
	}
	b.agg.add(text, func(sentence string) {
		b.stopTextAggregation()
		b.speak(ctx, id, sentence)
	})
}

func (b *Base) stopTextAggregation() {
	if b.aggStart.IsZero() {
		return
	}
	elapsed := time.Since(b.aggStart)
	b.aggStart = time.Time{}
	if b.StartConfig().EnableMetrics {
		b.emitMetrics(frames.TextAggregationMetricsData{MetricsBase: b.metricsBase(), Value: elapsed})
	}
}

func (b *Base) finishTurn(ctx context.Context) {
	id := b.turnID
	if id == "" {
		return
	}
	rest := b.agg.flush()
	b.stopTextAggregation()
	if rest != "" {
		b.speak(ctx, id, rest)
	}
	b.closeContext(id)
	b.turnID = ""
	for _, f := range b.opts.Filters {
		f.Reset()
	}
}

func (b *Base) speakAlone(ctx context.Context, text string) {
	id := NewContextID()
	b.speak(ctx, id, text)
	b.closeContext(id)
}

func (b *Base) closeContext(id string) {
	_ = b.PushFrame(frames.NewTTSQueuedTextCompleteFrame(id), frames.Downstream)
	if !b.pb.isLive(id) {
		return
	}
	if b.http != nil {
		b.pb.markDone(id)
		return
	}
	b.pb.markFlushed(id)
	if err := b.stream.Flush(id); err != nil {
		b.log.Warn("flush %s: %v", id, err)
		b.pb.markDone(id)
	}
}

func (b *Base) prepare(text string) string {
	for _, f := range b.opts.Filters {
		text = f.Filter(text)
	}
	if b.opts.Aggregation == AggregateTokens {
		return text
	}
	return strings.TrimSpace(text)
}

func (b *Base) speak(ctx context.Context, id, text string) {
	if b.stopped {
		return
	}
	spokenText := text
	if b.opts.Aggregation != AggregateTokens {
		spokenText = strings.TrimSpace(text)
	}
	text = b.prepare(text)
	if text == "" || (strings.TrimSpace(text) == "" && !b.pb.isLive(id)) {
		return
	}
	if !b.started {
		b.start()
	}
	if b.pb.open(id) {
		started := frames.NewTTSStartedFrameWithContext(id)
		_ = b.PushFrame(started, frames.Upstream)
	}
	textType := frames.TextTypeSentence
	if b.opts.Aggregation == AggregateTokens {
		textType = frames.TextTypeToken
	}
	queued := frames.NewTTSQueuedTextFrame(spokenText, textType, id)
	queued.PlaybackTracked = b.http != nil || b.opts.WordTimestamps
	_ = b.PushFrame(queued, frames.Downstream)
	cfg := b.StartConfig()
	if cfg.EnableUsageMetrics {
		b.emitMetrics(frames.TTSUsageMetricsData{MetricsBase: b.metricsBase(), Value: utf8.RuneCountInString(text)})
	}
	if b.http != nil {
		began := time.Now()
		b.synthesizeHTTP(ctx, id, text)
		if cfg.EnableMetrics {
			b.emitMetrics(frames.ProcessingMetricsData{MetricsBase: b.metricsBase(), Value: time.Since(began)})
		}
	} else if err := b.stream.Speak(id, text); err != nil {
		b.pushError("TTS request failed", err, false)
		return
	}
	if !b.opts.WordTimestamps {
		spoken := frames.NewTTSTextFrame(spokenText, textType, id)
		spoken.WillBeSpoken = true
		b.pb.addFrame(id, spoken)
	}
}

func (b *Base) chunkBytes() int {
	n := int(b.opts.ChunkDuration.Seconds()*float64(b.SampleRate())) * b.opts.BytesPerSample
	if n < 320 {
		n = 320
	}
	return n - n%2
}

func (b *Base) synthesizeHTTP(ctx context.Context, id, text string) {
	body, err := b.http.Synthesize(ctx, Request{ContextID: id, Text: text, Settings: b.Settings(), SampleRate: b.SampleRate()})
	if err != nil {
		if ctx.Err() == nil {
			b.pushError("TTS request failed", err, false)
		}
		return
	}
	defer body.Close()
	size := b.chunkBytes()
	for {
		buf := make([]byte, size)
		n, err := io.ReadFull(body, buf)
		if n > 0 && !b.pb.addAudio(id, buf[:n]) {
			return
		}
		if err != nil {
			if !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) && ctx.Err() == nil {
				b.pushError("TTS stream failed", err, false)
			}
			return
		}
	}
}

func (b *Base) audioFrame(id string, data []byte) *frames.TTSAudioFrame {
	f := frames.NewTTSAudioFrame(data, b.SampleRate(), 1)
	f.ContextID = id
	f.SetMetadata("codec", b.Codec())
	f.SetMetadata("context_id", id)
	return f
}

func (b *Base) metricsBase() frames.MetricsBase {
	return frames.MetricsBase{Processor: b.Name(), Model: b.Settings().Model}
}

func (b *Base) emitMetrics(data frames.MetricsData) {
	_ = b.PushFrame(frames.NewMetricsFrame(data), frames.Downstream)
}

func (b *Base) claimTTFB() bool {
	cfg := b.StartConfig()
	if !cfg.EnableMetrics {
		return false
	}
	return !cfg.ReportOnlyInitialTTFB || b.ttfbReported.CompareAndSwap(false, true)
}

func (b *Base) measuresTTFA() bool {
	return b.opts.BytesPerSample == 2 && audio.NormalizeCodec(b.Codec()) == audio.CodecLinear16
}

type sink struct {
	b *Base
}

func (s sink) Audio(contextID string, audio []byte) {
	if len(audio) == 0 {
		return
	}
	s.b.pb.addAudio(contextID, audio)
}

func (s sink) Words(contextID string, words []Word) {
	for _, w := range words {
		f := frames.NewTTSTextFrame(w.Text, frames.TextTypeWord, contextID)
		f.WillBeSpoken = true
		f.SetMetadata("word_start_time", w.Start.Seconds())
		if !s.b.pb.addFrame(contextID, f) {
			return
		}
	}
}

func (s sink) Done(contextID string) {
	s.b.pb.markDone(contextID)
}

func (s sink) Error(err error, fatal bool) {
	s.b.pushError(s.b.Name()+" error", err, fatal)
}

func (b *Base) pushError(msg string, err error, fatal bool) {
	f := frames.NewCategorizedErrorFrame(fmt.Errorf("%s: %w", msg, err), frames.ClassifyError(err))
	f.Fatal = fatal
	_ = b.PushErrorFrame(f)
}
