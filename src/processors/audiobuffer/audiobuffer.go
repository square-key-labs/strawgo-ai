package audiobuffer

import (
	"context"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/audio/wav"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	silenceGapThreshold     = 200 * time.Millisecond
	defaultMaxBufferSecs    = 300
	defaultMaxTurnAudioSecs = 120
	userPreRollSecs         = 1
)

type TurnAudio struct {
	TurnNumber  int
	Audio       []byte
	SampleRate  int
	NumChannels int
}

type Config struct {
	SampleRate         int
	NumChannels        int
	BufferSize         int
	MaxBufferSize      int
	MaxTurnAudioSize   int
	EnableTurnAudio    bool
	AutoStartRecording bool
}

type Handlers struct {
	OnAudioData        func(audio []byte, sampleRate, numChannels int)
	OnTrackAudioData   func(user, bot []byte, sampleRate, numChannels int)
	OnUserTurnAudio    func(TurnAudio)
	OnBotTurnAudio     func(TurnAudio)
	OnRecordingStarted func()
	OnRecordingStopped func()
}

type track struct {
	buf        []byte
	last       time.Time
	rs         *resample.Resampler
	mono       []byte
	resampled  []byte
	run        []byte
	turn       []byte
	speaking   bool
	needsReset bool
}

type Processor struct {
	*processors.BaseProcessor
	cfg Config
	h   Handlers

	mu         sync.Mutex
	sampleRate int
	recording  bool
	user, bot  track
	turnNumber int
	pending    []func()
}

func New(cfg Config, h Handlers) *Processor {
	if cfg.NumChannels != 2 {
		cfg.NumChannels = 1
	}
	p := &Processor{cfg: cfg, h: h, sampleRate: cfg.SampleRate}
	p.BaseProcessor = processors.NewBaseProcessor("AudioBufferProcessor", p)
	return p
}

func (p *Processor) SampleRate() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.sampleRate
}

func (p *Processor) NumChannels() int { return p.cfg.NumChannels }

func (p *Processor) HasAudio() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.user.buf) > 0 || len(p.bot.buf) > 0
}

func (p *Processor) Recording() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.recording
}

func (p *Processor) StartRecording() {
	p.mu.Lock()
	p.startLocked()
	p.mu.Unlock()
	p.runPending()
}

func (p *Processor) StopRecording() {
	p.mu.Lock()
	p.stopLocked()
	p.mu.Unlock()
	p.runPending()
}

func (p *Processor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	p.mu.Lock()
	switch f := frame.(type) {
	case *frames.StartFrame:
		if p.sampleRate == 0 {
			p.sampleRate = f.AudioOutSampleRate
		}
		if p.cfg.AutoStartRecording {
			p.startLocked()
		}
	case *frames.AudioBufferStartRecordingFrame:
		p.startLocked()
	case *frames.AudioBufferStopRecordingFrame:
		p.stopLocked()
	}
	if p.recording {
		p.processLocked(frame)
	}
	switch frame.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		p.stopLocked()
	}
	p.mu.Unlock()
	p.runPending()
	return p.PushFrame(frame, direction)
}

func (p *Processor) runPending() {
	p.mu.Lock()
	fns := p.pending
	p.pending = nil
	p.mu.Unlock()
	for _, fn := range fns {
		fn()
	}
}

func (p *Processor) emit(fn func()) {
	p.pending = append(p.pending, fn)
}

func (p *Processor) startLocked() {
	if p.recording {
		return
	}
	if p.sampleRate == 0 {
		logger.Warn("[AudioBufferProcessor] sample rate unknown; recording starts once a StartFrame or audio arrives")
	}
	p.recording = true
	p.resetLocked(true)
	p.user.needsReset = true
	p.bot.needsReset = true
	if h := p.h.OnRecordingStarted; h != nil {
		p.emit(h)
	}
}

func (p *Processor) stopLocked() {
	if !p.recording {
		return
	}
	p.flushLocked()
	if p.cfg.EnableTurnAudio {
		p.foldRun(&p.user)
		p.foldRun(&p.bot)
		p.emitTurnLocked()
	}
	p.resetLocked(false)
	p.recording = false
	if h := p.h.OnRecordingStopped; h != nil {
		p.emit(h)
	}
}

func (p *Processor) resetLocked(clearTurn bool) {
	p.user.buf, p.bot.buf = nil, nil
	p.user.last, p.bot.last = time.Time{}, time.Time{}
	if clearTurn {
		p.user.run, p.user.turn = nil, nil
		p.bot.run, p.bot.turn = nil, nil
	}
}

func (p *Processor) flushThreshold() int {
	if p.cfg.BufferSize > 0 {
		return p.cfg.BufferSize
	}
	if p.cfg.MaxBufferSize > 0 {
		return p.cfg.MaxBufferSize
	}
	return defaultMaxBufferSecs * max(p.sampleRate, 8000) * 2
}

func (p *Processor) maxTurnBytes() int {
	if p.cfg.MaxTurnAudioSize > 0 {
		return p.cfg.MaxTurnAudioSize
	}
	return defaultMaxTurnAudioSecs * max(p.sampleRate, 8000) * 2
}

func (p *Processor) processLocked(frame frames.Frame) {
	var resampled []byte
	isUser := false
	switch f := frame.(type) {
	case *frames.UserStartedSpeakingFrame:
		if p.cfg.EnableTurnAudio && (len(p.bot.turn) > 0 || len(p.bot.run) > 0) {
			p.foldRun(&p.bot)
			p.emitTurnLocked()
		}
		p.user.speaking = true
	case *frames.UserStoppedSpeakingFrame:
		p.user.speaking = false
		if p.cfg.EnableTurnAudio {
			p.foldRun(&p.user)
		}
	case *frames.BotStartedSpeakingFrame:
		p.bot.speaking = true
		p.bot.needsReset = true
	case *frames.BotStoppedSpeakingFrame:
		p.bot.speaking = false
		if p.cfg.EnableTurnAudio {
			p.foldRun(&p.bot)
		}
	case *frames.UserMuteStoppedFrame:
		p.user.needsReset = true
	case *frames.AudioFrame:
		resampled = p.resampleLocked(&p.user, f.Data, f.SampleRate, f.Channels)
		isUser = true
		if len(resampled) > 0 {
			now := time.Now()
			p.fillGapLocked(&p.user, now, len(resampled))
			p.user.last = now
			if !p.bot.speaking {
				p.bot.buf = syncTo(p.bot.buf, len(p.user.buf))
				p.bot.last = now
			}
			p.user.buf = append(p.user.buf, resampled...)
		}
	case *frames.TTSAudioFrame:
		resampled = p.resampleLocked(&p.bot, f.Data, f.SampleRate, f.Channels)
		if len(resampled) > 0 {
			now := time.Now()
			p.fillGapLocked(&p.bot, now, len(resampled))
			p.bot.last = now
			if !p.user.speaking {
				p.user.buf = syncTo(p.user.buf, len(p.bot.buf))
				p.user.last = now
			}
			p.bot.buf = append(p.bot.buf, resampled...)
		}
	}

	if th := p.flushThreshold(); len(p.user.buf) >= th || len(p.bot.buf) >= th {
		p.flushLocked()
		now := time.Now()
		p.user.last, p.bot.last = now, now
	}

	if p.cfg.EnableTurnAudio && len(resampled) > 0 {
		limit := p.maxTurnBytes()
		if isUser {
			p.user.run = appendCapped(p.user.run, resampled, limit)
			preRoll := userPreRollSecs * p.sampleRate * 2
			if !p.user.speaking && len(p.user.run) > 2*preRoll {
				n := copy(p.user.run, p.user.run[len(p.user.run)-preRoll:])
				p.user.run = p.user.run[:n]
			}
		} else {
			p.bot.run = appendCapped(p.bot.run, resampled, limit)
		}
	}
}

func appendCapped(dst, src []byte, limit int) []byte {
	room := limit - len(dst)
	if room <= 0 {
		return dst
	}
	if len(src) > room {
		src = src[:room&^1]
	}
	return append(dst, src...)
}

func (p *Processor) resampleLocked(t *track, data []byte, rate, channels int) []byte {
	if len(data) < 2 {
		return nil
	}
	if p.sampleRate == 0 {
		p.sampleRate = rate
	}
	if channels > 1 {
		t.mono = pcm.ToMono(t.mono[:0], data, channels)
		data = t.mono
	}
	if rate <= 0 || rate == p.sampleRate {
		return data
	}
	if t.rs == nil || t.rs.InRate() != rate || t.rs.OutRate() != p.sampleRate {
		r, err := resample.New(rate, p.sampleRate)
		if err != nil {
			logger.Error("[AudioBufferProcessor] resampler %d->%d: %v", rate, p.sampleRate, err)
			return nil
		}
		t.rs = r
		t.needsReset = false
	}
	if t.needsReset {
		t.rs.Reset()
		t.needsReset = false
	}
	t.resampled = t.rs.ProcessBytes(t.resampled[:0], data)
	return t.resampled
}

func (p *Processor) fillGapLocked(t *track, now time.Time, frameBytes int) {
	if t.last.IsZero() || p.sampleRate == 0 {
		return
	}
	frameDur := time.Duration(float64(frameBytes) / float64(p.sampleRate*2) * float64(time.Second))
	gap := now.Sub(t.last) - frameDur
	if gap <= silenceGapThreshold {
		return
	}
	n := int(gap.Seconds()*float64(p.sampleRate)) * 2
	n = min(n, p.flushThreshold())
	t.buf = pcm.AppendSilence(t.buf, n)
}

func syncTo(buf []byte, target int) []byte {
	if len(buf) >= target {
		return buf
	}
	return pcm.AppendSilence(buf, target-len(buf))
}

func (p *Processor) flushLocked() {
	if len(p.user.buf) == 0 && len(p.bot.buf) == 0 {
		return
	}
	target := max(len(p.user.buf), len(p.bot.buf))
	user := syncTo(p.user.buf, target)
	bot := syncTo(p.bot.buf, target)
	p.user.buf, p.bot.buf = nil, nil
	rate, ch := p.sampleRate, p.cfg.NumChannels
	if h := p.h.OnAudioData; h != nil {
		var merged []byte
		if ch == 2 {
			merged = pcm.Interleave(make([]byte, 0, 2*target), user, bot)
		} else {
			merged = pcm.Mix(make([]byte, 0, target), user, bot)
		}
		p.emit(func() { h(merged, rate, ch) })
	}
	if h := p.h.OnTrackAudioData; h != nil {
		p.emit(func() { h(user, bot, rate, ch) })
	}
}

func (p *Processor) foldRun(t *track) {
	if len(t.run) == 0 {
		return
	}
	t.turn = appendCapped(t.turn, t.run, p.maxTurnBytes())
	t.run = t.run[:0]
}

func (p *Processor) emitTurnLocked() {
	n := p.turnNumber
	p.turnNumber++
	rate := p.sampleRate
	if len(p.user.turn) > 0 {
		if h := p.h.OnUserTurnAudio; h != nil {
			ta := TurnAudio{TurnNumber: n, Audio: p.user.turn, SampleRate: rate, NumChannels: 1}
			p.emit(func() { h(ta) })
		}
		p.user.turn = nil
	}
	if len(p.bot.turn) > 0 {
		if h := p.h.OnBotTurnAudio; h != nil {
			ta := TurnAudio{TurnNumber: n, Audio: p.bot.turn, SampleRate: rate, NumChannels: 1}
			p.emit(func() { h(ta) })
		}
		p.bot.turn = nil
	}
}

func EncodeWAV(audio []byte, sampleRate, numChannels int) []byte {
	return wav.Encode(audio, sampleRate, numChannels)
}

func WriteWAVFile(path string, audio []byte, sampleRate, numChannels int) error {
	return wav.WriteFile(path, audio, sampleRate, numChannels)
}
