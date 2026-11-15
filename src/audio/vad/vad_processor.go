package vad

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultSpeechActivityPeriod = 200 * time.Millisecond
	DefaultAudioIdleTimeout     = time.Second
	fallbackVADSampleRate       = 16000
)

type Option func(*VADInputProcessor)

func WithTurnAnalyzer(a turn.TurnAnalyzer) Option {
	return func(p *VADInputProcessor) { p.turnAnalyzer = a }
}

func WithSpeechActivityPeriod(d time.Duration) Option {
	return func(p *VADInputProcessor) { p.speechActivityPeriod = d }
}

func WithAudioIdleTimeout(d time.Duration) Option {
	return func(p *VADInputProcessor) { p.audioIdleTimeout = d }
}

type turnParamsProvider interface {
	Params() turn.TurnAnalyzerParams
}

type audioIdleFrame struct {
	*frames.SystemFrame
}

func newAudioIdleFrame() *audioIdleFrame {
	return &audioIdleFrame{SystemFrame: &frames.SystemFrame{BaseFrame: frames.NewBaseFrame("VADAudioIdleFrame")}}
}

type VADInputProcessor struct {
	*processors.BaseProcessor
	analyzer     VADAnalyzer
	turnAnalyzer turn.TurnAnalyzer

	speechActivityPeriod time.Duration
	audioIdleTimeout     time.Duration

	inRate    int
	vadRate   int
	resampler *resample.Resampler
	pending   []byte
	resampled []byte

	currentState atomic.Int32
	vadState     VADState
	lastAudio    time.Time
	lastActivity time.Time
	idleTimer    *time.Timer

	turnReq    chan struct{}
	turnGen    atomic.Uint64
	workerOnce sync.Once
	turnMu     sync.Mutex
	turnCancel context.CancelFunc
}

func NewVADInputProcessor(analyzer VADAnalyzer, opts ...Option) *VADInputProcessor {
	p := &VADInputProcessor{
		analyzer:             analyzer,
		speechActivityPeriod: DefaultSpeechActivityPeriod,
		audioIdleTimeout:     DefaultAudioIdleTimeout,
		vadState:             VADStateQuiet,
		turnReq:              make(chan struct{}, 1),
	}
	p.currentState.Store(int32(VADStateQuiet))
	for _, o := range opts {
		o(p)
	}
	p.BaseProcessor = processors.NewBaseProcessor("VADInput", p)
	return p
}

func NewVADInputProcessorWithTurn(analyzer VADAnalyzer, turnAnalyzer turn.TurnAnalyzer, opts ...Option) *VADInputProcessor {
	return NewVADInputProcessor(analyzer, append(opts, WithTurnAnalyzer(turnAnalyzer))...)
}

func (p *VADInputProcessor) GetCurrentState() VADState {
	return VADState(p.currentState.Load())
}

func (p *VADInputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *audioIdleFrame:
		p.handleAudioIdle()
		return nil
	case *frames.AudioFrame:
		if err := p.PushFrame(f, direction); err != nil {
			return err
		}
		p.handleAudio(f)
		return nil
	case *frames.StartFrame:
		if err := p.PushFrame(f, direction); err != nil {
			return err
		}
		p.handleStart(f)
		return nil
	case *frames.VADParamsUpdateFrame:
		p.updateParams(ParamsFromFrame(f.Params))
	case *frames.EndFrame:
		p.stop()
	case *frames.CancelFrame:
		p.stop()
	}
	return p.PushFrame(frame, direction)
}

func (p *VADInputProcessor) handleStart(f *frames.StartFrame) {
	rate := f.AudioInSampleRate
	if rate <= 0 {
		if r, ok := f.Metadata()["sampleRate"].(int); ok {
			rate = r
		}
	}
	if rate > 0 {
		p.configureRate(rate)
	}
	p.lastAudio = time.Now()
	if ps, ok := p.analyzer.(ParamsSetter); ok {
		params := ps.GetParams()
		p.setTurnStartSecs(params)
		p.broadcastParams(&params)
	} else if p.turnAnalyzer != nil {
		p.broadcastParams(nil)
	}
	if p.turnAnalyzer != nil {
		p.workerOnce.Do(func() { go p.turnWorker(p.Context()) })
	}
}

func (p *VADInputProcessor) updateParams(params VADParams) {
	ps, ok := p.analyzer.(ParamsSetter)
	if !ok {
		logger.Warn("[VADInput] analyzer does not support parameter updates")
		return
	}
	params = ps.GetParams().Merge(params)
	ps.SetParams(params)
	p.setTurnStartSecs(params)
	p.broadcastParams(&params)
}

func (p *VADInputProcessor) setTurnStartSecs(params VADParams) {
	if s, ok := p.turnAnalyzer.(interface{ SetVADStartSecs(float64) }); ok {
		s.SetVADStartSecs(float64(params.StartSecs))
	}
}

func (p *VADInputProcessor) broadcastParams(params *VADParams) {
	var turnParams any
	if p.turnAnalyzer != nil {
		turnParams = p.turnParams()
	}
	_ = p.BroadcastFrame(p.Context(), func() frames.Frame {
		var fp *frames.VADParams
		if params != nil {
			v := params.ToFrameParams()
			fp = &v
		}
		return frames.NewSpeechControlParamsFrame(fp, turnParams)
	})
}

func (p *VADInputProcessor) turnParams() turn.TurnAnalyzerParams {
	if tp, ok := p.turnAnalyzer.(turnParamsProvider); ok {
		return tp.Params()
	}
	return turn.TurnAnalyzerParams{}
}

func (p *VADInputProcessor) configureRate(rate int) {
	if rate == p.inRate {
		return
	}
	p.inRate = rate
	p.resampler = nil
	p.pending = p.pending[:0]
	vadRate := rate
	if err := p.analyzer.SetSampleRate(rate); err != nil {
		vadRate = fallbackVADSampleRate
		if err := p.analyzer.SetSampleRate(vadRate); err != nil {
			logger.Error("[VADInput] cannot configure VAD sample rate: %v", err)
			return
		}
		r, err := resample.New(rate, vadRate)
		if err != nil {
			logger.Error("[VADInput] resampler %d->%d: %v", rate, vadRate, err)
			return
		}
		p.resampler = r
		logger.Info("[VADInput] resampling %d Hz -> %d Hz for VAD", rate, vadRate)
	}
	p.vadRate = vadRate
	if p.turnAnalyzer != nil {
		p.turnAnalyzer.SetSampleRate(vadRate)
	}
}

func (p *VADInputProcessor) cancelTurnAnalysis() {
	p.turnMu.Lock()
	if p.turnCancel != nil {
		p.turnCancel()
	}
	p.turnMu.Unlock()
}

func (p *VADInputProcessor) resetTurn() bool {
	if p.turnAnalyzer == nil {
		return false
	}
	open := p.turnAnalyzer.SpeechTriggered()
	p.turnGen.Add(1)
	p.cancelTurnAnalysis()
	p.turnAnalyzer.Clear()
	return open
}

func (p *VADInputProcessor) stop() {
	if p.idleTimer != nil {
		p.idleTimer.Stop()
	}
	p.resetTurn()
	p.analyzer.Restart()
	p.pending = p.pending[:0]
	p.vadState = VADStateQuiet
	p.currentState.Store(int32(VADStateQuiet))
}

func (p *VADInputProcessor) handleAudio(f *frames.AudioFrame) {
	if f.SampleRate > 0 && f.SampleRate != p.inRate {
		p.configureRate(f.SampleRate)
	}
	p.lastAudio = time.Now()

	data := f.Data
	if p.resampler != nil {
		p.resampled = p.resampler.ProcessBytes(p.resampled[:0], data)
		data = p.resampled
	}
	p.pending = append(p.pending, data...)

	required := p.analyzer.NumFramesRequired() * 2
	off := 0
	for len(p.pending)-off >= required {
		chunk := p.pending[off : off+required]
		off += required
		newState, err := p.analyzer.AnalyzeAudio(chunk)
		if err != nil {
			logger.Error("[VADInput] VAD analysis error: %v", err)
			continue
		}
		prev := VADState(p.currentState.Swap(int32(newState)))
		p.handleTransition(chunk, prev, newState)
	}
	if off > 0 {
		n := copy(p.pending, p.pending[off:])
		p.pending = p.pending[:n]
	}

	if p.vadState == VADStateSpeaking {
		if p.idleTimer != nil {
			p.idleTimer.Reset(p.audioIdleTimeout)
		}
		p.maybeSpeechActivity()
	}
}

func (p *VADInputProcessor) handleTransition(chunk []byte, prev, cur VADState) {
	started := (prev == VADStateQuiet || prev == VADStateStarting) && cur == VADStateSpeaking
	stopped := (prev == VADStateSpeaking || prev == VADStateStopping) && cur == VADStateQuiet

	if p.turnAnalyzer != nil {
		isSpeech := cur == VADStateSpeaking || cur == VADStateStopping
		turnState := p.turnAnalyzer.AppendAudio(chunk, isSpeech)
		if started {
			p.turnGen.Add(1)
			p.cancelTurnAnalysis()
			p.push(frames.NewUserStartedSpeakingFrame())
		}
		if turnState == turn.TurnComplete {
			p.turnGen.Add(1)
			logger.Info("[VADInput] Turn complete (silence timeout)")
			p.push(frames.NewUserStoppedSpeakingFrame())
		} else if stopped && p.turnAnalyzer.SpeechTriggered() {
			select {
			case p.turnReq <- struct{}{}:
			default:
			}
		}
	} else {
		if started {
			p.push(frames.NewUserStartedSpeakingFrame())
		}
		if stopped {
			p.push(frames.NewUserStoppedSpeakingFrame())
		}
	}

	if cur == p.vadState || cur == VADStateStarting || cur == VADStateStopping {
		return
	}
	p.vadState = cur
	params := p.currentParams()
	if cur == VADStateSpeaking {
		p.armIdleTimer()
		startSecs := float64(params.StartSecs)
		_ = p.BroadcastFrame(p.Context(), func() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(startSecs) })
		return
	}
	if p.idleTimer != nil {
		p.idleTimer.Stop()
	}
	stopSecs := float64(params.StopSecs)
	_ = p.BroadcastFrame(p.Context(), func() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(stopSecs) })
}

func (p *VADInputProcessor) currentParams() VADParams {
	if ps, ok := p.analyzer.(ParamsSetter); ok {
		return ps.GetParams()
	}
	return DefaultVADParams()
}

func (p *VADInputProcessor) maybeSpeechActivity() {
	now := time.Now()
	if now.Sub(p.lastActivity) < p.speechActivityPeriod {
		return
	}
	p.lastActivity = now
	_ = p.BroadcastFrame(p.Context(), func() frames.Frame { return frames.NewUserSpeakingFrame() })
}

func (p *VADInputProcessor) armIdleTimer() {
	if p.audioIdleTimeout <= 0 {
		return
	}
	if p.idleTimer == nil {
		p.idleTimer = time.AfterFunc(p.audioIdleTimeout, func() {
			_ = p.QueueFrame(newAudioIdleFrame(), frames.Downstream)
		})
		return
	}
	p.idleTimer.Reset(p.audioIdleTimeout)
}

func (p *VADInputProcessor) handleAudioIdle() {
	if p.vadState != VADStateSpeaking || p.audioIdleTimeout <= 0 {
		return
	}
	if idle := time.Since(p.lastAudio); idle < p.audioIdleTimeout {
		p.idleTimer.Reset(p.audioIdleTimeout - idle)
		return
	}
	logger.Warn("[VADInput] no audio received while speaking, forcing speech stop")
	p.vadState = VADStateQuiet
	p.currentState.Store(int32(VADStateQuiet))
	p.analyzer.Restart()
	if p.turnAnalyzer == nil || p.resetTurn() {
		p.push(frames.NewUserStoppedSpeakingFrame())
	}
	stopSecs := float64(p.currentParams().StopSecs)
	_ = p.BroadcastFrame(p.Context(), func() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(stopSecs) })
}

func (p *VADInputProcessor) push(f frames.Frame) {
	if err := p.PushFrame(f, frames.Downstream); err != nil {
		logger.Error("[VADInput] push %s: %v", f.Name(), err)
	}
}

func (p *VADInputProcessor) userSpeaking() bool {
	state := p.GetCurrentState()
	return state == VADStateSpeaking || state == VADStateStopping
}

func (p *VADInputProcessor) turnWorker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-p.turnReq:
		}
		gen := p.turnGen.Load()
		if p.userSpeaking() {
			continue
		}
		actx, cancel := context.WithCancel(ctx)
		p.turnMu.Lock()
		p.turnCancel = cancel
		p.turnMu.Unlock()
		if p.turnGen.Load() != gen || p.userSpeaking() {
			cancel()
		}
		state, metrics, err := turn.AnalyzeEndOfTurn(actx, p.turnAnalyzer)
		cancelled := actx.Err() != nil
		p.turnMu.Lock()
		p.turnCancel = nil
		p.turnMu.Unlock()
		cancel()
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			if cancelled {
				logger.Debug("[VADInput] end-of-turn analysis cancelled: user resumed or turn was reset")
				continue
			}
			logger.Error("[VADInput] Turn analysis error, ending the turn at this pause: %v", err)
			state, metrics = turn.TurnComplete, nil
		}
		if metrics != nil {
			p.push(frames.NewMetricsFrame(frames.TurnMetricsData{
				MetricsBase:       frames.MetricsBase{Processor: p.Name()},
				IsComplete:        metrics.IsComplete,
				Probability:       metrics.Probability,
				E2EProcessingTime: time.Duration(metrics.TotalTimeMs * float64(time.Millisecond)),
			}))
		}
		if state != turn.TurnComplete {
			continue
		}
		if p.userSpeaking() || !p.turnGen.CompareAndSwap(gen, gen+1) {
			logger.Debug("[VADInput] discarding end-of-turn result: user resumed or turn already ended")
			continue
		}
		p.push(frames.NewUserStoppedSpeakingFrame())
	}
}
