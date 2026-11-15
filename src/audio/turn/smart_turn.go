package turn

import (
	"context"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const (
	DefaultBreakerFailureThreshold = 3
	DefaultBreakerCooldown         = 5 * time.Second
	DefaultBreakerMaxCooldown      = time.Minute
)

type SmartTurnParams struct {
	StopSecs        float64
	PreSpeechMs     float64
	MaxDurationSecs float64
}

func DefaultSmartTurnParams() *SmartTurnParams {
	return &SmartTurnParams{
		StopSecs:        3.0,
		PreSpeechMs:     500,
		MaxDurationSecs: 8.0,
	}
}

func (p *SmartTurnParams) baseParams() *TurnAnalyzerParams {
	return &TurnAnalyzerParams{
		StopSecs:        p.StopSecs,
		PreSpeechMs:     p.PreSpeechMs,
		MaxDurationSecs: p.MaxDurationSecs,
	}
}

type BreakerConfig struct {
	Disabled         bool
	FailureThreshold int
	Cooldown         time.Duration
	MaxCooldown      time.Duration
}

func (c BreakerConfig) withDefaults() BreakerConfig {
	if c.FailureThreshold <= 0 {
		c.FailureThreshold = DefaultBreakerFailureThreshold
	}
	if c.Cooldown <= 0 {
		c.Cooldown = DefaultBreakerCooldown
	}
	if c.MaxCooldown <= 0 {
		c.MaxCooldown = DefaultBreakerMaxCooldown
	}
	c.MaxCooldown = max(c.MaxCooldown, c.Cooldown)
	return c
}

type breaker struct {
	cfg       BreakerConfig
	mu        sync.Mutex
	failures  int
	open      bool
	cooldown  time.Duration
	openUntil time.Time
}

func (b *breaker) allow(now time.Time) bool {
	if b.cfg.Disabled {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return !b.open || !now.Before(b.openUntil)
}

func (b *breaker) isOpen() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.open
}

func (b *breaker) success() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	recovered := b.open
	b.failures, b.open, b.cooldown = 0, false, 0
	return recovered
}

func (b *breaker) failure(now time.Time) (opened bool, cooldown time.Duration) {
	if b.cfg.Disabled {
		return false, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.failures++
	switch {
	case b.open:
		b.cooldown = min(2*b.cooldown, b.cfg.MaxCooldown)
	case b.failures >= b.cfg.FailureThreshold:
		b.open = true
		b.cooldown = b.cfg.Cooldown
	default:
		return false, 0
	}
	b.openUntil = now.Add(b.cooldown)
	return true, b.cooldown
}

type prediction struct {
	complete    bool
	probability float64
	inferenceMs float64
}

type predictFunc func(ctx context.Context, samples []int16) (prediction, error)

type smartTurn struct {
	*BaseTurnAnalyzer
	name           string
	predict        predictFunc
	breaker        breaker
	requestTimeout time.Duration

	sem     chan struct{}
	segment []int16
}

func newSmartTurn(name string, params *SmartTurnParams, bc BreakerConfig, predict predictFunc) *smartTurn {
	if params == nil {
		params = DefaultSmartTurnParams()
	}
	return &smartTurn{
		BaseTurnAnalyzer: NewBaseTurnAnalyzer(ModelSampleRate, params.baseParams()),
		name:             name,
		predict:          predict,
		breaker:          breaker{cfg: bc.withDefaults()},
		sem:              make(chan struct{}, 1),
	}
}

func (s *smartTurn) BreakerOpen() bool { return s.breaker.isOpen() }

func (s *smartTurn) AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error) {
	return s.AnalyzeEndOfTurnContext(context.Background())
}

func (s *smartTurn) AnalyzeEndOfTurnContext(ctx context.Context) (EndOfTurnState, *TurnMetrics, error) {
	select {
	case s.sem <- struct{}{}:
	case <-ctx.Done():
		return TurnIncomplete, nil, ctx.Err()
	}
	defer func() { <-s.sem }()
	if err := ctx.Err(); err != nil {
		return TurnIncomplete, nil, err
	}

	start := time.Now()
	s.segment = s.AudioSegment(s.segment)
	if len(s.segment) == 0 {
		return TurnIncomplete, nil, nil
	}
	if !s.breaker.allow(start) {
		s.completeTurn()
		return TurnComplete, nil, nil
	}

	bound := ctx
	if stop := time.Duration(s.Params().StopSecs * float64(time.Second)); stop > 0 {
		var cancel context.CancelFunc
		bound, cancel = context.WithTimeout(ctx, stop)
		defer cancel()
	}
	request := bound
	if s.requestTimeout > 0 {
		var cancel context.CancelFunc
		request, cancel = context.WithTimeout(bound, s.requestTimeout)
		defer cancel()
	}

	pred, err := s.predict(request, s.segment)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return TurnIncomplete, nil, cerr
		}
		if opened, cooldown := s.breaker.failure(time.Now()); opened {
			logger.Warn("[%s] inference unavailable, using silence-based end of turn for %v: %v", s.name, cooldown, err)
		}
		if bound.Err() != nil {
			logger.Debug("[%s] end of turn complete: inference exceeded stop_secs", s.name)
			s.completeTurn()
			return TurnComplete, nil, nil
		}
		return TurnIncomplete, nil, err
	}
	if s.breaker.success() {
		logger.Info("[%s] inference recovered", s.name)
	}

	totalMs := float64(time.Since(start).Microseconds()) / 1000
	state := TurnIncomplete
	if pred.complete {
		state = TurnComplete
		s.completeTurn()
	}
	logger.Debug("[%s] prediction: %s (prob=%.3f, inference=%.1fms, total=%.1fms)", s.name, state, pred.probability, pred.inferenceMs, totalMs)
	return state, &TurnMetrics{
		IsComplete:      pred.complete,
		Probability:     pred.probability,
		InferenceTimeMs: pred.inferenceMs,
		TotalTimeMs:     totalMs,
	}, nil
}
