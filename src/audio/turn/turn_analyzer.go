package turn

import (
	"context"
	"encoding/binary"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const ModelSampleRate = 16000

type EndOfTurnState int

const (
	TurnIncomplete EndOfTurnState = iota
	TurnComplete
)

func (s EndOfTurnState) String() string {
	switch s {
	case TurnIncomplete:
		return "INCOMPLETE"
	case TurnComplete:
		return "COMPLETE"
	default:
		return "UNKNOWN"
	}
}

type TurnAnalyzerParams struct {
	StopSecs        float64
	PreSpeechMs     float64
	MaxDurationSecs float64
}

func DefaultTurnAnalyzerParams() *TurnAnalyzerParams {
	return &TurnAnalyzerParams{
		StopSecs:        3.0,
		MaxDurationSecs: 8.0,
	}
}

type TurnMetrics struct {
	IsComplete      bool
	Probability     float64
	InferenceTimeMs float64
	TotalTimeMs     float64
}

type TurnAnalyzer interface {
	AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState
	AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error)
	SpeechTriggered() bool
	SetSampleRate(sampleRate int)
	Clear()
}

type ContextTurnAnalyzer interface {
	TurnAnalyzer
	AnalyzeEndOfTurnContext(ctx context.Context) (EndOfTurnState, *TurnMetrics, error)
}

func AnalyzeEndOfTurn(ctx context.Context, a TurnAnalyzer) (EndOfTurnState, *TurnMetrics, error) {
	if c, ok := a.(ContextTurnAnalyzer); ok {
		return c.AnalyzeEndOfTurnContext(ctx)
	}
	if err := ctx.Err(); err != nil {
		return TurnIncomplete, nil, err
	}
	return a.AnalyzeEndOfTurn()
}

type BaseTurnAnalyzer struct {
	mu sync.Mutex

	params       TurnAnalyzerParams
	stopMs       float64
	sampleRate   int
	vadStartSecs float64

	resampler *resample.Resampler
	decoded   []int16
	converted []int16

	ring        []int16
	head        int
	total       int64
	speechStart int64

	speechTriggered bool
	silenceMs       float64
}

func NewBaseTurnAnalyzer(sampleRate int, params *TurnAnalyzerParams) *BaseTurnAnalyzer {
	if params == nil {
		params = DefaultTurnAnalyzerParams()
	}
	p := *params
	if p.MaxDurationSecs <= 0 {
		p.MaxDurationSecs = 8.0
	}
	b := &BaseTurnAnalyzer{
		params:      p,
		stopMs:      p.StopSecs * 1000,
		ring:        make([]int16, int(p.MaxDurationSecs*ModelSampleRate)),
		speechStart: -1,
	}
	b.setSampleRateLocked(sampleRate)
	return b
}

func (b *BaseTurnAnalyzer) SpeechTriggered() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.speechTriggered
}

func (b *BaseTurnAnalyzer) SampleRate() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.sampleRate
}

func (b *BaseTurnAnalyzer) Params() TurnAnalyzerParams {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.params
}

func (b *BaseTurnAnalyzer) SetSampleRate(sampleRate int) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.setSampleRateLocked(sampleRate)
}

func (b *BaseTurnAnalyzer) setSampleRateLocked(sampleRate int) {
	if sampleRate <= 0 {
		sampleRate = ModelSampleRate
	}
	if sampleRate == b.sampleRate {
		return
	}
	b.sampleRate = sampleRate
	b.resampler = nil
	if sampleRate != ModelSampleRate {
		r, err := resample.New(sampleRate, ModelSampleRate)
		if err != nil {
			logger.Error("[TurnAnalyzer] resampler %d->%d: %v", sampleRate, ModelSampleRate, err)
			return
		}
		b.resampler = r
	}
}

func (b *BaseTurnAnalyzer) SetVADStartSecs(secs float64) {
	b.mu.Lock()
	b.vadStartSecs = secs
	b.mu.Unlock()
}

func (b *BaseTurnAnalyzer) Clear() {
	b.mu.Lock()
	b.clearLocked(TurnComplete)
	b.mu.Unlock()
}

func (b *BaseTurnAnalyzer) clearLocked(state EndOfTurnState) {
	b.speechTriggered = state == TurnIncomplete
	b.head = 0
	b.total = 0
	b.speechStart = -1
	b.silenceMs = 0
}

func (b *BaseTurnAnalyzer) AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := len(buffer) / 2
	b.decoded = b.decoded[:0]
	for i := 0; i < n; i++ {
		b.decoded = append(b.decoded, int16(binary.LittleEndian.Uint16(buffer[2*i:])))
	}
	samples := b.decoded
	if b.resampler != nil {
		b.converted = b.resampler.Process(b.converted[:0], b.decoded)
		samples = b.converted
	}

	if isSpeech && b.speechStart < 0 {
		b.speechStart = b.total
	}
	b.writeLocked(samples)

	state := TurnIncomplete
	if isSpeech {
		b.silenceMs = 0
		b.speechTriggered = true
	} else if b.speechTriggered {
		b.silenceMs += float64(n) * 1000 / float64(b.sampleRate)
		if b.silenceMs >= b.stopMs {
			state = TurnComplete
			b.clearLocked(state)
		}
	}
	return state
}

func (b *BaseTurnAnalyzer) writeLocked(samples []int16) {
	c := len(b.ring)
	if len(samples) >= c {
		copy(b.ring, samples[len(samples)-c:])
		b.head = 0
		b.total += int64(len(samples))
		return
	}
	k := copy(b.ring[b.head:], samples)
	if k < len(samples) {
		copy(b.ring, samples[k:])
	}
	b.head = (b.head + len(samples)) % c
	b.total += int64(len(samples))
}

func (b *BaseTurnAnalyzer) AudioSegment(dst []int16) []int16 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.segmentLocked(dst)
}

func (b *BaseTurnAnalyzer) segmentLocked(dst []int16) []int16 {
	c := int64(len(b.ring))
	available := min(b.total, c)
	if available == 0 {
		return dst[:0]
	}
	from := b.total - available
	if b.speechStart >= 0 {
		pre := int64((b.params.PreSpeechMs + b.vadStartSecs*1000) * ModelSampleRate / 1000)
		from = max(from, b.speechStart-pre)
	}
	n := int(b.total - from)
	if cap(dst) < n {
		dst = make([]int16, n)
	}
	dst = dst[:n]
	start := int((int64(b.head) - int64(n) + c) % c)
	k := copy(dst, b.ring[start:])
	if k < n {
		copy(dst[k:], b.ring[:n-k])
	}
	return dst
}

func (b *BaseTurnAnalyzer) completeTurn() {
	b.mu.Lock()
	b.clearLocked(TurnComplete)
	b.mu.Unlock()
}
