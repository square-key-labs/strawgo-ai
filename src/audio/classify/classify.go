package classify

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type Result struct {
	Label      string
	Confidence float64
}

type Window struct {
	// Audio is reused for the next window as soon as Classify returns; copy it to keep it.
	Audio      []byte
	SampleRate int
	Channels   int
	Offset     time.Duration
}

func (w Window) Duration() time.Duration {
	if w.SampleRate <= 0 || w.Channels <= 0 {
		return 0
	}
	samples := len(w.Audio) / 2 / w.Channels
	return time.Duration(samples) * time.Second / time.Duration(w.SampleRate)
}

type Classifier interface {
	Classify(ctx context.Context, window Window) (Result, error)
}

type ClassifierFunc func(ctx context.Context, window Window) (Result, error)

func (f ClassifierFunc) Classify(ctx context.Context, window Window) (Result, error) {
	return f(ctx, window)
}

type ResultFrame struct {
	*frames.DataFrame
	Result     Result
	Classifier string
	Offset     time.Duration
	Duration   time.Duration
}

func NewResultFrame(classifier string, result Result, offset, duration time.Duration) *ResultFrame {
	return &ResultFrame{
		DataFrame:  &frames.DataFrame{BaseFrame: frames.NewBaseFrame("AudioClassificationFrame")},
		Result:     result,
		Classifier: classifier,
		Offset:     offset,
		Duration:   duration,
	}
}

const (
	LabelSilence = "silence"
	LabelSound   = "sound"

	DefaultSilenceDBFS = -45.0
	energyFullScaleDB  = 20.0
)

type EnergyClassifier struct {
	SilenceDBFS float64
}

func NewEnergyClassifier() *EnergyClassifier {
	return &EnergyClassifier{SilenceDBFS: DefaultSilenceDBFS}
}

func (c *EnergyClassifier) Classify(_ context.Context, window Window) (Result, error) {
	threshold := c.SilenceDBFS
	if threshold == 0 {
		threshold = DefaultSilenceDBFS
	}
	level := pcm.DBFS(window.Audio)
	label := LabelSilence
	if level >= threshold {
		label = LabelSound
	}
	confidence := 0.5 + math.Abs(level-threshold)/(2*energyFullScaleDB)
	return Result{Label: label, Confidence: math.Min(confidence, 1)}, nil
}

type PanicError struct {
	Value any
}

func (e *PanicError) Error() string {
	return fmt.Sprintf("classifier panic: %v", e.Value)
}
