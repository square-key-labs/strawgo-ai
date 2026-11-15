package vad

import (
	"math"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type VADState int

const (
	VADStateQuiet VADState = iota + 1
	VADStateStarting
	VADStateSpeaking
	VADStateStopping
)

func (s VADState) String() string {
	switch s {
	case VADStateQuiet:
		return "quiet"
	case VADStateStarting:
		return "starting"
	case VADStateSpeaking:
		return "speaking"
	case VADStateStopping:
		return "stopping"
	default:
		return "unknown"
	}
}

type VADParams struct {
	Confidence     float32
	StartSecs      float32
	StopSecs       float32
	MinVolume      float32
	HoldConfidence float32
}

func (p VADParams) EffectiveHoldConfidence() float32 {
	if p.HoldConfidence <= 0 || p.HoldConfidence > p.Confidence {
		return p.Confidence
	}
	return p.HoldConfidence
}

func (p VADParams) Merge(update VADParams) VADParams {
	if update.Confidence > 0 {
		p.Confidence = update.Confidence
	}
	if update.StartSecs > 0 {
		p.StartSecs = update.StartSecs
	}
	if update.StopSecs > 0 {
		p.StopSecs = update.StopSecs
	}
	if update.MinVolume > 0 {
		p.MinVolume = update.MinVolume
	}
	if update.HoldConfidence > 0 {
		p.HoldConfidence = update.HoldConfidence
	}
	return p
}

func DefaultVADParams() VADParams {
	return VADParams{
		Confidence: 0.7,
		StartSecs:  0.2,
		StopSecs:   0.2,
		MinVolume:  0.1,
	}
}

func (p VADParams) ToFrameParams() frames.VADParams {
	return frames.VADParams{Confidence: p.Confidence, StartSecs: p.StartSecs, StopSecs: p.StopSecs, MinVolume: p.MinVolume, HoldConfidence: p.HoldConfidence}
}

func ParamsFromFrame(p frames.VADParams) VADParams {
	return VADParams{Confidence: p.Confidence, StartSecs: p.StartSecs, StopSecs: p.StopSecs, MinVolume: p.MinVolume, HoldConfidence: p.HoldConfidence}
}

type VADAnalyzer interface {
	SetSampleRate(sampleRate int) error
	NumFramesRequired() int
	VoiceConfidence(buffer []byte) float32
	AnalyzeAudio(buffer []byte) (VADState, error)
	Restart()
}

type ParamsSetter interface {
	SetParams(params VADParams)
	GetParams() VADParams
}

type BaseVADAnalyzer struct {
	params     VADParams
	sampleRate int

	state          VADState
	startFrames    int
	stopFrames     int
	startThreshold int
	stopThreshold  int
	thresholdRate  int
	thresholdSize  int

	smoothedVolume float32

	mu sync.RWMutex
}

const recommendedStopSecs float32 = 0.2

const volumeSmoothingFactor = 0.2

func NewBaseVADAnalyzer(sampleRate int, params VADParams) *BaseVADAnalyzer {
	warnStopSecs(params)
	return &BaseVADAnalyzer{
		params:     params,
		sampleRate: sampleRate,
		state:      VADStateQuiet,
	}
}

func warnStopSecs(params VADParams) {
	if params.StopSecs != recommendedStopSecs {
		logger.Warn("[VADAnalyzer] StopSecs=%.2fs (recommended: %.2fs). "+
			"Values != %.2fs may affect turn-detection latency or cause premature turn endings.",
			params.StopSecs, recommendedStopSecs, recommendedStopSecs)
	}
}

func (v *BaseVADAnalyzer) SetSampleRate(sampleRate int) error {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.sampleRate = sampleRate
	return nil
}

func (v *BaseVADAnalyzer) GetSampleRate() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.sampleRate
}

func (v *BaseVADAnalyzer) GetParams() VADParams {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.params
}

func (v *BaseVADAnalyzer) SetParams(params VADParams) {
	warnStopSecs(params)
	v.mu.Lock()
	defer v.mu.Unlock()
	v.params = params
	v.thresholdSize = 0
}

func (v *BaseVADAnalyzer) GetState() VADState {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.state
}

func (v *BaseVADAnalyzer) Restart() {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.resetLocked()
}

func (v *BaseVADAnalyzer) resetLocked() {
	v.state = VADStateQuiet
	v.startFrames = 0
	v.stopFrames = 0
	v.smoothedVolume = 0.0
	v.thresholdSize = 0
}

func framesFor(secs float32, numFramesRequired, sampleRate int) int {
	if secs <= 0 || numFramesRequired <= 0 || sampleRate <= 0 {
		return 0
	}
	return int(math.Round(float64(secs) * float64(sampleRate) / float64(numFramesRequired)))
}

func (v *BaseVADAnalyzer) ProcessAudio(buffer []byte, voiceConfidence float32, numFramesRequired int) (VADState, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	volume := float32(pcm.RMS(buffer))
	v.smoothedVolume = float32(pcm.ExpSmoothing(float64(volume), float64(v.smoothedVolume), volumeSmoothingFactor))

	if numFramesRequired != v.thresholdSize || v.sampleRate != v.thresholdRate {
		v.thresholdSize = numFramesRequired
		v.thresholdRate = v.sampleRate
		v.startThreshold = framesFor(v.params.StartSecs, numFramesRequired, v.sampleRate)
		v.stopThreshold = framesFor(v.params.StopSecs, numFramesRequired, v.sampleRate)
	}

	if v.smoothedVolume < v.params.MinVolume {
		voiceConfidence = 0.0
	}

	oldState := v.state
	threshold := v.params.Confidence
	if oldState == VADStateSpeaking || oldState == VADStateStopping {
		threshold = v.params.EffectiveHoldConfidence()
	}
	speech := voiceConfidence >= threshold

	switch v.state {
	case VADStateQuiet:
		if speech {
			v.startFrames++
			if v.startFrames >= v.startThreshold {
				v.state = VADStateSpeaking
				v.startFrames = 0
			} else {
				v.state = VADStateStarting
			}
		}
	case VADStateStarting:
		if speech {
			v.startFrames++
			v.stopFrames = 0
			if v.startFrames >= v.startThreshold {
				v.state = VADStateSpeaking
				v.startFrames = 0
			}
		} else {
			v.stopFrames++
			startingStopThreshold := max(v.stopThreshold/4, 2)
			if v.stopFrames >= startingStopThreshold {
				v.state = VADStateQuiet
				v.startFrames = 0
				v.stopFrames = 0
			}
		}
	case VADStateSpeaking:
		if !speech {
			v.stopFrames++
			if v.stopFrames >= v.stopThreshold {
				v.state = VADStateQuiet
				v.stopFrames = 0
			} else {
				v.state = VADStateStopping
			}
		} else {
			v.stopFrames = 0
		}
	case VADStateStopping:
		if !speech {
			v.stopFrames++
			if v.stopFrames >= v.stopThreshold {
				v.state = VADStateQuiet
				v.stopFrames = 0
			}
		} else {
			v.state = VADStateSpeaking
			v.stopFrames = 0
		}
	}

	if oldState != v.state && logger.IsDebugEnabled() {
		logger.Debug("[VADAnalyzer] %s -> %s (confidence=%.3f, volume=%.3f)", oldState, v.state, voiceConfidence, v.smoothedVolume)
	}

	return v.state, nil
}
