package observers

import (
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type LatencyBreakdown struct {
	TotalLatency time.Duration
	STTLatency   time.Duration
	LLMLatency   time.Duration
	TTSLatency   time.Duration
}

type UserBotLatencyObserver struct {
	mu sync.Mutex

	OnLatencyBreakdown      func(breakdown LatencyBreakdown)
	OnFirstBotSpeechLatency func(latency time.Duration)

	userStoppedAt      time.Time
	hasUserStoppedAt   bool
	transcriptionAt    time.Time
	hasTranscriptionAt bool
	llmTextAt          time.Time
	hasLLMTextAt       bool
	latencyReported    bool

	clientConnectedAt    time.Time
	hasClientConnectedAt bool
	firstSpeechReported  bool
}

func NewUserBotLatencyObserver() *UserBotLatencyObserver {
	return &UserBotLatencyObserver{}
}

func (o *UserBotLatencyObserver) OnProcessFrame(event pipeline.ProcessFrameEvent) {
	o.handleFrame(event.Frame, event.Timestamp)
}

func (o *UserBotLatencyObserver) OnPushFrame(event pipeline.PushFrameEvent) {
	o.handleFrame(event.Frame, event.Timestamp)
}

func (o *UserBotLatencyObserver) OnPipelineStarted() {
	o.reset()
}

func (o *UserBotLatencyObserver) OnPipelineStopped() {
	o.reset()
}

func (o *UserBotLatencyObserver) handleFrame(frame frames.Frame, now time.Time) {
	o.mu.Lock()

	switch frame.(type) {
	case *frames.ClientConnectedFrame:
		o.clientConnectedAt = now
		o.hasClientConnectedAt = true
	case *frames.UserStoppedSpeakingFrame:
		o.userStoppedAt = now
		o.hasUserStoppedAt = true
		o.transcriptionAt = time.Time{}
		o.hasTranscriptionAt = false
		o.llmTextAt = time.Time{}
		o.hasLLMTextAt = false
		o.latencyReported = false
	case *frames.TranscriptionFrame:
		if o.hasUserStoppedAt && !o.hasTranscriptionAt {
			o.transcriptionAt = now
			o.hasTranscriptionAt = true
		}
	case *frames.LLMTextFrame:
		if o.hasTranscriptionAt && !o.hasLLMTextAt {
			o.llmTextAt = now
			o.hasLLMTextAt = true
		}
	case *frames.TTSAudioFrame:
		firstLatencyCB := o.OnFirstBotSpeechLatency
		var firstLatency time.Duration
		shouldEmitFirstLatency := false

		if o.hasClientConnectedAt && !o.firstSpeechReported {
			firstLatency = now.Sub(o.clientConnectedAt)
			o.firstSpeechReported = true
			shouldEmitFirstLatency = firstLatencyCB != nil
		}

		breakdownCB := o.OnLatencyBreakdown
		var breakdown LatencyBreakdown
		shouldEmitBreakdown := false

		if o.hasUserStoppedAt && !o.latencyReported {
			breakdown.TotalLatency = now.Sub(o.userStoppedAt)
			if o.hasTranscriptionAt {
				breakdown.STTLatency = o.transcriptionAt.Sub(o.userStoppedAt)
			}
			if o.hasTranscriptionAt && o.hasLLMTextAt {
				breakdown.LLMLatency = o.llmTextAt.Sub(o.transcriptionAt)
				breakdown.TTSLatency = now.Sub(o.llmTextAt)
			}
			o.latencyReported = true
			shouldEmitBreakdown = breakdownCB != nil
		}

		o.mu.Unlock()

		if shouldEmitFirstLatency {
			go firstLatencyCB(firstLatency)
		}
		if shouldEmitBreakdown {
			go breakdownCB(breakdown)
		}
		return
	}

	o.mu.Unlock()
}

func (o *UserBotLatencyObserver) reset() {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.userStoppedAt = time.Time{}
	o.hasUserStoppedAt = false
	o.transcriptionAt = time.Time{}
	o.hasTranscriptionAt = false
	o.llmTextAt = time.Time{}
	o.hasLLMTextAt = false
	o.latencyReported = false

	o.clientConnectedAt = time.Time{}
	o.hasClientConnectedAt = false
	o.firstSpeechReported = false
}
