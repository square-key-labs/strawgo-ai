package voicemail

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type Detector struct {
	*processors.BaseProcessor

	classifier Classifier
	cfg        config
	gate       *TTSGate
	wake       chan struct{}
	verdict    atomic.Int32

	mu           sync.Mutex
	pending      []string
	classifying  bool
	last         Verdict
	decision     Verdict
	userSpeaking bool
	decisionDue  bool
	decisionGen  uint64
	decisionTmr  *time.Timer
	messageGen   uint64
	messageTmr   *time.Timer
	messageLeft  bool
	halted       bool

	runMu     sync.Mutex
	runCancel context.CancelFunc
	runDone   chan struct{}
}

func newDetector(classifier Classifier, cfg config, gate *TTSGate) *Detector {
	d := &Detector{
		classifier: classifier,
		cfg:        cfg,
		gate:       gate,
		wake:       make(chan struct{}, 1),
	}
	d.BaseProcessor = processors.NewBaseProcessor("VoicemailDetector", d)
	return d
}

func (d *Detector) Decision() Verdict {
	return Verdict(d.verdict.Load())
}

func (d *Detector) Start(ctx context.Context) error {
	if err := d.BaseProcessor.Start(ctx); err != nil {
		return err
	}
	runCtx, cancel := context.WithCancel(d.Context())
	done := make(chan struct{})
	d.runMu.Lock()
	d.runCancel, d.runDone = cancel, done
	d.runMu.Unlock()
	go d.run(runCtx, done)
	return nil
}

func (d *Detector) Stop() error {
	d.halt()
	d.runMu.Lock()
	done := d.runDone
	d.runMu.Unlock()
	if done != nil {
		<-done
	}
	return d.BaseProcessor.Stop()
}

func (d *Detector) halt() {
	d.mu.Lock()
	d.halted = true
	d.pending = nil
	d.decisionDue = false
	stopTimer(&d.decisionTmr)
	stopTimer(&d.messageTmr)
	d.mu.Unlock()

	d.runMu.Lock()
	cancel := d.runCancel
	d.runMu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (d *Detector) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.TranscriptionFrame:
		d.onTranscription(f.Text)
	case *frames.UserStartedSpeakingFrame:
		d.mu.Lock()
		d.userSpeaking = true
		d.cancelDecisionTimerLocked()
		d.cancelMessageTimerLocked()
		d.mu.Unlock()
	case *frames.UserStoppedSpeakingFrame:
		d.mu.Lock()
		d.userSpeaking = false
		switch {
		case d.decision == Undecided:
			d.restartDecisionTimerLocked()
		case d.decision == Voicemail && !d.messageLeft:
			d.restartMessageTimerLocked()
		}
		d.mu.Unlock()
	case *frames.EndFrame, *frames.CancelFrame:
		d.halt()
	}

	if d.Decision() == Voicemail && !passesClosedGate(frame) {
		return nil
	}
	return d.PushFrame(frame, direction)
}

func passesClosedGate(frame frames.Frame) bool {
	if frames.IsSystem(frame) {
		return true
	}
	switch frame.(type) {
	case *frames.EndFrame, *frames.StopFrame, *frames.EndTaskFrame, *frames.InterruptionTaskFrame:
		return true
	}
	return false
}

func (d *Detector) onTranscription(text string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}
	d.mu.Lock()
	if d.halted || d.decision != Undecided {
		d.mu.Unlock()
		return
	}
	d.pending = append(d.pending, text)
	if !d.userSpeaking {
		d.restartDecisionTimerLocked()
	}
	d.mu.Unlock()
	select {
	case d.wake <- struct{}{}:
	default:
	}
}

func (d *Detector) run(ctx context.Context, done chan struct{}) {
	defer close(done)
	var transcript strings.Builder
	for {
		d.mu.Lock()
		if d.halted {
			d.mu.Unlock()
			return
		}
		if len(d.pending) == 0 {
			if d.decisionDue && d.decision == Undecided {
				d.decisionDue = false
				verdict := d.decideLocked()
				d.mu.Unlock()
				d.act(ctx, verdict)
				continue
			}
			d.mu.Unlock()
			select {
			case <-ctx.Done():
				return
			case <-d.wake:
			}
			continue
		}
		segments := d.pending
		d.pending = nil
		undecided := d.decision == Undecided
		d.classifying = true
		d.mu.Unlock()

		for _, s := range segments {
			if transcript.Len() > 0 {
				transcript.WriteByte(' ')
			}
			transcript.WriteString(s)
		}
		if transcript.Len() > maxTranscriptLen {
			tail := transcript.String()[transcript.Len()-maxTranscriptLen:]
			transcript.Reset()
			transcript.WriteString(tail)
		}

		verdict := Undecided
		if undecided {
			verdict = d.classify(ctx, transcript.String())
		}

		d.mu.Lock()
		d.classifying = false
		if verdict != Undecided {
			d.last = verdict
		}
		d.mu.Unlock()
	}
}

func (d *Detector) classify(ctx context.Context, transcript string) Verdict {
	cctx := ctx
	if d.cfg.classifyTimeout > 0 {
		var cancel context.CancelFunc
		cctx, cancel = context.WithTimeout(ctx, d.cfg.classifyTimeout)
		defer cancel()
	}
	snapshot := llm.Snapshot{Messages: []llm.Message{llm.System(d.cfg.prompt), llm.User(transcript)}}
	reply, err := d.classifier.RunInference(cctx, snapshot, d.cfg.settings)
	if err != nil {
		if ctx.Err() == nil {
			logger.Warn("[%s] Classification failed: %v", d.Name(), err)
		}
		return Undecided
	}
	verdict := ParseVerdict(reply)
	logger.Debug("[%s] Classified %q as %s", d.Name(), transcript, verdict)
	return verdict
}

func (d *Detector) restartDecisionTimerLocked() {
	if d.halted || d.decision != Undecided {
		return
	}
	d.cancelDecisionTimerLocked()
	d.decisionGen++
	gen := d.decisionGen
	d.decisionTmr = time.AfterFunc(d.cfg.decisionTimeout, func() { d.onDecisionTimer(gen) })
}

func (d *Detector) cancelDecisionTimerLocked() {
	d.decisionGen++
	d.decisionDue = false
	stopTimer(&d.decisionTmr)
}

func (d *Detector) onDecisionTimer(gen uint64) {
	d.mu.Lock()
	if d.halted || gen != d.decisionGen || d.decision != Undecided {
		d.mu.Unlock()
		return
	}
	if len(d.pending) > 0 || d.classifying {
		d.decisionDue = true
		d.mu.Unlock()
		select {
		case d.wake <- struct{}{}:
		default:
		}
		return
	}
	verdict := d.decideLocked()
	d.mu.Unlock()
	d.act(d.Context(), verdict)
}

func (d *Detector) decideLocked() Verdict {
	verdict := d.last
	if verdict == Undecided {
		logger.Warn("[%s] No classifier answer after %v of silence, assuming a conversation", d.Name(), d.cfg.decisionTimeout)
		verdict = Conversation
	}
	d.decision = verdict
	d.verdict.Store(int32(verdict))
	stopTimer(&d.decisionTmr)
	if verdict == Voicemail {
		d.restartMessageTimerLocked()
	}
	return verdict
}

func (d *Detector) act(ctx context.Context, verdict Verdict) {
	switch verdict {
	case Voicemail:
		logger.Info("[%s] Voicemail detected", d.Name())
		d.gate.discard()
		if err := d.BroadcastInterruption(ctx); err != nil && ctx.Err() == nil {
			logger.Warn("[%s] Interruption broadcast failed: %v", d.Name(), err)
		}
	case Conversation:
		logger.Info("[%s] Conversation detected", d.Name())
		d.gate.release()
		if h := d.cfg.onConversation; h != nil {
			h(d)
		}
	}
}

func (d *Detector) restartMessageTimerLocked() {
	if d.halted || d.messageLeft {
		return
	}
	d.cancelMessageTimerLocked()
	gen := d.messageGen
	d.messageTmr = time.AfterFunc(d.cfg.responseDelay, func() { d.onMessageTimer(gen) })
}

func (d *Detector) cancelMessageTimerLocked() {
	d.messageGen++
	stopTimer(&d.messageTmr)
}

func (d *Detector) onMessageTimer(gen uint64) {
	d.mu.Lock()
	if d.halted || gen != d.messageGen || d.messageLeft {
		d.mu.Unlock()
		return
	}
	d.messageLeft = true
	d.messageTmr = nil
	d.mu.Unlock()
	if h := d.cfg.onVoicemail; h != nil {
		h(d)
	}
}

func stopTimer(t **time.Timer) {
	if *t != nil {
		(*t).Stop()
		*t = nil
	}
}
