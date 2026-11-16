package voicemail

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/pcm"
	"github.com/square-key-labs/strawgo-ai/src/audio/tone"
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
	last         Result
	result       Result
	decision     Verdict
	userSpeaking bool
	decisionDue  bool
	decisionGen  uint64
	decisionTmr  *time.Timer
	messageGen   uint64
	messageTmr   *time.Timer
	messageLeft  bool
	halted       bool
	started      bool
	carrier      Class

	timing    *timing
	timingGen uint64
	timingTmr *time.Timer

	beep     *tone.Detector
	beepRate int
	beepOff  bool
	mono     []byte

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
	if cfg.timing != nil {
		d.timing = &timing{th: *cfg.timing}
	}
	d.BaseProcessor = processors.NewBaseProcessor("VoicemailDetector", d)
	return d
}

func (d *Detector) Decision() Verdict {
	return Verdict(d.verdict.Load())
}

func (d *Detector) Result() Result {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.result
}

func (d *Detector) ReportCarrierResult(c Class) bool {
	if c == ClassUnknown {
		return false
	}
	d.mu.Lock()
	if !d.started {
		if d.halted || d.decision != Undecided {
			d.mu.Unlock()
			return false
		}
		d.carrier = c
		d.mu.Unlock()
		return true
	}
	d.mu.Unlock()
	return d.decide(Result{Class: c, Source: SourceCarrier})
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

	d.mu.Lock()
	d.started = true
	carrier := d.carrier
	d.carrier = ClassUnknown
	d.mu.Unlock()
	if carrier != ClassUnknown {
		d.decide(Result{Class: carrier, Source: SourceCarrier})
	}
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
	stopTimer(&d.timingTmr)
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
	case *frames.VADUserStartedSpeakingFrame:
		if direction == frames.Downstream {
			d.onVAD(true, eventTime(f.Timestamp, f.StartSecs))
		}
	case *frames.VADUserStoppedSpeakingFrame:
		if direction == frames.Downstream {
			d.onVAD(false, eventTime(f.Timestamp, f.StopSecs))
		}
	case *frames.AudioFrame:
		if direction == frames.Downstream && d.cfg.beep {
			d.onAudio(f)
		}
	case *frames.EndFrame, *frames.CancelFrame:
		d.halt()
	}

	if d.Decision() == Voicemail && !passesClosedGate(frame) {
		return nil
	}
	return d.PushFrame(frame, direction)
}

func eventTime(stamp time.Time, lagSecs float64) time.Time {
	if stamp.IsZero() {
		stamp = time.Now()
	}
	return stamp.Add(-time.Duration(lagSecs * float64(time.Second)))
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
				r := d.fallbackLocked()
				verdict := d.decideLocked(r)
				d.mu.Unlock()
				d.act(ctx, r, verdict)
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

		class := ClassUnknown
		if undecided {
			class = d.classify(ctx, transcript.String())
		}

		d.mu.Lock()
		d.classifying = false
		if class != ClassUnknown {
			d.last = Result{Class: class, Source: SourceClassifier}
		}
		d.mu.Unlock()
		if class != ClassUnknown && d.timing != nil {
			d.evaluateTiming()
		}
	}
}

func (d *Detector) classify(ctx context.Context, transcript string) Class {
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
		return ClassUnknown
	}
	class := classOf(ParseVerdict(reply))
	if d.cfg.detailed {
		class = ParseClass(reply)
	}
	logger.Debug("[%s] Classified %q as %s", d.Name(), transcript, class)
	return class
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
	r := d.fallbackLocked()
	verdict := d.decideLocked(r)
	d.mu.Unlock()
	d.act(d.Context(), r, verdict)
}

func (d *Detector) fallbackLocked() Result {
	if d.last.Class == ClassUnknown {
		logger.Warn("[%s] No classifier answer after %v of silence, assuming a conversation", d.Name(), d.cfg.decisionTimeout)
		return Result{Class: ClassUnknown, Source: SourceTimeout}
	}
	return d.last
}

func (d *Detector) decide(r Result) bool {
	d.mu.Lock()
	if d.halted || d.decision != Undecided {
		d.mu.Unlock()
		return false
	}
	verdict := d.decideLocked(r)
	d.mu.Unlock()
	d.act(d.Context(), r, verdict)
	return true
}

func (d *Detector) decideLocked(r Result) Verdict {
	verdict := r.Class.Verdict()
	d.result = r
	d.decision = verdict
	d.verdict.Store(int32(verdict))
	d.decisionDue = false
	stopTimer(&d.decisionTmr)
	d.timingGen++
	stopTimer(&d.timingTmr)
	if verdict == Voicemail {
		d.restartMessageTimerLocked()
	}
	return verdict
}

func (d *Detector) act(ctx context.Context, r Result, verdict Verdict) {
	logger.Info("[%s] Answered by %s (from %s)", d.Name(), r.Class, r.Source)
	if h := d.cfg.onResult; h != nil {
		h(d, r)
	}
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

func (d *Detector) onVAD(speaking bool, at time.Time) {
	if d.timing == nil {
		return
	}
	d.mu.Lock()
	if d.halted || d.decision != Undecided {
		d.mu.Unlock()
		return
	}
	if speaking {
		d.timing.start(at)
	} else {
		d.timing.stop(at)
	}
	d.mu.Unlock()
	d.evaluateTiming()
}

func (d *Detector) evaluateTiming() {
	d.mu.Lock()
	if d.halted || d.decision != Undecided {
		d.mu.Unlock()
		return
	}
	d.timingGen++
	stopTimer(&d.timingTmr)
	class, next := d.timing.evaluate(time.Now())
	if class != ClassUnknown {
		if r, ok := d.timingResultLocked(class); ok {
			verdict := d.decideLocked(r)
			d.mu.Unlock()
			d.act(d.Context(), r, verdict)
			return
		}
	} else if !next.IsZero() {
		gen := d.timingGen
		d.timingTmr = time.AfterFunc(time.Until(next), func() { d.onTimingTimer(gen) })
	}
	d.mu.Unlock()
}

func (d *Detector) onTimingTimer(gen uint64) {
	d.mu.Lock()
	current := gen == d.timingGen
	d.mu.Unlock()
	if current {
		d.evaluateTiming()
	}
}

func (d *Detector) timingResultLocked(class Class) (Result, bool) {
	heard := d.last.Class
	switch class {
	case ClassHuman:
		if heard.machine() {
			return Result{}, false
		}
		return Result{Class: ClassHuman, Source: SourceTiming}, true
	case ClassVoicemail:
		if heard == ClassHuman {
			return Result{}, false
		}
		if heard.machine() {
			return Result{Class: heard, Source: SourceTiming}, true
		}
		return Result{Class: ClassVoicemail, Source: SourceTiming}, true
	}
	return Result{}, false
}

func (d *Detector) onAudio(f *frames.AudioFrame) {
	d.mu.Lock()
	if d.halted || d.messageLeft || d.decision == Conversation || d.beepOff || f.SampleRate <= 0 {
		d.mu.Unlock()
		return
	}
	if d.beep == nil || d.beepRate != f.SampleRate {
		cfg := d.cfg.beepTone
		cfg.SampleRate = f.SampleRate
		det, err := tone.NewDetector(cfg)
		if err != nil {
			logger.Warn("[%s] Beep detection disabled: %v", d.Name(), err)
			d.beepOff = true
			d.mu.Unlock()
			return
		}
		d.beep, d.beepRate = det, f.SampleRate
	}
	data := f.Data
	if f.Channels > 1 {
		d.mono = pcm.ToMono(d.mono[:0], data, f.Channels)
		data = d.mono
	}
	events := d.beep.Write(data)
	d.mu.Unlock()

	for _, ev := range events {
		switch ev.Kind {
		case tone.Started:
			logger.Info("[%s] Beep at %.0f Hz", d.Name(), ev.Frequency)
			d.decide(Result{Class: ClassVoicemail, Source: SourceBeep})
		case tone.Ended:
			d.mu.Lock()
			if d.halted || d.messageLeft || d.decision != Voicemail {
				d.mu.Unlock()
				continue
			}
			d.leaveMessageAndUnlock()
		}
	}
}

func (d *Detector) restartMessageTimerLocked() {
	if d.halted || d.messageLeft {
		return
	}
	d.cancelMessageTimerLocked()
	gen := d.messageGen
	delay := d.cfg.responseDelay
	if d.cfg.beep {
		delay = d.cfg.beepTimeout
	}
	d.messageTmr = time.AfterFunc(delay, func() { d.onMessageTimer(gen) })
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
	d.leaveMessageAndUnlock()
}

func (d *Detector) leaveMessageAndUnlock() {
	d.messageLeft = true
	d.cancelMessageTimerLocked()
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
