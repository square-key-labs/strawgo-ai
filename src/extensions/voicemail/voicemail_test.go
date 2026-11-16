package voicemail

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/dtmf"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type fakeClassifier struct {
	mu      sync.Mutex
	reply   string
	release chan struct{}
	calls   int
}

func (f *fakeClassifier) RunInference(ctx context.Context, _ llm.Snapshot, _ llm.Settings) (string, error) {
	f.mu.Lock()
	f.calls++
	reply, release := f.reply, f.release
	f.mu.Unlock()
	if release != nil {
		select {
		case <-release:
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	return reply, nil
}

type sink struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (s *sink) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error { return nil }
func (s *sink) QueueFrame(frame frames.Frame, _ frames.FrameDirection) error {
	s.mu.Lock()
	s.frames = append(s.frames, frame)
	s.mu.Unlock()
	return nil
}
func (s *sink) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (s *sink) Link(processors.FrameProcessor)                      {}
func (s *sink) SetPrev(processors.FrameProcessor)                   {}
func (s *sink) Start(context.Context) error                         { return nil }
func (s *sink) Stop() error                                         { return nil }
func (s *sink) Name() string                                        { return "sink" }

func (s *sink) interruptions() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, f := range s.frames {
		if _, ok := f.(*frames.InterruptionFrame); ok {
			n++
		}
	}
	return n
}

type rig struct {
	t             *testing.T
	vm            *VoicemailDetector
	d             *Detector
	out           *sink
	conversations atomic.Int32
	voicemails    atomic.Int32
	voicemailAt   atomic.Int64
	mu            sync.Mutex
	results       []Result
}

func newRig(t *testing.T, c Classifier, opts ...Option) *rig {
	t.Helper()
	r := &rig{t: t, out: &sink{}}
	opts = append([]Option{
		OnConversationDetected(func(*Detector) { r.conversations.Add(1) }),
		OnVoicemailDetected(func(*Detector) {
			r.voicemailAt.Store(time.Now().UnixNano())
			r.voicemails.Add(1)
		}),
		OnResult(func(_ *Detector, res Result) {
			r.mu.Lock()
			r.results = append(r.results, res)
			r.mu.Unlock()
		}),
	}, opts...)
	vm, err := New(c, opts...)
	if err != nil {
		t.Fatal(err)
	}
	r.vm, r.d = vm, vm.Detector()
	r.d.Link(r.out)
	return r
}

func (r *rig) start() *rig {
	r.t.Helper()
	if err := r.d.Start(context.Background()); err != nil {
		r.t.Fatal(err)
	}
	r.t.Cleanup(func() { _ = r.d.Stop() })
	r.send(frames.NewStartFrame())
	return r
}

func (r *rig) send(f frames.Frame) {
	r.t.Helper()
	if err := r.d.HandleFrame(context.Background(), f, frames.Downstream); err != nil {
		r.t.Fatal(err)
	}
}

func (r *rig) say(text string) {
	r.send(frames.NewUserStartedSpeakingFrame())
	r.send(frames.NewTranscriptionFrame(text, true))
	r.send(frames.NewUserStoppedSpeakingFrame())
}

func (r *rig) vad(speaking bool, at time.Time) {
	if speaking {
		f := frames.NewVADUserStartedSpeakingFrame(0)
		f.Timestamp = at
		r.send(f)
		return
	}
	f := frames.NewVADUserStoppedSpeakingFrame(0)
	f.Timestamp = at
	r.send(f)
}

func (r *rig) resultList() []Result {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Result(nil), r.results...)
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func settle() {
	time.Sleep(150 * time.Millisecond)
}

func weakBeepPCM(rate int, d time.Duration) []byte {
	n := int(d.Seconds() * float64(rate))
	out := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		t := float64(i) / float64(rate)
		v := 8000*math.Sin(2*math.Pi*1000*t) + 4000*math.Sin(2*math.Pi*3000*t)
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(v)))
	}
	return out
}

func tonePCM(rate int, freq float64, d time.Duration) []byte {
	n := int(d.Seconds() * float64(rate))
	out := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(8000*math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))))
	}
	return out
}

func (r *rig) audio(rate int, data []byte) {
	chunk := 2 * rate / 50
	for len(data) > 0 {
		n := min(chunk, len(data))
		r.send(frames.NewAudioFrame(append([]byte(nil), data[:n]...), rate, 1))
		data = data[n:]
	}
}

func TestDefaultsUnchanged(t *testing.T) {
	c := &fakeClassifier{reply: "VOICEMAIL"}
	r := newRig(t, c, WithDecisionTimeout(30*time.Millisecond), WithVoicemailResponseDelay(30*time.Millisecond)).start()
	if r.d.cfg.prompt != DefaultPrompt || r.d.cfg.detailed || r.d.cfg.beep || r.d.timing != nil {
		t.Fatalf("defaults changed: %+v", r.d.cfg)
	}

	now := time.Now()
	r.vad(true, now.Add(-5*time.Second))
	r.vad(false, now.Add(-4900*time.Millisecond))
	r.audio(8000, tonePCM(8000, 1000, 600*time.Millisecond))
	r.audio(8000, make([]byte, 1600))
	settle()
	if r.vm.Decision() != Undecided {
		t.Fatalf("decided %s from VAD or audio without opting in", r.vm.Decision())
	}

	r.say("You've reached Sam, leave a message")
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
	if got := r.vm.Result(); got != (Result{Class: ClassVoicemail, Source: SourceClassifier}) {
		t.Fatalf("result = %+v", got)
	}
	waitFor(t, "interruption", func() bool { return r.out.interruptions() > 0 })
	if r.conversations.Load() != 0 {
		t.Fatalf("conversations %d", r.conversations.Load())
	}
}

func TestDefaultModeIgnoresDetailedReplies(t *testing.T) {
	c := &fakeClassifier{reply: "IVR"}
	r := newRig(t, c, WithDecisionTimeout(30*time.Millisecond)).start()
	r.say("Press 1 for billing")
	waitFor(t, "conversation", func() bool { return r.conversations.Load() == 1 })
	if got := r.vm.Result(); got != (Result{Class: ClassUnknown, Source: SourceTimeout}) {
		t.Fatalf("result = %+v", got)
	}
}

func TestParseClass(t *testing.T) {
	cases := map[string]Class{
		"CONVERSATION":               ClassHuman,
		"human":                      ClassHuman,
		"Voicemail.":                 ClassVoicemail,
		"ivr":                        ClassIVR,
		"ANNOUNCEMENT":               ClassAnnouncement,
		"UNKNOWN":                    ClassUnknown,
		"":                           ClassUnknown,
		"IVR, not VOICEMAIL":         ClassIVR,
		"the answer is ANNOUNCEMENT": ClassAnnouncement,
		"DIVRT":                      ClassUnknown,
	}
	for reply, want := range cases {
		if got := ParseClass(reply); got != want {
			t.Errorf("ParseClass(%q) = %s, want %s", reply, got, want)
		}
	}
}

func TestClassFromAnsweredBy(t *testing.T) {
	cases := map[string]Class{
		"human":               ClassHuman,
		"machine_start":       ClassVoicemail,
		"machine_end_beep":    ClassVoicemail,
		"machine_end_silence": ClassVoicemail,
		"machine_end_other":   ClassVoicemail,
		"fax":                 ClassUnknown,
		"unknown":             ClassUnknown,
		"":                    ClassUnknown,
	}
	for in, want := range cases {
		if got := ClassFromAnsweredBy(in); got != want {
			t.Errorf("ClassFromAnsweredBy(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestDetailedPromptIsValidAndOverridable(t *testing.T) {
	if err := ValidatePrompt(DetailedPrompt); err != nil {
		t.Fatal(err)
	}
	vm, err := New(&fakeClassifier{}, WithDetailedClasses())
	if err != nil || vm.Detector().cfg.prompt != DetailedPrompt {
		t.Fatalf("detailed prompt not selected: %v", err)
	}
	custom := "Answer CONVERSATION, VOICEMAIL, IVR or ANNOUNCEMENT."
	vm, err = New(&fakeClassifier{}, WithPrompt(custom), WithDetailedClasses())
	if err != nil || vm.Detector().cfg.prompt != custom {
		t.Fatalf("custom prompt replaced: %v", err)
	}
	if _, err := New(&fakeClassifier{}, WithDetailedClasses(), WithPrompt("Answer IVR")); err != ErrInvalidPrompt {
		t.Fatalf("err = %v", err)
	}
}

func TestEachClassReachesCallbacks(t *testing.T) {
	cases := []struct {
		reply        string
		want         Result
		conversation bool
	}{
		{"CONVERSATION", Result{ClassHuman, SourceClassifier}, true},
		{"VOICEMAIL", Result{ClassVoicemail, SourceClassifier}, false},
		{"IVR", Result{ClassIVR, SourceClassifier}, true},
		{"ANNOUNCEMENT", Result{ClassAnnouncement, SourceClassifier}, false},
		{"UNKNOWN", Result{ClassUnknown, SourceTimeout}, true},
	}
	for _, tc := range cases {
		t.Run(tc.reply, func(t *testing.T) {
			r := newRig(t, &fakeClassifier{reply: tc.reply}, WithDetailedClasses(),
				WithDecisionTimeout(30*time.Millisecond), WithVoicemailResponseDelay(30*time.Millisecond)).start()
			r.say("something was heard")
			if tc.conversation {
				waitFor(t, "conversation", func() bool { return r.conversations.Load() == 1 })
			} else {
				waitFor(t, "voicemail", func() bool { return r.voicemails.Load() == 1 })
			}
			settle()
			if got := r.resultList(); len(got) != 1 || got[0] != tc.want {
				t.Fatalf("results = %+v", got)
			}
			if r.vm.Result() != tc.want || r.vm.Decision() != tc.want.Class.Verdict() {
				t.Fatalf("result %+v decision %s", r.vm.Result(), r.vm.Decision())
			}
			if tc.conversation && r.voicemails.Load() != 0 || !tc.conversation && r.conversations.Load() != 0 {
				t.Fatalf("both callbacks ran")
			}
		})
	}
}

func TestTimingScriptedSequences(t *testing.T) {
	th := DefaultTimingThresholds()
	t0 := time.Unix(1000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	type step struct {
		start bool
		ms    int
	}
	cases := []struct {
		name   string
		steps  []step
		now    int
		want   Class
		wantAt int
	}{
		{"short greeting then silence", []step{{true, 0}, {false, 600}}, 1400, ClassHuman, -1},
		{"short greeting not yet quiet", []step{{true, 0}, {false, 600}}, 1000, ClassUnknown, 1400},
		{"two short words in one run", []step{{true, 0}, {false, 400}, {true, 700}, {false, 1100}}, 1900, ClassHuman, -1},
		{"long uninterrupted speech", []step{{true, 0}}, 3000, ClassVoicemail, -1},
		{"speech still under machine length", []step{{true, 0}}, 2000, ClassUnknown, 3000},
		{"greeting with short pauses", []step{{true, 0}, {false, 1200}, {true, 1500}, {false, 2400}, {true, 2700}}, 3000, ClassVoicemail, -1},
		{"medium greeting defers", []step{{true, 0}, {false, 2000}}, 4000, ClassUnknown, -1},
		{"two separate utterances defer", []step{{true, 0}, {false, 500}, {true, 1500}, {false, 2000}}, 4000, ClassUnknown, -1},
		{"nothing heard", nil, 5000, ClassUnknown, -1},
	}
	for _, tc := range cases {
		tm := &timing{th: th}
		for _, s := range tc.steps {
			if s.start {
				tm.start(at(s.ms))
			} else {
				tm.stop(at(s.ms))
			}
		}
		got, next := tm.evaluate(at(tc.now))
		if got != tc.want {
			t.Errorf("%s: class %s, want %s", tc.name, got, tc.want)
		}
		if tc.wantAt >= 0 && !next.Equal(at(tc.wantAt)) {
			t.Errorf("%s: next check %v, want %v", tc.name, next.Sub(t0), at(tc.wantAt).Sub(t0))
		}
		if tc.wantAt < 0 && !next.IsZero() {
			t.Errorf("%s: unexpected next check %v", tc.name, next.Sub(t0))
		}
	}
}

func TestTimingDecidesHumanBeforeSlowClassifier(t *testing.T) {
	c := &fakeClassifier{reply: "VOICEMAIL", release: make(chan struct{})}
	r := newRig(t, c, WithTimingHeuristic(TimingThresholds{}), WithDecisionTimeout(50*time.Millisecond)).start()
	now := time.Now()
	r.vad(true, now.Add(-1500*time.Millisecond))
	r.send(frames.NewTranscriptionFrame("hello?", true))
	r.vad(false, now.Add(-900*time.Millisecond))
	waitFor(t, "conversation", func() bool { return r.conversations.Load() == 1 })
	if got := r.vm.Result(); got != (Result{ClassHuman, SourceTiming}) {
		t.Fatalf("result = %+v", got)
	}
	close(c.release)
	r.send(frames.NewUserStoppedSpeakingFrame())
	settle()
	if r.conversations.Load() != 1 || r.voicemails.Load() != 0 || len(r.resultList()) != 1 || r.vm.Decision() != Conversation {
		t.Fatalf("second decision: conversations %d voicemails %d results %+v", r.conversations.Load(), r.voicemails.Load(), r.resultList())
	}
}

func TestTimingDecidesHumanAfterSilenceTimer(t *testing.T) {
	r := newRig(t, &fakeClassifier{}, WithTimingHeuristic(TimingThresholds{HumanSilence: 80 * time.Millisecond})).start()
	now := time.Now()
	r.vad(true, now.Add(-400*time.Millisecond))
	r.vad(false, now)
	if r.vm.Decision() != Undecided {
		t.Fatal("decided before the silence elapsed")
	}
	waitFor(t, "conversation", func() bool { return r.conversations.Load() == 1 })
	if r.vm.Result() != (Result{ClassHuman, SourceTiming}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
}

func TestTimingDecidesMachineDuringLongSpeech(t *testing.T) {
	c := &fakeClassifier{reply: "VOICEMAIL"}
	r := newRig(t, c, WithTimingHeuristic(TimingThresholds{MachineSpeech: 100 * time.Millisecond}),
		WithDecisionTimeout(time.Minute), WithVoicemailResponseDelay(20*time.Millisecond)).start()
	r.send(frames.NewTranscriptionFrame("you have reached the voicemail of", true))
	waitFor(t, "classification", func() bool {
		r.d.mu.Lock()
		defer r.d.mu.Unlock()
		return r.d.last.Class == ClassVoicemail
	})
	r.vad(true, time.Now())
	waitFor(t, "interruption", func() bool { return r.out.interruptions() > 0 })
	if r.vm.Result() != (Result{ClassVoicemail, SourceTiming}) {
		t.Fatalf("result %+v", r.vm.Result())
	}
	r.vad(false, time.Now())
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
}

func TestTimingLongSpeechAloneDoesNotDecide(t *testing.T) {
	c := &fakeClassifier{reply: "CONVERSATION", release: make(chan struct{})}
	r := newRig(t, c, WithTimingHeuristic(TimingThresholds{}), WithDecisionTimeout(50*time.Millisecond)).start()
	r.vad(true, time.Now().Add(-4*time.Second))
	r.send(frames.NewTranscriptionFrame("Hello, this is Rahul from ABC Enterprises, who is calling please?", true))
	r.vad(false, time.Now())
	settle()
	if r.vm.Decision() != Undecided {
		t.Fatalf("talkative human decided by timing alone: %+v", r.vm.Result())
	}
	close(c.release)
	waitFor(t, "conversation", func() bool { return r.conversations.Load() == 1 })
	settle()
	if r.vm.Result() != (Result{ClassHuman, SourceClassifier}) || r.voicemails.Load() != 0 || r.out.interruptions() != 0 {
		t.Fatalf("result %+v voicemails %d interruptions %d", r.vm.Result(), r.voicemails.Load(), r.out.interruptions())
	}
}

func TestTimingLongSpeechWaitsForClassifierThenDecides(t *testing.T) {
	c := &fakeClassifier{reply: "VOICEMAIL", release: make(chan struct{})}
	r := newRig(t, c, WithTimingHeuristic(TimingThresholds{}), WithDecisionTimeout(time.Minute)).start()
	r.vad(true, time.Now().Add(-4*time.Second))
	r.send(frames.NewTranscriptionFrame("Sorry I can't take your call right now", true))
	settle()
	if r.vm.Decision() != Undecided {
		t.Fatalf("decided before the classifier answered: %+v", r.vm.Result())
	}
	close(c.release)
	waitFor(t, "voicemail verdict", func() bool { return r.vm.Decision() == Voicemail })
	if r.vm.Result() != (Result{ClassVoicemail, SourceTiming}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
}

func TestTimingDefersWhenClassifierDisagrees(t *testing.T) {
	c := &fakeClassifier{reply: "VOICEMAIL"}
	r := newRig(t, c, WithTimingHeuristic(TimingThresholds{}), WithDecisionTimeout(500*time.Millisecond),
		WithVoicemailResponseDelay(20*time.Millisecond)).start()
	r.send(frames.NewTranscriptionFrame("hi this is Sam", true))
	waitFor(t, "classification", func() bool {
		r.d.mu.Lock()
		defer r.d.mu.Unlock()
		return r.d.last.Class == ClassVoicemail
	})
	now := time.Now()
	r.vad(true, now.Add(-1500*time.Millisecond))
	r.vad(false, now.Add(-900*time.Millisecond))
	if r.vm.Decision() != Undecided {
		t.Fatalf("timing overrode the classifier: %+v", r.vm.Result())
	}
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
	if r.vm.Result() != (Result{ClassVoicemail, SourceClassifier}) || r.conversations.Load() != 0 {
		t.Fatalf("result %+v conversations %d", r.vm.Result(), r.conversations.Load())
	}
}

func TestTimingMachineTakesClassifierClass(t *testing.T) {
	c := &fakeClassifier{reply: "ANNOUNCEMENT"}
	r := newRig(t, c, WithDetailedClasses(), WithTimingHeuristic(TimingThresholds{}), WithDecisionTimeout(time.Minute)).start()
	r.send(frames.NewTranscriptionFrame("the number you are calling is switched off", true))
	waitFor(t, "classification", func() bool {
		r.d.mu.Lock()
		defer r.d.mu.Unlock()
		return r.d.last.Class == ClassAnnouncement
	})
	r.vad(true, time.Now().Add(-4*time.Second))
	waitFor(t, "verdict", func() bool { return r.vm.Decision() == Voicemail })
	if r.vm.Result() != (Result{ClassAnnouncement, SourceTiming}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
}

func TestBeepLeavesMessageAfterTone(t *testing.T) {
	for _, rate := range []int{8000, 16000} {
		c := &fakeClassifier{reply: "VOICEMAIL"}
		r := newRig(t, c, WithBeepDetection(2*time.Second), WithDecisionTimeout(30*time.Millisecond)).start()
		r.say("Sorry I missed your call, leave a message after the tone")
		waitFor(t, "voicemail verdict", func() bool { return r.vm.Decision() == Voicemail })
		r.audio(rate, make([]byte, 2*rate/5))
		r.audio(rate, tonePCM(rate, 1000, 500*time.Millisecond))
		if r.voicemails.Load() != 0 {
			t.Fatalf("rate %d: message started during the tone", rate)
		}
		start := time.Now()
		r.audio(rate, make([]byte, 2*rate/25))
		waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
		if wait := time.Duration(r.voicemailAt.Load() - start.UnixNano()); wait > 500*time.Millisecond {
			t.Fatalf("rate %d: message waited %v after the tone", rate, wait)
		}
		r.audio(rate, tonePCM(rate, 1000, 500*time.Millisecond))
		r.audio(rate, make([]byte, 2*rate/25))
		settle()
		if r.voicemails.Load() != 1 {
			t.Fatalf("rate %d: handler ran %d times", rate, r.voicemails.Load())
		}
	}
}

func TestBeepFallbackTimeout(t *testing.T) {
	c := &fakeClassifier{reply: "VOICEMAIL"}
	r := newRig(t, c, WithBeepDetection(120*time.Millisecond), WithVoicemailResponseDelay(time.Minute),
		WithDecisionTimeout(30*time.Millisecond)).start()
	r.say("Please leave a message")
	waitFor(t, "voicemail verdict", func() bool { return r.vm.Decision() == Voicemail })
	decided := time.Now()
	r.audio(8000, make([]byte, 1600))
	waitFor(t, "fallback", func() bool { return r.voicemails.Load() == 1 })
	if wait := time.Since(decided); wait < 100*time.Millisecond {
		t.Fatalf("fallback ran after %v", wait)
	}
}

func TestBeepBeforeClassificationDecidesVoicemail(t *testing.T) {
	c := &fakeClassifier{reply: "CONVERSATION", release: make(chan struct{})}
	r := newRig(t, c, WithBeepDetection(time.Minute)).start()
	r.audio(8000, tonePCM(8000, 850, 500*time.Millisecond))
	if r.vm.Result() != (Result{ClassVoicemail, SourceBeep}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
	r.audio(8000, make([]byte, 640))
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
	if r.conversations.Load() != 0 {
		t.Fatal("conversation callback ran")
	}
}

func TestWeakBeepAloneDoesNotDecide(t *testing.T) {
	r := newRig(t, &fakeClassifier{}, WithBeepDetection(time.Minute)).start()
	r.audio(8000, weakBeepPCM(8000, 600*time.Millisecond))
	r.audio(8000, make([]byte, 640))
	settle()
	if r.vm.Decision() != Undecided || r.voicemails.Load() != 0 {
		t.Fatalf("weak beep decided %+v", r.vm.Result())
	}
}

func TestWeakBeepCorroboratedByGreeting(t *testing.T) {
	r := newRig(t, &fakeClassifier{}, WithBeepDetection(time.Minute)).start()
	now := time.Now()
	r.vad(true, now.Add(-3*time.Second))
	r.vad(false, now.Add(-500*time.Millisecond))
	r.audio(8000, weakBeepPCM(8000, 600*time.Millisecond))
	if r.vm.Result() != (Result{ClassVoicemail, SourceBeep}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
	r.audio(8000, make([]byte, 640))
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
}

func TestWeakBeepCorroboratedByClassifier(t *testing.T) {
	r := newRig(t, &fakeClassifier{reply: "VOICEMAIL"}, WithBeepDetection(time.Minute), WithDecisionTimeout(time.Minute)).start()
	r.send(frames.NewTranscriptionFrame("leave a message after the tone", true))
	waitFor(t, "classification", func() bool {
		r.d.mu.Lock()
		defer r.d.mu.Unlock()
		return r.d.last.Class == ClassVoicemail
	})
	r.audio(8000, weakBeepPCM(8000, 600*time.Millisecond))
	if r.vm.Result() != (Result{ClassVoicemail, SourceBeep}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
}

func TestBeepBeforeVoicemailVerdictLeavesMessageAtVerdict(t *testing.T) {
	r := newRig(t, &fakeClassifier{reply: "VOICEMAIL"}, WithBeepDetection(time.Minute), WithDecisionTimeout(30*time.Millisecond)).start()
	r.audio(8000, weakBeepPCM(8000, 600*time.Millisecond))
	r.audio(8000, make([]byte, 640))
	if r.vm.Decision() != Undecided {
		t.Fatalf("decided early: %+v", r.vm.Result())
	}
	r.say("leave a message")
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
	if r.vm.Result() != (Result{ClassVoicemail, SourceClassifier}) || r.out.interruptions() == 0 {
		t.Fatalf("result %+v interruptions %d", r.vm.Result(), r.out.interruptions())
	}
}

func TestBeepIgnoresDTMF(t *testing.T) {
	r := newRig(t, &fakeClassifier{}, WithBeepDetection(time.Minute)).start()
	seq, err := dtmf.Sequence("159#", 8000)
	if err != nil {
		t.Fatal(err)
	}
	r.audio(8000, seq)
	if r.vm.Decision() != Undecided {
		t.Fatalf("DTMF decided %+v", r.vm.Result())
	}
}

func TestCarrierResult(t *testing.T) {
	r := newRig(t, &fakeClassifier{}, WithVoicemailResponseDelay(20*time.Millisecond))
	if !r.vm.ReportCarrierResult(ClassFromAnsweredBy("machine_end_beep")) {
		t.Fatal("carrier result before start was refused")
	}
	if r.vm.Decision() != Undecided {
		t.Fatal("decided before start")
	}
	r.start()
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
	waitFor(t, "interruption", func() bool { return r.out.interruptions() > 0 })
	if r.vm.Result() != (Result{ClassVoicemail, SourceCarrier}) {
		t.Fatalf("result = %+v", r.vm.Result())
	}
	if r.vm.ReportCarrierResult(ClassHuman) || r.conversations.Load() != 0 {
		t.Fatal("second decision accepted")
	}

	h := newRig(t, &fakeClassifier{}).start()
	if h.vm.ReportCarrierResult(ClassFromAnsweredBy("fax")) {
		t.Fatal("unknown carrier result accepted")
	}
	if !h.vm.ReportCarrierResult(ClassFromAnsweredBy("human")) || h.conversations.Load() != 1 {
		t.Fatal("human carrier result not applied")
	}
}

func TestCarrierResultWaitsForStartFrame(t *testing.T) {
	r := newRig(t, &fakeClassifier{})
	r.vm.ReportCarrierResult(ClassHuman)
	if err := r.d.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = r.d.Stop() })
	settle()
	if r.vm.Decision() != Undecided || r.conversations.Load() != 0 {
		t.Fatal("carrier result acted before the start frame")
	}
	r.send(frames.NewStartFrame())
	if r.vm.Decision() != Conversation || r.conversations.Load() != 1 {
		t.Fatalf("decision %s conversations %d", r.vm.Decision(), r.conversations.Load())
	}
	r.out.mu.Lock()
	first := r.out.frames[0]
	r.out.mu.Unlock()
	if _, ok := first.(*frames.StartFrame); !ok {
		t.Fatalf("first frame downstream was %T", first)
	}
}

func TestAnsweredByMachineEndBeepLeavesMessageAtOnce(t *testing.T) {
	r := newRig(t, &fakeClassifier{}, WithBeepDetection(time.Minute), WithVoicemailResponseDelay(time.Minute))
	if !r.vm.ReportAnsweredBy("machine_end_beep") {
		t.Fatal("answered-by refused")
	}
	r.start()
	waitFor(t, "voicemail handler", func() bool { return r.voicemails.Load() == 1 })
	if r.vm.Result() != (Result{ClassVoicemail, SourceCarrier}) || r.out.interruptions() == 0 {
		t.Fatalf("result %+v interruptions %d", r.vm.Result(), r.out.interruptions())
	}

	s := newRig(t, &fakeClassifier{}, WithBeepDetection(time.Minute)).start()
	s.vm.ReportAnsweredBy("machine_start")
	settle()
	if s.vm.Decision() != Voicemail || s.voicemails.Load() != 0 {
		t.Fatalf("machine_start: decision %s voicemails %d", s.vm.Decision(), s.voicemails.Load())
	}
	if !s.vm.ReportAnsweredBy("machine_end_beep") {
		t.Fatal("later machine_end_beep refused")
	}
	if s.voicemails.Load() != 1 {
		t.Fatalf("voicemails %d", s.voicemails.Load())
	}
}

func TestGreetingOver(t *testing.T) {
	for in, want := range map[string]bool{"machine_end_beep": true, "machine_end_silence": true, "MACHINE_END_OTHER": true, "machine_start": false, "human": false, "": false} {
		if got := GreetingOver(in); got != want {
			t.Errorf("GreetingOver(%q) = %v", in, got)
		}
	}
}
