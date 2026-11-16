package voicemail

import "time"

const (
	DefaultHumanMaxGreeting = 1500 * time.Millisecond
	DefaultHumanSilence     = 800 * time.Millisecond
	DefaultMachineSpeech    = 3 * time.Second
)

type TimingThresholds struct {
	HumanMaxGreeting time.Duration
	HumanSilence     time.Duration
	MachineSpeech    time.Duration
}

func DefaultTimingThresholds() TimingThresholds {
	return TimingThresholds{
		HumanMaxGreeting: DefaultHumanMaxGreeting,
		HumanSilence:     DefaultHumanSilence,
		MachineSpeech:    DefaultMachineSpeech,
	}
}

func (t TimingThresholds) withDefaults() TimingThresholds {
	def := DefaultTimingThresholds()
	if t.HumanMaxGreeting <= 0 {
		t.HumanMaxGreeting = def.HumanMaxGreeting
	}
	if t.HumanSilence <= 0 {
		t.HumanSilence = def.HumanSilence
	}
	if t.MachineSpeech <= 0 {
		t.MachineSpeech = def.MachineSpeech
	}
	return t
}

type timing struct {
	th          TimingThresholds
	speaking    bool
	spoke       bool
	runs        int
	runStart    time.Time
	speechStart time.Time
	lastEnd     time.Time
	total       time.Duration
}

func (t *timing) start(at time.Time) {
	if t.speaking {
		return
	}
	if !t.spoke || at.Sub(t.lastEnd) >= t.th.HumanSilence {
		t.runs++
		t.runStart = at
	}
	t.speaking = true
	t.spoke = true
	t.speechStart = at
}

func (t *timing) stop(at time.Time) {
	if !t.speaking {
		return
	}
	if at.Before(t.speechStart) {
		at = t.speechStart
	}
	t.speaking = false
	t.total += at.Sub(t.speechStart)
	t.lastEnd = at
}

func (t *timing) evaluate(now time.Time) (Class, time.Time) {
	if !t.spoke {
		return ClassUnknown, time.Time{}
	}
	machineAt := t.runStart.Add(t.th.MachineSpeech)
	if t.speaking {
		if !now.Before(machineAt) {
			return ClassVoicemail, time.Time{}
		}
		return ClassUnknown, machineAt
	}
	if !t.lastEnd.Before(machineAt) {
		return ClassVoicemail, time.Time{}
	}
	quietAt := t.lastEnd.Add(t.th.HumanSilence)
	if now.Before(quietAt) {
		return ClassUnknown, quietAt
	}
	if t.runs == 1 && t.total <= t.th.HumanMaxGreeting {
		return ClassHuman, time.Time{}
	}
	return ClassUnknown, time.Time{}
}
