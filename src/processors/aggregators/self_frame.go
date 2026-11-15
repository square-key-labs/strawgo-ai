package aggregators

import "github.com/square-key-labs/strawgo-ai/src/frames"

type selfFrameKind int

const (
	selfTurnTimer selfFrameKind = iota
	selfUserIdle
	selfSummary
	selfDTMFTimeout
)

type selfFrame struct {
	*frames.ControlFrame
	kind    selfFrameKind
	gen     uint64
	payload any
}

func newSelfFrame(kind selfFrameKind, gen uint64, payload any) *selfFrame {
	return &selfFrame{
		ControlFrame: &frames.ControlFrame{BaseFrame: frames.NewUninterruptibleBaseFrame("AggregatorTimerFrame")},
		kind:         kind,
		gen:          gen,
		payload:      payload,
	}
}
