package user_stop

import (
	"testing"
)

// stubFrame implements namedFrame for test-only signaling.
type stubFrame struct{ n string }

func (s stubFrame) Name() string { return s.n }

// boolStopper is the legacy bool-result interface only.
type boolStopper struct {
	want bool
}

func (b *boolStopper) ShouldStop(_ any) bool { return b.want }
func (b *boolStopper) Reset()                {}

// v2Stopper implements both legacy and V2 interfaces. The V2 result
// drives behavior; ShouldStop returns the bool projection.
type v2Stopper struct {
	want StopResult
}

func (v *v2Stopper) ShouldStop(_ any) bool {
	return v.ShouldStopV2(nil) != StopResultContinue
}
func (v *v2Stopper) ShouldStopV2(_ any) StopResult { return v.want }
func (v *v2Stopper) Reset()                        {}

func TestStopResultEnumOrdering(t *testing.T) {
	if StopResultContinue != 0 {
		t.Fatalf("StopResultContinue should be 0")
	}
	if StopResultStopShortCircuit == StopResultStop {
		t.Fatalf("StopResultStopShortCircuit and Stop must differ")
	}
}

func TestV2StopperImplementsV1(t *testing.T) {
	var _ UserTurnStopStrategy = &v2Stopper{want: StopResultContinue}
	var _ UserTurnStopStrategyV2 = &v2Stopper{want: StopResultStop}
}

func TestV2BoolProjection(t *testing.T) {
	if (&v2Stopper{want: StopResultContinue}).ShouldStop(stubFrame{}) {
		t.Fatal("Continue must project to false")
	}
	if !(&v2Stopper{want: StopResultStop}).ShouldStop(stubFrame{}) {
		t.Fatal("Stop must project to true")
	}
	if !(&v2Stopper{want: StopResultStopShortCircuit}).ShouldStop(stubFrame{}) {
		t.Fatal("ShortCircuit must project to true")
	}
}

func TestV1OnlyStopperStillSatisfiesContract(t *testing.T) {
	var _ UserTurnStopStrategy = &boolStopper{}
}
