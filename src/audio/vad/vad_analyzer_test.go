package vad

import (
	"testing"
)

func TestVADParams_Defaults(t *testing.T) {
	params := DefaultVADParams()

	// Verify default StopSecs is 0.2 (not 0.8)
	if params.StopSecs != 0.2 {
		t.Fatalf("expected StopSecs default to be 0.2, got %.1f", params.StopSecs)
	}

	// Verify other defaults are correct
	if params.Confidence != 0.7 {
		t.Fatalf("expected Confidence default to be 0.7, got %.1f", params.Confidence)
	}

	if params.StartSecs != 0.2 {
		t.Fatalf("expected StartSecs default to be 0.2, got %.1f", params.StartSecs)
	}

	if params.MinVolume != 0.1 {
		t.Fatalf("expected MinVolume default to be 0.1, got %.1f", params.MinVolume)
	}
}
