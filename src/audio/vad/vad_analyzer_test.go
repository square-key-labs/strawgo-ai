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

// processN calls ProcessAudio n times with the given confidence and a silent buffer.
// numFramesRequired and sampleRate determine the threshold calculations.
func processN(v *BaseVADAnalyzer, confidence float32, n int) VADState {
	// 512 zero bytes = 256 int16 samples of silence (valid 16kHz frame)
	buf := make([]byte, 512)
	var state VADState
	for range n {
		s, _ := v.ProcessAudio(buf, confidence, 512)
		state = s
	}
	return state
}

func newTestAnalyzer(startSecs, stopSecs, confThresh float32) *BaseVADAnalyzer {
	params := VADParams{
		Confidence: confThresh,
		StartSecs:  startSecs,
		StopSecs:   stopSecs,
		MinVolume:  0.0, // disable volume gate in unit tests
	}
	// Use 16000 Hz; 512 samples/frame → frameTime = 32ms
	return NewBaseVADAnalyzer(16000, params)
}

// TestVADStateMachine_QuietToSpeaking verifies QUIET→STARTING→SPEAKING transitions.
func TestVADStateMachine_QuietToSpeaking(t *testing.T) {
	// startSecs=0.064s, frameTime=32ms → startThreshold=2
	// stopSecs=0.2s (irrelevant here)
	v := newTestAnalyzer(0.064, 0.2, 0.7)

	// QUIET initially
	if v.GetState() != VADStateQuiet {
		t.Fatalf("expected initial QUIET, got %s", v.GetState())
	}

	// 1 voice frame → STARTING
	state := processN(v, 0.9, 1)
	if state != VADStateStarting {
		t.Errorf("after 1 voice frame: expected STARTING, got %s", state)
	}

	// 2nd voice frame → SPEAKING (startThreshold=2)
	state = processN(v, 0.9, 1)
	if state != VADStateSpeaking {
		t.Errorf("after 2 voice frames: expected SPEAKING, got %s", state)
	}
}

// TestVADStateMachine_SpeakingToQuiet verifies SPEAKING→STOPPING→QUIET transitions.
func TestVADStateMachine_SpeakingToQuiet(t *testing.T) {
	// startSecs=0.032s → startThreshold=1; stopSecs=0.064s → stopThreshold=2
	v := newTestAnalyzer(0.032, 0.064, 0.7)

	// Reach SPEAKING in 1 voice frame
	state := processN(v, 0.9, 1)
	if state != VADStateSpeaking {
		t.Fatalf("setup: expected SPEAKING, got %s", state)
	}

	// 1 silent frame → STOPPING (stopThreshold=2, stopFrames=1)
	state = processN(v, 0.0, 1)
	if state != VADStateStopping {
		t.Errorf("1 silent frame: expected STOPPING, got %s", state)
	}

	// 2nd silent frame → QUIET (stopFrames=2 >= stopThreshold=2)
	state = processN(v, 0.0, 1)
	if state != VADStateQuiet {
		t.Errorf("2 silent frames: expected QUIET, got %s (possible VAD stuck in SPEAKING/STOPPING)", state)
	}
}

// TestVADStateMachine_NoStuckInSpeaking is the regression test for the Pipecat
// "VAD stuck in SPEAKING" bug. After audio stops, the VAD must eventually reach
// QUIET regardless of how many silent frames arrive.
func TestVADStateMachine_NoStuckInSpeaking(t *testing.T) {
	// Use default stopSecs=0.2 → stopThreshold=6 (at 32ms/frame)
	v := newTestAnalyzer(0.032, 0.2, 0.7)

	// Reach SPEAKING
	processN(v, 0.9, 1)
	if v.GetState() != VADStateSpeaking {
		t.Fatalf("setup failed: expected SPEAKING, got %s", v.GetState())
	}

	// Feed 20 silent frames (well over stopThreshold=6) — must reach QUIET
	finalState := processN(v, 0.0, 20)
	if finalState != VADStateQuiet {
		t.Errorf("after 20 silent frames: expected QUIET, got %s (VAD stuck!)", finalState)
	}
}

// TestVADStateMachine_StoppingResumption verifies STOPPING→SPEAKING when voice resumes.
func TestVADStateMachine_StoppingResumption(t *testing.T) {
	// startThreshold=1, stopThreshold=3
	v := newTestAnalyzer(0.032, 0.096, 0.7)

	// Reach SPEAKING
	processN(v, 0.9, 1)

	// 1 silent frame → STOPPING
	state := processN(v, 0.0, 1)
	if state != VADStateStopping {
		t.Fatalf("expected STOPPING after 1 silent frame, got %s", state)
	}

	// Voice resumes → back to SPEAKING
	state = processN(v, 0.9, 1)
	if state != VADStateSpeaking {
		t.Errorf("voice resumed: expected SPEAKING, got %s", state)
	}
}

// TestVADStateMachine_Restart verifies that Restart() clears all state.
func TestVADStateMachine_Restart(t *testing.T) {
	v := newTestAnalyzer(0.032, 0.2, 0.7)

	// Reach SPEAKING
	processN(v, 0.9, 1)
	if v.GetState() != VADStateSpeaking {
		t.Fatalf("setup: expected SPEAKING")
	}

	v.Restart()

	if v.GetState() != VADStateQuiet {
		t.Errorf("after Restart: expected QUIET, got %s", v.GetState())
	}

	// After restart, a single voice frame should go to STARTING (startThreshold=1)
	// i.e. thresholds still work after reset
	state := processN(v, 0.9, 1)
	if state != VADStateSpeaking {
		t.Errorf("after Restart + 1 voice frame: expected SPEAKING, got %s", state)
	}
}
