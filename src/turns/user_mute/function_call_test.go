package user_mute_test

import (
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
)

func TestFunctionCallMuteStrategy(t *testing.T) {
	strategy := user_mute.NewFunctionCallUserMuteStrategy(false)

	if strategy.ShouldMute(frames.NewTextFrame("idle")) {
		t.Fatalf("expected no mute before function call")
	}

	start := frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "lookup"}})
	if !strategy.ShouldMute(start) {
		t.Fatalf("expected mute after function call starts")
	}

	result := frames.NewFunctionCallResultFrame("1", "lookup", "ok", nil)
	if strategy.ShouldMute(result) {
		t.Fatalf("expected unmute after function call result")
	}

	strategy.Reset()
	if strategy.ShouldMute(frames.NewTextFrame("after reset")) {
		t.Fatalf("expected no mute after reset")
	}

	if strategy.EnableInterruptions() {
		t.Fatalf("expected enable interruptions to be false")
	}
}
