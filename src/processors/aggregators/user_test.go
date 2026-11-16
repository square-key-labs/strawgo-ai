package aggregators

import (
	"context"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type neverStop struct{}

func (neverStop) ShouldStop(any) bool { return false }
func (neverStop) Reset()              {}

func startUserAggregator(t *testing.T, c *llm.Context, params UserAggregatorParams, allowInterruptions bool) (*LLMUserAggregator, *captureProc, *captureProc) {
	t.Helper()
	u := NewLLMUserAggregatorWithParams(c, params)
	down, up := &captureProc{}, &captureProc{}
	u.Link(down)
	u.SetPrev(up)
	if err := u.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = u.Stop() })
	_ = u.QueueFrame(frames.NewStartFrameWithConfig(allowInterruptions, turns.UserTurnStrategies{}), frames.Downstream)
	return u, down, up
}

func transcriptionStrategies(timeout time.Duration) turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewTranscriptionUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewSpeechTimeoutUserTurnStopStrategy(timeout)},
	}
}

func countNamed(fs []frames.Frame, name string) int {
	n := 0
	for _, f := range fs {
		if f.Name() == name {
			n++
		}
	}
	return n
}

func TestUserAggregator_InterimNotLeakedAndNotDuplicated(t *testing.T) {
	c := llm.NewContext()
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: transcriptionStrategies(30 * time.Millisecond)}, true)

	for i := 0; i < 3; i++ {
		_ = u.QueueFrame(frames.NewTranscriptionFrame("hello world", false), frames.Downstream)
		_ = u.QueueFrame(frames.NewInterimTranscriptionFrame("hello world", ""), frames.Downstream)
	}
	_ = u.QueueFrame(frames.NewTranscriptionFrame("hello world", true), frames.Downstream)

	down.waitFor(t, "LLMContextFrame", time.Second)
	got := down.get()
	if countNamed(got, "TranscriptionFrame")+countNamed(got, "InterimTranscriptionFrame") != 0 {
		t.Fatal("expected transcriptions to be consumed")
	}
	msgs := c.Messages()
	if len(msgs) != 1 || msgs[0].Role != llm.RoleUser || msgs[0].Text() != "hello world" {
		t.Fatalf("expected exactly one user message, got %+v", msgs)
	}
}

func TestUserAggregator_StopStrategyEndsTurnWithoutPolling(t *testing.T) {
	c := llm.NewContext()
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: transcriptionStrategies(40 * time.Millisecond)}, true)

	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("book a table", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	time.Sleep(15 * time.Millisecond)
	if countNamed(down.get(), "LLMContextFrame") != 0 {
		t.Fatal("expected the turn to stay open until the speech timeout")
	}
	down.waitFor(t, "LLMContextFrame", time.Second)
	if c.Len() != 1 {
		t.Fatalf("expected the user message in context, got %d", c.Len())
	}
}

func TestUserAggregator_InterruptsWhenBotSpeaking(t *testing.T) {
	c := llm.NewContext()
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewVADUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{neverStop{}},
	}
	u, down, up := startUserAggregator(t, c, UserAggregatorParams{Strategies: strategies}, true)

	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	up.waitFor(t, "InterruptionFrame", time.Second)
}

func TestUserAggregator_NoInterruptionWhenDisallowed(t *testing.T) {
	c := llm.NewContext()
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewVADUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{neverStop{}},
	}
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: strategies}, false)

	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "UserStartedSpeakingFrame", time.Second)
	if countNamed(down.get(), "InterruptionFrame") != 0 {
		t.Fatal("expected no interruption when interruptions are disallowed")
	}
}

func TestUserAggregator_MuteSurvivesInterruption(t *testing.T) {
	c := llm.NewContext()
	strategies := transcriptionStrategies(20 * time.Millisecond)
	strategies.MuteStrategies = []user_mute.UserMuteStrategy{user_mute.NewFirstSpeechUserMuteStrategy()}
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: strategies}, true)

	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	down.waitFor(t, "UserMuteStartedFrame", time.Second)
	_ = u.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("ignored while muted", true), frames.Downstream)
	time.Sleep(60 * time.Millisecond)
	if countNamed(down.get(), "UserMuteStoppedFrame") != 0 || c.Len() != 0 {
		t.Fatal("expected mute state and suppression to survive an interruption")
	}

	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	down.waitFor(t, "UserMuteStoppedFrame", time.Second)
}

func TestUserAggregator_StopTimeoutEndsStuckTurn(t *testing.T) {
	c := llm.NewContext()
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewTranscriptionUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{neverStop{}},
	}
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: strategies, UserTurnStopTimeout: 40 * time.Millisecond}, true)

	_ = u.QueueFrame(frames.NewTranscriptionFrame("are you there", true), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)
	if c.Len() != 1 {
		t.Fatalf("expected the stuck turn to be committed, got %d messages", c.Len())
	}
}

func TestUserAggregator_UpstreamContextTriggersInference(t *testing.T) {
	c := llm.NewContext(llm.User("hi"))
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{}, true)

	_ = u.QueueFrame(frames.NewLLMContextFrame(c), frames.Upstream)
	_ = u.QueueFrame(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.System("nudge")}, true), frames.Upstream)
	deadline := time.Now().Add(time.Second)
	for countNamed(down.get(), "LLMContextFrame") < 2 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if countNamed(down.get(), "LLMContextFrame") != 2 || c.Len() != 2 {
		t.Fatalf("expected two downstream inference requests and an appended message")
	}
}

func TestUserAggregator_IdleTimeout(t *testing.T) {
	c := llm.NewContext()
	idle := make(chan struct{}, 1)
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{
		UserIdleTimeout: 30 * time.Millisecond,
		OnUserIdle:      func() { idle <- struct{}{} },
	}, true)

	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	down.waitFor(t, "UserIdleTimeoutFrame", time.Second)
	<-idle

	_ = u.QueueFrame(frames.NewUserIdleTimeoutUpdateFrame(0), frames.Downstream)
	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	time.Sleep(80 * time.Millisecond)
	if countNamed(down.get(), "UserIdleTimeoutFrame") != 1 {
		t.Fatal("expected idle detection to be disabled by the update frame")
	}
}

func TestUserAggregator_MutedSpeakingStateReachesAssistant(t *testing.T) {
	c := llm.NewContext(llm.User("what's the weather"))
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewVADUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{neverStop{}},
		MuteStrategies:  []user_mute.UserMuteStrategy{user_mute.NewFunctionCallUserMuteStrategy()},
	}
	u := NewLLMUserAggregatorWithParams(c, UserAggregatorParams{Strategies: strategies})
	a := NewLLMAssistantAggregator(c, nil)
	down, up := &captureProc{}, &captureProc{}
	u.SetPrev(up)
	u.Link(a)
	a.SetPrev(u)
	a.Link(down)
	for _, p := range []interface{ Start(context.Context) error }{u, a} {
		if err := p.Start(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { _ = u.Stop(); _ = a.Stop() })

	_ = u.QueueFrame(frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{}), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "UserStartedSpeakingFrame", time.Second)
	_ = u.QueueFrame(frames.NewFunctionCallsStartedFrame([]frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "weather"}}), frames.Downstream)
	down.waitFor(t, "UserMuteStartedFrame", time.Second)
	_ = u.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "weather", nil, false), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "UserStoppedSpeakingFrame", time.Second)
	_ = u.QueueFrame(frames.NewFunctionCallResultFrame("1", "weather", "sunny", nil), frames.Downstream)

	down.waitFor(t, "LLMContextFrame", time.Second)
}

type metricsAnalyzer struct{}

func (metricsAnalyzer) AppendAudio([]byte, bool) turn.EndOfTurnState { return turn.TurnIncomplete }
func (metricsAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, *turn.TurnMetrics, error) {
	return turn.TurnComplete, &turn.TurnMetrics{IsComplete: true, Probability: 0.9, TotalTimeMs: 40}, nil
}
func (metricsAnalyzer) SpeechTriggered() bool { return false }
func (metricsAnalyzer) SetSampleRate(int)     {}
func (metricsAnalyzer) Clear()                {}

func TestUserAggregator_TurnAnalyzerMetricsArePushed(t *testing.T) {
	c := llm.NewContext()
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewTranscriptionUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewTurnAnalyzerUserTurnStopStrategy(metricsAnalyzer{})},
	}}, true)

	_ = u.QueueFrame(frames.NewTranscriptionFrame("book a table", true), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)

	var got []frames.TurnMetricsData
	for _, f := range down.get() {
		if m, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range m.Data {
				if tm, ok := d.(frames.TurnMetricsData); ok {
					got = append(got, tm)
				}
			}
		}
	}
	if len(got) != 1 {
		t.Fatalf("expected one turn metrics entry, got %d", len(got))
	}
	want := frames.TurnMetricsData{
		MetricsBase:       frames.MetricsBase{Processor: u.Name()},
		IsComplete:        true,
		Probability:       0.9,
		E2EProcessingTime: 40 * time.Millisecond,
	}
	if got[0] != want {
		t.Fatalf("turn metrics = %+v, want %+v", got[0], want)
	}
}

func sendPause(u *LLMUserAggregator, text string) {
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame(text, true), frames.Downstream)
	_ = u.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2), frames.Downstream)
}

func TestUserAggregator_DefaultStrategiesWaitForTurnAnalyzer(t *testing.T) {
	c := llm.NewContext()
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{}, true)

	_ = u.QueueFrame(frames.NewSpeechControlParamsFrame(&frames.VADParams{StopSecs: 0.2}, turn.TurnAnalyzerParams{StopSecs: 3}), frames.Downstream)
	sendPause(u, "my number is")
	time.Sleep(user_stop.DefaultUserSpeechTimeout + 200*time.Millisecond)
	if countNamed(down.get(), "LLMContextFrame") != 0 {
		t.Fatal("expected the turn to stay open while the turn analyzer says incomplete")
	}

	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)
	if msgs := c.Messages(); len(msgs) != 1 || msgs[0].Text() != "my number is" {
		t.Fatalf("expected one user message, got %+v", msgs)
	}
}

func TestUserAggregator_DefaultStrategiesWithoutTurnAnalyzerEndAtPause(t *testing.T) {
	c := llm.NewContext()
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{}, true)

	_ = u.QueueFrame(frames.NewSpeechControlParamsFrame(&frames.VADParams{StopSecs: 0.2}, nil), frames.Downstream)
	sendPause(u, "my number is")
	down.waitFor(t, "LLMContextFrame", 2*time.Second)
	if c.Len() != 1 {
		t.Fatalf("expected the user message in context, got %d", c.Len())
	}
}

func TestDTMFAggregator_InterruptsOnlyWhenAllowed(t *testing.T) {
	for _, allow := range []bool{true, false} {
		a := NewDTMFAggregator(WithDTMFTimeout(0))
		down, up := &captureProc{}, &captureProc{}
		a.Link(down)
		a.SetPrev(up)
		if err := a.Start(context.Background()); err != nil {
			t.Fatal(err)
		}
		_ = a.QueueFrame(frames.NewStartFrameWithConfig(allow, turns.UserTurnStrategies{}), frames.Downstream)
		_ = a.QueueFrame(frames.NewInputDTMFFrame(frames.KeypadOne), frames.Downstream)
		_ = a.QueueFrame(frames.NewInputDTMFFrame(frames.KeypadPound), frames.Downstream)
		down.waitFor(t, "TranscriptionFrame", time.Second)
		got := countNamed(down.get(), "InterruptionFrame") + countNamed(up.get(), "InterruptionFrame")
		_ = a.Stop()
		if want := map[bool]int{true: 2, false: 0}[allow]; got != want {
			t.Fatalf("AllowInterruptions=%v: got %d InterruptionFrames, want %d", allow, got, want)
		}
	}
}
