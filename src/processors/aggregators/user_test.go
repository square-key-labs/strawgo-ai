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

func emptyTurnStrategies() turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewVADUserTurnStartStrategy(true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{neverStop{}},
	}
}

func startEmptyTurnPair(t *testing.T, c *llm.Context, cfg *turns.EmptyUserTurnConfig) (*LLMUserAggregator, *LLMAssistantAggregator, *captureProc, *captureProc) {
	t.Helper()
	pair := NewContextAggregatorPair(c, ContextAggregatorPairParams{User: UserAggregatorParams{
		Strategies:          emptyTurnStrategies(),
		UserTurnStopTimeout: 30 * time.Millisecond,
		EmptyUserTurn:       cfg,
	}})
	u, a := pair.User(), pair.Assistant()
	userDown, asstDown := &captureProc{}, &captureProc{}
	u.Link(userDown)
	u.SetPrev(&captureProc{})
	a.Link(asstDown)
	a.SetPrev(&captureProc{})
	for _, p := range []interface{ Start(context.Context) error }{u, a} {
		if err := p.Start(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { _ = u.Stop(); _ = a.Stop() })
	_ = u.QueueFrame(frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{}), frames.Downstream)
	return u, a, userDown, asstDown
}

func userTurn(u *LLMUserAggregator, transcript string) {
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	if transcript != "" {
		_ = u.QueueFrame(frames.NewTranscriptionFrame(transcript, true), frames.Downstream)
	}
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
}

func waitTurnEnd(t *testing.T, down *captureProc, n int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for countNamed(down.get(), "UserStoppedSpeakingFrame") < n {
		if time.Now().After(deadline) {
			t.Fatal("user turn did not end")
		}
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(60 * time.Millisecond)
}

func speakWords(a *LLMAssistantAggregator, contextID string, words ...string) {
	for _, w := range words {
		_ = a.QueueFrame(frames.NewTTSTextFrame(w, frames.TextTypeWord, contextID), frames.Downstream)
	}
}

func stopTTS(a *LLMAssistantAggregator, contextID string) {
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	_ = a.QueueFrame(stopped, frames.Downstream)
}

func ttsSpeakTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if s, ok := f.(*frames.TTSSpeakFrame); ok {
			out = append(out, s.Text)
		}
	}
	return out
}

func assistantTexts(c *llm.Context) []string {
	var out []string
	for _, m := range c.Messages() {
		if m.Role == llm.RoleAssistant {
			out = append(out, m.Text())
		}
	}
	return out
}

func developerCount(c *llm.Context) int {
	n := 0
	for _, m := range c.Messages() {
		if m.Role == llm.RoleDeveloper {
			n++
		}
	}
	return n
}

func botSpeaksResponse(t *testing.T, u *LLMUserAggregator, a *LLMAssistantAggregator, asstDown *captureProc, contextID string, sentences ...string) {
	t.Helper()
	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = a.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext(contextID), frames.Downstream)
	for _, s := range sentences {
		_ = a.QueueFrame(frames.NewTTSQueuedTextFrame(s, frames.TextTypeSentence, contextID), frames.Downstream)
	}
	dataBarrier(t, a, asstDown)
}

func interruptAssistant(t *testing.T, a *LLMAssistantAggregator, asstDown *captureProc) {
	t.Helper()
	dataBarrier(t, a, asstDown)
	_ = a.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	systemBarrier(t, a, asstDown)
}

func TestUserAggregator_EmptyTurnWithoutBotSpeech(t *testing.T) {
	t.Run("idle turn is left unanswered", func(t *testing.T) {
		c := llm.NewContext(llm.User("hi"))
		u, _, down, _ := startEmptyTurnPair(t, c, nil)
		_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
		_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
		userTurn(u, "")
		waitTurnEnd(t, down, 1)
		if n := countNamed(down.get(), "LLMContextFrame"); n != 0 || c.Len() != 1 {
			t.Fatalf("idle empty turn answered: %d inferences, %d messages", n, c.Len())
		}
	})

	t.Run("idle prompt answers once in a row", func(t *testing.T) {
		c := llm.NewContext(llm.User("hi"))
		u, _, down, _ := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{IdlePrompt: "ask again"})
		_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
		_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
		userTurn(u, "")
		waitTurnEnd(t, down, 1)
		userTurn(u, "")
		waitTurnEnd(t, down, 2)
		msgs := c.Messages()
		if n := countNamed(down.get(), "LLMContextFrame"); n != 1 || developerCount(c) != 1 || msgs[len(msgs)-1].Text() != "ask again" {
			t.Fatalf("expected one idle recovery, got %d inferences and %+v", n, msgs)
		}
	})

	t.Run("turn before the bot first spoke counts as interrupted", func(t *testing.T) {
		c := llm.NewContext(llm.User("hi"))
		u, _, down, _ := startEmptyTurnPair(t, c, nil)
		userTurn(u, "")
		down.waitFor(t, "LLMContextFrame", time.Second)
		msgs := c.Messages()
		if last := msgs[len(msgs)-1]; last.Role != llm.RoleDeveloper || last.Text() != turns.DefaultEmptyUserTurnInterruptedPrompt {
			t.Fatalf("expected the interrupted prompt, got %+v", msgs)
		}
	})

	t.Run("disabled", func(t *testing.T) {
		c := llm.NewContext(llm.User("hi"))
		u, _, down, _ := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{Disabled: true})
		userTurn(u, "")
		waitTurnEnd(t, down, 1)
		if n := countNamed(down.get(), "LLMContextFrame"); n != 0 || c.Len() != 1 {
			t.Fatalf("disabled recovery still answered: %d inferences", n)
		}
	})
}

func TestUserAggregator_EmptyTurnInterruptingBotRunsPromptByDefault(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, nil)
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.")
	speakWords(a, "c1", "Sure.", "The", "weather")
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	interruptAssistant(t, a, asstDown)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)

	if got := ttsSpeakTexts(down.get()); len(got) != 0 {
		t.Fatalf("resume is opt-in, got %q", got)
	}
	msgs := c.Messages()
	if len(msgs) != 3 || msgs[1].Text() != "Sure. The weather" || msgs[2].Role != llm.RoleDeveloper {
		t.Fatalf("unexpected context %+v", msgs)
	}
}

func TestUserAggregator_EmptyTurnResumesInterruptedSpeech(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.", "Anything else?")
	speakWords(a, "c1", "Sure.", "The", "weather")
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	interruptAssistant(t, a, asstDown)
	if got := assistantTexts(c); len(got) != 1 || got[0] != "Sure. The weather" {
		t.Fatalf("expected the heard part committed, got %q", got)
	}
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "TTSSpeakFrame", time.Second)

	if got := ttsSpeakTexts(down.get()); len(got) != 1 || got[0] != "The weather in Paris is sunny today. Anything else?" {
		t.Fatalf("expected resume from the interrupted sentence, got %q", got)
	}
	if countNamed(down.get(), "LLMContextFrame") != 0 || developerCount(c) != 0 {
		t.Fatal("resume must not run the LLM")
	}

	_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext("c2"), frames.Downstream)
	_ = a.QueueFrame(frames.NewTTSQueuedTextFrame("The weather in Paris is sunny today. Anything else?", frames.TextTypeSentence, "c2"), frames.Downstream)
	_ = a.QueueFrame(frames.NewTTSTextFrame("The weather in Paris is sunny today. Anything else?", frames.TextTypeSentence, "c2"), frames.Downstream)
	stopTTS(a, "c2")
	dataBarrier(t, a, asstDown)

	got := assistantTexts(c)
	if len(got) != 1 || got[0] != "Sure. The weather in Paris is sunny today. Anything else?" {
		t.Fatalf("expected the full assistant text once, got %q", got)
	}
}

func TestUserAggregator_TranscribedTurnNeverResumes(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.")
	speakWords(a, "c1", "Sure.", "The")
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	interruptAssistant(t, a, asstDown)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("stop", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)

	userTurn(u, "")
	waitTurnEnd(t, down, 2)
	if got := ttsSpeakTexts(down.get()); len(got) != 0 {
		t.Fatalf("a transcribed turn must not resume, got %q", got)
	}
	msgs := c.Messages()
	if msgs[1].Text() != "Sure. The" || msgs[2].Role != llm.RoleUser || msgs[2].Text() != "stop" {
		t.Fatalf("unexpected context %+v", msgs)
	}
}

func TestUserAggregator_ResumeSkippedWithFunctionCallInFlight(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Booking now.", "This takes a moment.")
	speakWords(a, "c1", "Booking")
	started := []frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "book"}}
	_ = u.QueueFrame(frames.NewFunctionCallsStartedFrame(started), frames.Downstream)
	_ = a.QueueFrame(frames.NewFunctionCallsStartedFrame(started), frames.Downstream)
	systemBarrier(t, a, asstDown)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "book", nil, false), frames.Downstream)

	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	interruptAssistant(t, a, asstDown)
	_ = u.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)

	if got := ttsSpeakTexts(down.get()); len(got) != 0 {
		t.Fatalf("resume must be skipped with a function call in flight, got %q", got)
	}
	if developerCount(c) != 1 {
		t.Fatal("expected the plain empty-turn prompt as fallback")
	}
}

func startResume(t *testing.T, cfg *turns.EmptyUserTurnConfig) (*llm.Context, *LLMUserAggregator, *LLMAssistantAggregator, *captureProc, *captureProc) {
	t.Helper()
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, cfg)
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.", "Anything else?")
	speakWords(a, "c1", "Sure.", "The", "weather")
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	interruptAssistant(t, a, asstDown)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "TTSSpeakFrame", time.Second)

	remainder := ttsSpeakTexts(down.get())[0]
	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext("c2"), frames.Downstream)
	_ = a.QueueFrame(frames.NewTTSQueuedTextFrame(remainder, frames.TextTypeSentence, "c2"), frames.Downstream)
	speakWords(a, "c2", "The", "weather", "in", "Paris", "is", "sunny", "today.", "Anything")
	dataBarrier(t, a, asstDown)
	return c, u, a, down, asstDown
}

func TestUserAggregator_SecondInterruptionDuringResume(t *testing.T) {
	t.Run("real turn", func(t *testing.T) {
		c, u, a, down, asstDown := startResume(t, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
		before := countNamed(down.get(), "InterruptionFrame")
		_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
		deadline := time.Now().Add(time.Second)
		for countNamed(down.get(), "InterruptionFrame") == before && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		interruptAssistant(t, a, asstDown)
		_ = u.QueueFrame(frames.NewTranscriptionFrame("wait what", true), frames.Downstream)
		_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
		down.waitFor(t, "LLMContextFrame", time.Second)

		if got := ttsSpeakTexts(down.get()); len(got) != 1 {
			t.Fatalf("expected no second resume, got %q", got)
		}
		msgs := c.Messages()
		if len(msgs) != 3 || msgs[1].Text() != "Sure. The weather in Paris is sunny today. Anything" || msgs[2].Text() != "wait what" {
			t.Fatalf("unexpected context %+v", msgs)
		}
	})

	t.Run("empty turn resumes again", func(t *testing.T) {
		c, u, a, down, asstDown := startResume(t, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true, MaxConsecutiveRecoveries: 2})
		before := countNamed(down.get(), "InterruptionFrame")
		_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
		deadline := time.Now().Add(time.Second)
		for countNamed(down.get(), "InterruptionFrame") == before && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		interruptAssistant(t, a, asstDown)
		_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
		deadline = time.Now().Add(time.Second)
		for len(ttsSpeakTexts(down.get())) < 2 && time.Now().Before(deadline) {
			time.Sleep(5 * time.Millisecond)
		}
		got := ttsSpeakTexts(down.get())
		if len(got) != 2 || got[1] != "Anything else?" {
			t.Fatalf("expected a second resume from the next sentence, got %q", got)
		}

		_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext("c3"), frames.Downstream)
		_ = a.QueueFrame(frames.NewTTSQueuedTextFrame("Anything else?", frames.TextTypeSentence, "c3"), frames.Downstream)
		speakWords(a, "c3", "Anything", "else?")
		stopTTS(a, "c3")
		dataBarrier(t, a, asstDown)
		if texts := assistantTexts(c); len(texts) != 1 || texts[0] != "Sure. The weather in Paris is sunny today. Anything else?" {
			t.Fatalf("expected the full assistant text once, got %q", texts)
		}
	})
}
