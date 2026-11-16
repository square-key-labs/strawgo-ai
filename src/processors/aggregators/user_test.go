package aggregators

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/strawtest"
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

func queueSpeech(a *LLMAssistantAggregator, contextID string, tracked, complete bool, texts ...string) {
	for _, text := range texts {
		q := frames.NewTTSQueuedTextFrame(text, frames.TextTypeSentence, contextID)
		q.PlaybackTracked = tracked
		_ = a.QueueFrame(q, frames.Downstream)
	}
	if complete {
		_ = a.QueueFrame(frames.NewTTSQueuedTextCompleteFrame(contextID), frames.Downstream)
	}
}

func botStartsResponse(t *testing.T, u *LLMUserAggregator, a *LLMAssistantAggregator, asstDown *captureProc, contextID string, tracked, complete bool, sentences ...string) {
	t.Helper()
	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = a.QueueFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)
	_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext(contextID), frames.Downstream)
	queueSpeech(a, contextID, tracked, complete, sentences...)
	dataBarrier(t, a, asstDown)
}

func botSpeaksResponse(t *testing.T, u *LLMUserAggregator, a *LLMAssistantAggregator, asstDown *captureProc, contextID string, sentences ...string) {
	t.Helper()
	botStartsResponse(t, u, a, asstDown, contextID, true, true, sentences...)
}

func interruptAssistant(t *testing.T, a *LLMAssistantAggregator, asstDown *captureProc) {
	t.Helper()
	dataBarrier(t, a, asstDown)
	_ = a.QueueFrame(frames.NewInterruptionFrame(), frames.Downstream)
	systemBarrier(t, a, asstDown)
}

func interruptBot(t *testing.T, u *LLMUserAggregator, a *LLMAssistantAggregator, down, asstDown *captureProc) {
	t.Helper()
	before := countNamed(down.get(), "InterruptionFrame")
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	deadline := time.Now().Add(time.Second)
	for countNamed(down.get(), "InterruptionFrame") == before {
		if time.Now().After(deadline) {
			t.Fatal("no interruption")
		}
		time.Sleep(2 * time.Millisecond)
	}
	interruptAssistant(t, a, asstDown)
	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
}

func waitSpeakCount(t *testing.T, down *captureProc, n int) []string {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for len(ttsSpeakTexts(down.get())) < n {
		if time.Now().After(deadline) {
			t.Fatalf("expected %d TTSSpeakFrames, got %q", n, ttsSpeakTexts(down.get()))
		}
		time.Sleep(2 * time.Millisecond)
	}
	return ttsSpeakTexts(down.get())
}

func emptyTurnAfterInterruption(t *testing.T, u *LLMUserAggregator, a *LLMAssistantAggregator, down, asstDown *captureProc) {
	t.Helper()
	interruptBot(t, u, a, down, asstDown)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
}

func playResume(t *testing.T, u *LLMUserAggregator, a *LLMAssistantAggregator, asstDown *captureProc, contextID, text string, words ...string) {
	t.Helper()
	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext(contextID), frames.Downstream)
	queueSpeech(a, contextID, true, true, text)
	speakWords(a, contextID, words...)
	dataBarrier(t, a, asstDown)
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
		_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
		_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
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

	t.Run("recovery follows the active LLM", func(t *testing.T) {
		c := llm.NewContext(llm.User("hi"))
		u, _, down, _ := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{IdlePrompt: "ask again"})
		idleEmptyTurn := func(n int) {
			_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
			_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
			userTurn(u, "")
			waitTurnEnd(t, down, n)
		}
		_ = u.QueueFrame(frames.NewLLMServiceMetadataFrame("live", true), frames.Upstream)
		idleEmptyTurn(1)
		if n := countNamed(down.get(), "LLMContextFrame"); n != 0 || c.Len() != 1 {
			t.Fatalf("recovery ran with a realtime service: %d inferences, %d messages", n, c.Len())
		}
		_ = u.QueueFrame(frames.NewLLMServiceMetadataFrame("cascaded", false), frames.Upstream)
		idleEmptyTurn(2)
		if n := countNamed(down.get(), "LLMContextFrame"); n != 1 || developerCount(c) != 1 {
			t.Fatalf("recovery not back on after a cascaded LLM announced itself: %d inferences", n)
		}
	})
}

func TestUserAggregator_EmptyTurnInterruptingBotRunsPromptByDefault(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, nil)
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.")
	speakWords(a, "c1", "Sure.", "The", "weather")
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	down.waitFor(t, "LLMContextFrame", time.Second)

	if got := ttsSpeakTexts(down.get()); len(got) != 0 {
		t.Fatalf("resume is opt-in, got %q", got)
	}
	msgs := c.Messages()
	if len(msgs) != 3 || msgs[1].Text() != "Sure. The weather" || msgs[2].Role != llm.RoleDeveloper {
		t.Fatalf("unexpected context %+v", msgs)
	}
}

func TestUserAggregator_RecoveryPromptRemovedByNextUserMessage(t *testing.T) {
	c := llm.NewContext(llm.User("hi"))
	u, _, down, _ := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{IdlePrompt: "ask again"})
	userTurn(u, "")
	down.waitFor(t, "LLMContextFrame", time.Second)
	if developerCount(c) != 1 {
		t.Fatal("expected the recovery prompt")
	}
	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	userTurn(u, "book a table")
	waitTurnEnd(t, down, 2)
	msgs := c.Messages()
	if developerCount(c) != 0 || len(msgs) != 2 || msgs[1].Text() != "book a table" {
		t.Fatalf("recovery prompt not removed: %+v", msgs)
	}

	_ = u.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	userTurn(u, "")
	waitTurnEnd(t, down, 3)
	if developerCount(c) != 1 {
		t.Fatal("expected the idle recovery prompt")
	}
	_ = u.QueueFrame(frames.NewEndFrame(), frames.Downstream)
	down.waitFor(t, "EndFrame", time.Second)
	if developerCount(c) != 0 {
		t.Fatal("recovery prompt kept at the end of the session")
	}
}

func TestUserAggregator_EmptyTurnResumesInterruptedSpeech(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.", "Anything else?")
	speakWords(a, "c1", "Sure.", "The", "weather")
	interruptBot(t, u, a, down, asstDown)
	if got := assistantTexts(c); len(got) != 1 || got[0] != "Sure. The weather" {
		t.Fatalf("expected the heard part committed, got %q", got)
	}
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)

	if got := waitSpeakCount(t, down, 1); got[0] != "The weather in Paris is sunny today. Anything else?" {
		t.Fatalf("expected resume from the interrupted sentence, got %q", got)
	}
	if countNamed(down.get(), "LLMContextFrame") != 0 || developerCount(c) != 0 {
		t.Fatal("resume must not run the LLM")
	}

	_ = a.QueueFrame(frames.NewTTSStartedFrameWithContext("c2"), frames.Downstream)
	queueSpeech(a, "c2", true, true, "The weather in Paris is sunny today. Anything else?")
	_ = a.QueueFrame(frames.NewTTSTextFrame("The weather in Paris is sunny today. Anything else?", frames.TextTypeSentence, "c2"), frames.Downstream)
	stopTTS(a, "c2")
	dataBarrier(t, a, asstDown)

	got := assistantTexts(c)
	if len(got) != 1 || got[0] != "Sure. The weather in Paris is sunny today. Anything else?" {
		t.Fatalf("expected the full assistant text once, got %q", got)
	}
}

func TestUserAggregator_ResumeFromSentenceTrackedTTS(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.", "Anything else?")
	_ = a.QueueFrame(frames.NewTTSTextFrame("Sure.", frames.TextTypeSentence, "c1"), frames.Downstream)
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	if got := waitSpeakCount(t, down, 1); got[0] != "The weather in Paris is sunny today. Anything else?" {
		t.Fatalf("expected resume from the unheard sentence, got %q", got)
	}
}

func TestUserAggregator_ResumeKeepsAbbreviationsInSentence(t *testing.T) {
	c := llm.NewContext(llm.User("who?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "I booked Dr. Smith for you. See you soon.")
	speakWords(a, "c1", "I", "booked", "Dr.")
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	if got := waitSpeakCount(t, down, 1); got[0] != "I booked Dr. Smith for you. See you soon." {
		t.Fatalf("expected resume from the sentence start, got %q", got)
	}
}

func TestUserAggregator_ResumeRefusedWithoutPlaybackTracking(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botStartsResponse(t, u, a, asstDown, "c1", false, true, "Sure.", "The weather in Paris is sunny today.", "Anything else?")
	for _, s := range []string{"Sure.", "The weather in Paris is sunny today.", "Anything else?"} {
		_ = a.QueueFrame(frames.NewTTSTextFrame(s, frames.TextTypeSentence, "c1"), frames.Downstream)
	}
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	down.waitFor(t, "LLMContextFrame", time.Second)
	if got := ttsSpeakTexts(down.get()); len(got) != 0 || developerCount(c) != 1 {
		t.Fatalf("expected the prompt fallback, got resume %q", got)
	}
}

func TestUserAggregator_ResumeRefusedBeforeResponseEnded(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botStartsResponse(t, u, a, asstDown, "c1", true, false, "Let me book that.")
	speakWords(a, "c1", "Let", "me")
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	down.waitFor(t, "LLMContextFrame", time.Second)
	if got := ttsSpeakTexts(down.get()); len(got) != 0 || developerCount(c) != 1 {
		t.Fatalf("expected the prompt fallback for an unfinished response, got resume %q", got)
	}
}

func TestUserAggregator_TranscribedTurnNeverResumes(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.")
	speakWords(a, "c1", "Sure.", "The")
	interruptBot(t, u, a, down, asstDown)
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

func TestUserAggregator_FunctionResultDuringEmptyTurnRunsLLMOnce(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Booking now.", "This takes a moment.")
	speakWords(a, "c1", "Booking")
	started := []frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "book"}}
	_ = u.QueueFrame(frames.NewFunctionCallsStartedFrame(started), frames.Upstream)
	_ = a.QueueFrame(frames.NewFunctionCallsStartedFrame(started), frames.Downstream)
	systemBarrier(t, a, asstDown)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "book", nil, false), frames.Downstream)

	interruptBot(t, u, a, down, asstDown)
	_ = u.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Upstream)
	_ = u.QueueFrame(frames.NewLLMContextFrame(c), frames.Upstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	waitTurnEnd(t, down, 1)

	if got := ttsSpeakTexts(down.get()); len(got) != 0 {
		t.Fatalf("resume must be skipped with a function call in flight, got %q", got)
	}
	if n := countNamed(down.get(), "LLMContextFrame"); n != 1 || developerCount(c) != 0 {
		t.Fatalf("expected only the function result's inference, got %d runs and %d prompts", n, developerCount(c))
	}
}

func TestUserAggregator_CompletedResumeResetsRecoveryCount(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "It is sunny.")
	speakWords(a, "c1", "Sure.")
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	remainder := waitSpeakCount(t, down, 1)[0]
	playResume(t, u, a, asstDown, "c2", remainder, "It", "is", "sunny.")
	stopTTS(a, "c2")
	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	dataBarrier(t, a, asstDown)

	botSpeaksResponse(t, u, a, asstDown, "c3", "Anything else?", "I can help.")
	speakWords(a, "c3", "Anything")
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	if got := waitSpeakCount(t, down, 2); got[1] != "Anything else? I can help." {
		t.Fatalf("expected a second resume after a completed one, got %q", got)
	}
}

func startResume(t *testing.T, cfg *turns.EmptyUserTurnConfig) (*llm.Context, *LLMUserAggregator, *LLMAssistantAggregator, *captureProc, *captureProc) {
	t.Helper()
	c := llm.NewContext(llm.User("weather?"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, cfg)
	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.", "Anything else?")
	speakWords(a, "c1", "Sure.", "The", "weather")
	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	remainder := waitSpeakCount(t, down, 1)[0]
	playResume(t, u, a, asstDown, "c2", remainder, "The", "weather", "in", "Paris", "is", "sunny", "today.", "Anything")
	return c, u, a, down, asstDown
}

func TestUserAggregator_SecondInterruptionDuringResume(t *testing.T) {
	t.Run("real turn", func(t *testing.T) {
		c, u, a, down, asstDown := startResume(t, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
		interruptBot(t, u, a, down, asstDown)
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
		emptyTurnAfterInterruption(t, u, a, down, asstDown)
		if got := waitSpeakCount(t, down, 2); got[1] != "Anything else?" {
			t.Fatalf("expected a second resume from the next sentence, got %q", got)
		}

		playResume(t, u, a, asstDown, "c3", "Anything else?", "Anything", "else?")
		stopTTS(a, "c3")
		dataBarrier(t, a, asstDown)
		if texts := assistantTexts(c); len(texts) != 1 || texts[0] != "Sure. The weather in Paris is sunny today. Anything else?" {
			t.Fatalf("expected the full assistant text once, got %q", texts)
		}
	})
}

func TestUserAggregator_ResumeSkippedWhenFunctionResultAwaitsAnswer(t *testing.T) {
	c := llm.NewContext(llm.User("book it"))
	u, a, down, asstDown := startEmptyTurnPair(t, c, &turns.EmptyUserTurnConfig{ResumeInterruptedSpeech: true})
	started := []frames.FunctionCallInfo{{ToolCallID: "1", FunctionName: "book"}}
	_ = u.QueueFrame(frames.NewFunctionCallsStartedFrame(started), frames.Upstream)
	_ = a.QueueFrame(frames.NewFunctionCallsStartedFrame(started), frames.Downstream)
	systemBarrier(t, a, asstDown)
	_ = a.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "book", nil, false), frames.Downstream)
	botSpeaksResponse(t, u, a, asstDown, "c1", "Let me book that.", "One moment please.")
	_ = a.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Downstream)
	systemBarrier(t, a, asstDown)
	speakWords(a, "c1", "Let", "me")
	_ = u.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	_ = a.QueueFrame(frames.NewFunctionCallResultFrame("1", "book", "booked", nil), frames.Downstream)
	dataBarrier(t, a, asstDown)
	down.waitFor(t, "FunctionCallResultFrame", time.Second)
	if a.HasFunctionCallsInProgress() {
		t.Fatal("call should be settled")
	}

	emptyTurnAfterInterruption(t, u, a, down, asstDown)
	down.waitFor(t, "LLMContextFrame", time.Second)
	if got := ttsSpeakTexts(down.get()); len(got) != 0 || developerCount(c) != 1 {
		t.Fatalf("expected the prompt path to answer the function result, got resume %q", got)
	}
}

type proposedStart struct{}

func (proposedStart) ShouldStart(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStartedSpeakingFrame)
	return ok
}
func (proposedStart) EnableInterruptions() bool { return true }
func (proposedStart) Reset()                    {}

type proposedStop struct{}

func (proposedStop) ShouldStop(frame any) bool {
	_, ok := frame.(*frames.ProposedUserStoppedSpeakingFrame)
	return ok
}
func (proposedStop) Reset() {}

type muteAll struct{}

func (muteAll) ShouldMute(any) bool { return true }
func (muteAll) Reset()              {}

type turnLLM struct {
	mu      sync.Mutex
	replies []func(snap llm.Snapshot, emit func(llm.Event))
	seen    []llm.Snapshot
}

func (p *turnLLM) Name() string { return "turnLLM" }

func (p *turnLLM) Stream(ctx context.Context, snap llm.Snapshot, _ llm.Settings, emit func(llm.Event)) error {
	p.mu.Lock()
	n := len(p.seen)
	p.seen = append(p.seen, snap)
	p.mu.Unlock()
	if n < len(p.replies) {
		p.replies[n](snap, emit)
	}
	return nil
}

func (p *turnLLM) requests() []llm.Snapshot {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]llm.Snapshot(nil), p.seen...)
}

func say(text string) func(llm.Snapshot, func(llm.Event)) {
	return func(_ llm.Snapshot, emit func(llm.Event)) { emit(llm.TextDelta{Text: text}) }
}

func callTool(id, name string) func(llm.Snapshot, func(llm.Event)) {
	return func(_ llm.Snapshot, emit func(llm.Event)) {
		emit(llm.ToolCall{ID: id, Name: name, Arguments: []byte(`{}`)})
	}
}

func eagerStrategies() turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{proposedStart{}},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewEagerUserTurnStopStrategy(nil, 0)},
	}
}

func runSpeculation(t *testing.T, c *llm.Context, strategies turns.UserTurnStrategies, provider *turnLLM, setup func(*llmservice.Service), send ...frames.Frame) []frames.Frame {
	t.Helper()
	u := NewLLMUserAggregatorWithParams(c, UserAggregatorParams{Strategies: strategies})
	svc := llmservice.New(provider)
	if setup != nil {
		setup(svc)
	}
	down, _ := strawtest.RunProcessors(t, []processors.FrameProcessor{u, svc}, send)
	return down
}

func llmTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if t, ok := f.(*frames.LLMTextFrame); ok {
			out = append(out, t.Text)
		}
	}
	return out
}

func lastUserText(snap llm.Snapshot) string {
	for i := len(snap.Messages) - 1; i >= 0; i-- {
		if snap.Messages[i].Role == llm.RoleUser {
			return snap.Messages[i].Text()
		}
	}
	return ""
}

func eagerThenFinal(eager string, between []frames.Frame, final string) []frames.Frame {
	send := []frames.Frame{
		frames.NewProposedUserStartedSpeakingFrame(),
		strawtest.Sleep(20 * time.Millisecond),
		frames.NewEagerTranscriptionFrame(eager, ""),
		strawtest.Sleep(80 * time.Millisecond),
	}
	send = append(send, between...)
	if final != "" {
		send = append(send, frames.NewTranscriptionFrame(final, true))
	}
	return append(send, frames.NewProposedUserStoppedSpeakingFrame(), strawtest.Sleep(80*time.Millisecond))
}

func TestSpeculation_ConfirmedReleasesHeldReplyWithoutRerun(t *testing.T) {
	c := llm.NewContext()
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("For two, done.")}}
	down := runSpeculation(t, c, eagerStrategies(), provider, nil, eagerThenFinal("book a table for two", nil, "Book a table for two.")...)

	reqs := provider.requests()
	if len(reqs) != 1 || lastUserText(reqs[0]) != "book a table for two" {
		t.Fatalf("want one speculative request, got %d", len(reqs))
	}
	if got := llmTexts(down); len(got) != 1 || got[0] != "For two, done." {
		t.Fatalf("texts = %v", got)
	}
	ci := slicesIndex(down, is[*frames.UserTurnSpeculationConfirmedFrame])
	si := slicesIndex(down, is[*frames.LLMFullResponseStartFrame])
	if ci < 0 || si < ci {
		t.Fatalf("reply released before confirmation: %s", strawtest.FrameTypes(down))
	}
	msgs := c.Messages()
	if len(msgs) != 1 || msgs[0].Text() != "Book a table for two." {
		t.Fatalf("context = %+v", msgs)
	}
}

func TestSpeculation_ChangedFinalTextDiscardsAndReruns(t *testing.T) {
	c := llm.NewContext()
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Speculative."), say("Committed.")}}
	down := runSpeculation(t, c, eagerStrategies(), provider, nil, eagerThenFinal("book a table", nil, "Book a table for six.")...)

	reqs := provider.requests()
	if len(reqs) != 2 || lastUserText(reqs[1]) != "Book a table for six." {
		t.Fatalf("want a rerun on the committed text, got %d requests", len(reqs))
	}
	if got := llmTexts(down); len(got) != 1 || got[0] != "Committed." {
		t.Fatalf("texts = %v", got)
	}
	if slicesIndex(down, is[*frames.UserTurnSpeculationConfirmedFrame]) >= 0 {
		t.Fatal("a changed transcript must not confirm")
	}
}

func TestSpeculation_UserResumesDiscards(t *testing.T) {
	cases := map[string][]frames.Frame{
		"turn resumed":           {frames.NewUserTurnResponseCancelFrame()},
		"user started speaking":  {frames.NewUserStartedSpeakingFrame(), frames.NewUserStoppedSpeakingFrame()},
		"interruption mid-hold":  {frames.NewInterruptionFrame()},
		"proposed start resumed": {frames.NewProposedUserStartedSpeakingFrame()},
	}
	for name, between := range cases {
		t.Run(name, func(t *testing.T) {
			provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Speculative."), say("Committed.")}}
			between = append(between, strawtest.Sleep(30*time.Millisecond))
			down := runSpeculation(t, llm.NewContext(), eagerStrategies(), provider, nil, eagerThenFinal("book a table", between, "book a table")...)
			if n := len(provider.requests()); n != 2 {
				t.Fatalf("want the committed turn answered by a fresh run, got %d requests", n)
			}
			if got := llmTexts(down); len(got) != 1 || got[0] != "Committed." {
				t.Fatalf("texts = %v", got)
			}
		})
	}
}

func TestSpeculation_ToolCallRunsOnlyAfterConfirmation(t *testing.T) {
	c := llm.NewContext()
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){callTool("c1", "book"), callTool("c2", "book")}}
	var mu sync.Mutex
	var userTurns []int
	down := runSpeculation(t, c, eagerStrategies(), provider, func(svc *llmservice.Service) {
		svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
			mu.Lock()
			userTurns = append(userTurns, c.Len())
			mu.Unlock()
			return llm.FunctionResult{Value: "booked"}, nil
		})
	}, eagerThenFinal("book a table", nil, "Book a table.")...)

	mu.Lock()
	defer mu.Unlock()
	if len(userTurns) != 1 || userTurns[0] != 1 {
		t.Fatalf("tool ran %d times (context sizes %v); want once, after the turn was written", len(userTurns), userTurns)
	}
	results := 0
	for _, f := range down {
		if r, ok := f.(*frames.FunctionCallResultFrame); ok {
			results++
			if r.ToolCallID != "c2" {
				t.Fatalf("speculative call %s produced a result", r.ToolCallID)
			}
		}
	}
	if results != 1 {
		t.Fatalf("results = %d: %s", results, strawtest.FrameTypes(down))
	}
}

func TestSpeculation_ToolCallDroppedOnDiscard(t *testing.T) {
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){callTool("c1", "book"), say("Which day?")}}
	var mu sync.Mutex
	ran := 0
	down := runSpeculation(t, llm.NewContext(), eagerStrategies(), provider, func(svc *llmservice.Service) {
		svc.RegisterFunction("book", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
			mu.Lock()
			ran++
			mu.Unlock()
			return llm.FunctionResult{Value: "booked"}, nil
		})
	}, eagerThenFinal("book a table", nil, "Book a table, but not yet.")...)
	mu.Lock()
	defer mu.Unlock()
	if ran != 0 {
		t.Fatalf("discarded speculative tool call ran %d times", ran)
	}
	if got := llmTexts(down); len(got) != 1 || got[0] != "Which day?" {
		t.Fatalf("texts = %v", got)
	}
}

func TestSpeculation_EmptyTurnStillRecovers(t *testing.T) {
	c := llm.NewContext()
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Speculative."), say("Sorry, could you repeat that?")}}
	down := runSpeculation(t, c, eagerStrategies(), provider, nil, eagerThenFinal("uh", nil, "")...)
	reqs := provider.requests()
	if len(reqs) != 2 {
		t.Fatalf("want the speculation and the recovery, got %d requests", len(reqs))
	}
	if last := reqs[1].Messages[len(reqs[1].Messages)-1]; last.Role != llm.RoleDeveloper {
		t.Fatalf("recovery request ends with %s", last.Role)
	}
	if got := llmTexts(down); len(got) != 1 || got[0] != "Sorry, could you repeat that?" {
		t.Fatalf("texts = %v", got)
	}
}

func TestSpeculation_MutedUserDoesNotSpeculate(t *testing.T) {
	strategies := eagerStrategies()
	strategies.MuteStrategies = []user_mute.UserMuteStrategy{muteAll{}}
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Speculative.")}}
	runSpeculation(t, llm.NewContext(), strategies, provider, nil, eagerThenFinal("book a table", nil, "book a table")...)
	if n := len(provider.requests()); n != 0 {
		t.Fatalf("muted user triggered %d inferences", n)
	}
}

func TestSpeculation_OffBehavesAsBefore(t *testing.T) {
	c := llm.NewContext()
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{proposedStart{}},
		StopStrategies:  []user_stop.UserTurnStopStrategy{proposedStop{}},
	}
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Done.")}}
	down := runSpeculation(t, c, strategies, provider, nil, eagerThenFinal("book a table", nil, "Book a table.")...)
	reqs := provider.requests()
	if len(reqs) != 1 || lastUserText(reqs[0]) != "Book a table." {
		t.Fatalf("want one inference on the committed text, got %d", len(reqs))
	}
	for _, f := range down {
		switch v := f.(type) {
		case *frames.UserTurnSpeculationConfirmedFrame, *frames.UserTurnResponseCancelFrame:
			t.Fatalf("unexpected %s with speculation off", f.Name())
		case *frames.LLMContextUserTurnFrame:
			if v.Speculation {
				t.Fatal("speculative user turn broadcast with speculation off")
			}
		}
	}
	if got := llmTexts(down); len(got) != 1 || got[0] != "Done." {
		t.Fatalf("texts = %v", got)
	}
}

func TestSpeculation_ContextChangeDuringHoldReruns(t *testing.T) {
	c := llm.NewContext()
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Speculative."), say("Committed.")}}
	between := []frames.Frame{frames.NewLLMMessagesAppendFrame([]llm.Message{llm.System("the table is booked")}, false), strawtest.Sleep(30 * time.Millisecond)}
	down := runSpeculation(t, c, eagerStrategies(), provider, nil, eagerThenFinal("book a table", between, "book a table")...)
	reqs := provider.requests()
	if len(reqs) != 2 {
		t.Fatalf("want the committed turn rerun after the context changed, got %d requests", len(reqs))
	}
	found := false
	for _, m := range reqs[1].Messages {
		if m.Text() == "the table is booked" {
			found = true
		}
	}
	if !found {
		t.Fatal("the rerun did not see the context change")
	}
	if got := llmTexts(down); len(got) != 1 || got[0] != "Committed." {
		t.Fatalf("texts = %v", got)
	}
}

func TestSpeculation_InferenceDuringHoldStillAnswersTurn(t *testing.T) {
	c := llm.NewContext()
	provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Speculative."), say("Run."), say("Committed.")}}
	between := []frames.Frame{frames.NewLLMRunFrame(), strawtest.Sleep(50 * time.Millisecond)}
	down := runSpeculation(t, c, eagerStrategies(), provider, nil, eagerThenFinal("book a table", between, "book a table")...)
	if n := len(provider.requests()); n != 3 {
		t.Fatalf("want speculation, run and committed answer, got %d requests", n)
	}
	got := llmTexts(down)
	if len(got) != 2 || got[0] != "Run." || got[1] != "Committed." {
		t.Fatalf("texts = %v", got)
	}
}

func TestSpeculation_EagerBeforeTurnStartStillSpeculates(t *testing.T) {
	for i := 0; i < 5; i++ {
		c := llm.NewContext()
		provider := &turnLLM{replies: []func(llm.Snapshot, func(llm.Event)){say("Done.")}}
		down := runSpeculation(t, c, eagerStrategies(), provider, nil,
			frames.NewProposedUserStartedSpeakingFrame(),
			frames.NewEagerTranscriptionFrame("book a table", ""),
			strawtest.Sleep(60*time.Millisecond),
			frames.NewTranscriptionFrame("book a table", true),
			frames.NewProposedUserStoppedSpeakingFrame(),
			strawtest.Sleep(60*time.Millisecond),
		)
		reqs := provider.requests()
		if len(reqs) != 1 || lastUserText(reqs[0]) != "book a table" || slicesIndex(down, is[*frames.UserTurnSpeculationConfirmedFrame]) < 0 {
			t.Fatalf("iteration %d: want one confirmed speculative request, got %d", i, len(reqs))
		}
	}
}

func TestSpeculation_WithdrawalAfterConfirmationAnswersCommittedTurn(t *testing.T) {
	c := llm.NewContext()
	u, down, _ := startUserAggregator(t, c, UserAggregatorParams{Strategies: eagerStrategies()}, true)
	_ = u.QueueFrame(frames.NewProposedUserStartedSpeakingFrame(), frames.Downstream)
	time.Sleep(20 * time.Millisecond)
	_ = u.QueueFrame(frames.NewEagerTranscriptionFrame("book a table", ""), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("Book a table.", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "UserTurnSpeculationConfirmedFrame", time.Second)

	stale := frames.NewUserTurnResponseCancelFrame()
	stale.Context = llm.NewContext()
	_ = u.QueueFrame(stale, frames.Upstream)
	withdrawn := frames.NewUserTurnResponseCancelFrame()
	for _, f := range down.get() {
		if cf, ok := f.(*frames.UserTurnSpeculationConfirmedFrame); ok {
			withdrawn.Context = cf.Context
		}
	}
	time.Sleep(30 * time.Millisecond)
	for _, f := range down.get() {
		if cf, ok := f.(*frames.LLMContextFrame); ok && !cf.Speculation {
			t.Fatal("a cancel for another speculation must not rerun the turn")
		}
	}
	_ = u.QueueFrame(withdrawn, frames.Upstream)
	deadline := time.Now().Add(time.Second)
	for {
		var committed *frames.LLMContextFrame
		for _, f := range down.get() {
			if cf, ok := f.(*frames.LLMContextFrame); ok && !cf.Speculation {
				committed = cf
			}
		}
		if committed != nil {
			if committed.Context != c {
				t.Fatal("the rerun must answer the committed context")
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("a speculation withdrawn after confirmation left the turn unanswered")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func is[T frames.Frame](f frames.Frame) bool {
	_, ok := f.(T)
	return ok
}

func slicesIndex(fs []frames.Frame, match func(frames.Frame) bool) int {
	for i, f := range fs {
		if match(f) {
			return i
		}
	}
	return -1
}

func sendAndSee(t *testing.T, u *LLMUserAggregator, f frames.Frame, dir frames.FrameDirection, seen *captureProc) {
	t.Helper()
	before := countNamed(seen.get(), f.Name())
	_ = u.QueueFrame(f, dir)
	deadline := time.Now().Add(time.Second)
	for countNamed(seen.get(), f.Name()) == before {
		if time.Now().After(deadline) {
			t.Fatalf("%s not processed", f.Name())
		}
		time.Sleep(time.Millisecond)
	}
}

func backchannelStrategies() turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewBackchannelUserTurnStartStrategy(user_start.BackchannelParams{}, true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewSpeechTimeoutUserTurnStopStrategy(20 * time.Millisecond)},
	}
}

func backchannelTexts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if b, ok := f.(*frames.UserBackchannelFrame); ok {
			out = append(out, b.Text)
		}
	}
	return out
}

func userMessages(c *llm.Context) []string {
	var out []string
	for _, m := range c.Messages() {
		if m.Role == llm.RoleUser {
			out = append(out, m.Text())
		}
	}
	return out
}

func TestUserAggregator_BackchannelKeepsBotTalking(t *testing.T) {
	c := llm.NewContext()
	var mu sync.Mutex
	var heard []string
	u, down, up := startUserAggregator(t, c, UserAggregatorParams{
		Strategies:        backchannelStrategies(),
		OnUserBackchannel: func(text string) { mu.Lock(); heard = append(heard, text); mu.Unlock() },
	}, true)

	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	sendAndSee(t, u, frames.NewUserStartedSpeakingFrame(), frames.Downstream, down)
	_ = u.QueueFrame(frames.NewInterimTranscriptionFrame("haan", ""), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("Haan ji.", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "UserBackchannelFrame", time.Second)
	time.Sleep(60 * time.Millisecond)

	got := down.get()
	if n := countNamed(got, "InterruptionFrame") + countNamed(up.get(), "InterruptionFrame"); n != 0 {
		t.Fatalf("backchannel interrupted the bot %d times", n)
	}
	if countNamed(got, "LLMContextFrame") != 0 || c.Len() != 0 {
		t.Fatalf("backchannel reached the LLM: %+v", c.Messages())
	}
	mu.Lock()
	if b := backchannelTexts(got); len(b) != 1 || b[0] != "Haan ji." || len(heard) != 1 || heard[0] != "Haan ji." {
		t.Fatalf("backchannel frames %q, callback %q", b, heard)
	}
	mu.Unlock()

	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	userTurn(u, "book a table")
	down.waitFor(t, "LLMContextFrame", time.Second)
	if msgs := userMessages(c); len(msgs) != 1 || msgs[0] != "book a table" {
		t.Fatalf("backchannel leaked into the next turn: %q", msgs)
	}
}

func TestUserAggregator_BackchannelMixedUtteranceInterrupts(t *testing.T) {
	c := llm.NewContext()
	u, down, up := startUserAggregator(t, c, UserAggregatorParams{Strategies: backchannelStrategies()}, true)

	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	sendAndSee(t, u, frames.NewUserStartedSpeakingFrame(), frames.Downstream, down)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("okay but what about tomorrow", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	down.waitFor(t, "LLMContextFrame", time.Second)
	if b := backchannelTexts(down.get()); len(b) != 0 {
		t.Fatalf("mixed utterance reported as a backchannel: %q", b)
	}
	if msgs := userMessages(c); len(msgs) != 1 || msgs[0] != "okay but what about tomorrow" {
		t.Fatalf("context = %q", msgs)
	}
}

func TestUserAggregator_DefaultStrategiesStillInterruptOnBackchannel(t *testing.T) {
	c := llm.NewContext()
	u, down, up := startUserAggregator(t, c, UserAggregatorParams{UserTurnStopTimeout: 50 * time.Millisecond}, true)

	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "InterruptionFrame", time.Second)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("okay", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", 2*time.Second)
	if b := backchannelTexts(down.get()); len(b) != 0 {
		t.Fatalf("default strategies reported a backchannel: %q", b)
	}
	if msgs := userMessages(c); len(msgs) != 1 || msgs[0] != "okay" {
		t.Fatalf("context = %q", msgs)
	}
}

func TestUserAggregator_BackchannelIsNotAnEmptyTurn(t *testing.T) {
	c := llm.NewContext(llm.User("weather?"))
	pair := NewContextAggregatorPair(c, ContextAggregatorPairParams{User: UserAggregatorParams{
		Strategies:          backchannelStrategies(),
		UserTurnStopTimeout: 30 * time.Millisecond,
		EmptyUserTurn:       &turns.EmptyUserTurnConfig{IdlePrompt: "ask again", MaxConsecutiveRecoveries: 3},
	}})
	u, a := pair.User(), pair.Assistant()
	down, up, asstDown := &captureProc{}, &captureProc{}, &captureProc{}
	u.Link(down)
	u.SetPrev(up)
	a.Link(asstDown)
	a.SetPrev(&captureProc{})
	for _, p := range []interface{ Start(context.Context) error }{u, a} {
		if err := p.Start(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { _ = u.Stop(); _ = a.Stop() })
	_ = u.QueueFrame(frames.NewStartFrameWithConfig(true, turns.UserTurnStrategies{}), frames.Downstream)

	botSpeaksResponse(t, u, a, asstDown, "c1", "Sure.", "The weather in Paris is sunny today.")
	speakWords(a, "c1", "Sure.", "The", "weather")
	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	sendAndSee(t, u, frames.NewUserStartedSpeakingFrame(), frames.Downstream, down)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("mm-hmm", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "UserBackchannelFrame", time.Second)
	time.Sleep(80 * time.Millisecond)
	_ = u.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	time.Sleep(80 * time.Millisecond)

	got := down.get()
	if n := countNamed(got, "InterruptionFrame"); n != 0 {
		t.Fatalf("backchannel interrupted the bot: %s", strawtest.FrameTypes(got))
	}
	if n := countNamed(got, "LLMContextFrame"); n != 0 || developerCount(c) != 0 || len(ttsSpeakTexts(got)) != 0 {
		t.Fatalf("backchannel treated as an empty turn: %d inferences, %+v", n, c.Messages())
	}
}

func TestUserAggregator_BackchannelDropsEagerSpeculation(t *testing.T) {
	c := llm.NewContext()
	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{user_start.NewBackchannelUserTurnStartStrategy(user_start.BackchannelParams{}, true)},
		StopStrategies:  []user_stop.UserTurnStopStrategy{user_stop.NewEagerUserTurnStopStrategy(nil, 0)},
	}
	u, down, up := startUserAggregator(t, c, UserAggregatorParams{Strategies: strategies}, true)

	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	sendAndSee(t, u, frames.NewUserStartedSpeakingFrame(), frames.Downstream, down)
	_ = u.QueueFrame(frames.NewEagerTranscriptionFrame("okay", ""), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("okay", true), frames.Downstream)
	down.waitFor(t, "UserBackchannelFrame", time.Second)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("stop please", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewProposedUserStoppedSpeakingFrame(), frames.Downstream)
	down.waitFor(t, "LLMContextFrame", time.Second)

	got := down.get()
	for _, f := range got {
		if cf, ok := f.(*frames.LLMContextFrame); ok && cf.Speculation {
			t.Fatal("the suppressed backchannel was answered speculatively")
		}
	}
	if b := backchannelTexts(got); len(b) != 1 || b[0] != "okay" {
		t.Fatalf("backchannel frames %q", b)
	}
	if countNamed(got, "InterruptionFrame") != 1 {
		t.Fatal("the repeated utterance did not interrupt")
	}
	if msgs := userMessages(c); len(msgs) != 1 || msgs[0] != "stop please" {
		t.Fatalf("context = %q", msgs)
	}
}

type toggleMute struct {
	mu sync.Mutex
	on bool
}

func (m *toggleMute) set(on bool) { m.mu.Lock(); m.on = on; m.mu.Unlock() }
func (m *toggleMute) ShouldMute(any) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.on
}
func (m *toggleMute) Reset() {}

func TestUserAggregator_MutedBackchannelIsNotReported(t *testing.T) {
	c := llm.NewContext()
	mute := &toggleMute{on: true}
	strategies := backchannelStrategies()
	strategies.MuteStrategies = []user_mute.UserMuteStrategy{mute}
	calls := 0
	var mu sync.Mutex
	u, down, up := startUserAggregator(t, c, UserAggregatorParams{
		Strategies:        strategies,
		OnUserBackchannel: func(string) { mu.Lock(); calls++; mu.Unlock() },
	}, true)

	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	down.waitFor(t, "UserMuteStartedFrame", time.Second)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("okay", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserBackchannelFrame("mhm", ""), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	time.Sleep(50 * time.Millisecond)
	if b := backchannelTexts(down.get()); len(b) != 0 {
		t.Fatalf("muted user reported backchannels %q", b)
	}

	mute.set(false)
	sendAndSee(t, u, frames.NewUserStartedSpeakingFrame(), frames.Downstream, down)
	down.waitFor(t, "UserMuteStoppedFrame", time.Second)
	_ = u.QueueFrame(frames.NewTranscriptionFrame("achha", true), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserBackchannelFrame("mhm", ""), frames.Downstream)
	_ = u.QueueFrame(frames.NewUserStoppedSpeakingFrame(), frames.Downstream)
	time.Sleep(50 * time.Millisecond)
	mu.Lock()
	defer mu.Unlock()
	if b := backchannelTexts(down.get()); len(b) != 2 || b[0] != "achha" || b[1] != "mhm" || calls != 2 {
		t.Fatalf("after unmute: frames %q, %d callbacks", b, calls)
	}
	if countNamed(down.get(), "InterruptionFrame") != 0 || c.Len() != 0 {
		t.Fatal("backchannel after unmute interrupted or reached the context")
	}
}

func TestUserAggregator_LongSpeechOverBotInterrupts(t *testing.T) {
	strategies := backchannelStrategies()
	strategies.StartStrategies = []user_start.UserTurnStartStrategy{
		user_start.NewBackchannelUserTurnStartStrategy(user_start.BackchannelParams{MaxSpeechDuration: 40 * time.Millisecond}, true),
	}
	u, down, up := startUserAggregator(t, llm.NewContext(), UserAggregatorParams{Strategies: strategies}, true)

	sendAndSee(t, u, frames.NewBotStartedSpeakingFrame(), frames.Upstream, up)
	_ = u.QueueFrame(frames.NewUserStartedSpeakingFrame(), frames.Downstream)
	time.Sleep(20 * time.Millisecond)
	if countNamed(down.get(), "InterruptionFrame") != 0 {
		t.Fatal("interrupted before the transcript or the speech limit")
	}
	down.waitFor(t, "InterruptionFrame", time.Second)
}
