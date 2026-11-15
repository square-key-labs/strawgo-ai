package vad

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const scriptedChunkSamples = 160

type scriptedAnalyzer struct {
	mu     sync.Mutex
	params VADParams
}

func (a *scriptedAnalyzer) SetSampleRate(int) error { return nil }
func (a *scriptedAnalyzer) NumFramesRequired() int  { return scriptedChunkSamples }
func (a *scriptedAnalyzer) VoiceConfidence([]byte) float32 {
	return 0
}
func (a *scriptedAnalyzer) AnalyzeAudio(buffer []byte) (VADState, error) {
	return VADState(buffer[0]), nil
}
func (a *scriptedAnalyzer) Restart() {}
func (a *scriptedAnalyzer) SetParams(params VADParams) {
	a.mu.Lock()
	a.params = params
	a.mu.Unlock()
}
func (a *scriptedAnalyzer) GetParams() VADParams {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.params
}

type blockingTurnAnalyzer struct {
	mu        sync.Mutex
	speech    []bool
	triggered bool
	entered   chan struct{}
	release   chan struct{}
}

func (a *blockingTurnAnalyzer) AppendAudio(_ []byte, isSpeech bool) turn.EndOfTurnState {
	a.mu.Lock()
	a.speech = append(a.speech, isSpeech)
	if isSpeech {
		a.triggered = true
	}
	a.mu.Unlock()
	return turn.TurnIncomplete
}

func (a *blockingTurnAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, *turn.TurnMetrics, error) {
	a.entered <- struct{}{}
	<-a.release
	return turn.TurnComplete, &turn.TurnMetrics{IsComplete: true, Probability: 0.9, TotalTimeMs: 12}, nil
}

func (a *blockingTurnAnalyzer) SpeechTriggered() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.triggered
}
func (a *blockingTurnAnalyzer) SetSampleRate(int) {}
func (a *blockingTurnAnalyzer) Clear()            {}

type frameCollector struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *frameCollector) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error {
	return nil
}
func (c *frameCollector) QueueFrame(frame frames.Frame, _ frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, frame)
	c.mu.Unlock()
	return nil
}
func (c *frameCollector) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (c *frameCollector) Link(processors.FrameProcessor)                      {}
func (c *frameCollector) SetPrev(processors.FrameProcessor)                   {}
func (c *frameCollector) Start(context.Context) error                         { return nil }
func (c *frameCollector) Stop() error                                         { return nil }
func (c *frameCollector) Name() string                                        { return "collector" }

func (c *frameCollector) count(name string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, f := range c.frames {
		if f.Name() == name {
			n++
		}
	}
	return n
}

func (c *frameCollector) turnMetrics() []frames.TurnMetricsData {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []frames.TurnMetricsData
	for _, f := range c.frames {
		if m, ok := f.(*frames.MetricsFrame); ok {
			for _, d := range m.Data {
				if t, ok := d.(frames.TurnMetricsData); ok {
					out = append(out, t)
				}
			}
		}
	}
	return out
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

type turnHarness struct {
	p    *VADInputProcessor
	turn *blockingTurnAnalyzer
	sink *frameCollector
}

func newTurnHarness(t *testing.T) *turnHarness {
	t.Helper()
	h := &turnHarness{
		turn: &blockingTurnAnalyzer{entered: make(chan struct{}, 8), release: make(chan struct{}, 8)},
		sink: &frameCollector{},
	}
	h.p = NewVADInputProcessor(&scriptedAnalyzer{params: DefaultVADParams()}, WithTurnAnalyzer(h.turn), WithAudioIdleTimeout(0))
	h.p.Link(h.sink)
	ctx, cancel := context.WithCancel(context.Background())
	if err := h.p.Start(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cancel()
		_ = h.p.Stop()
	})
	start := frames.NewStartFrame()
	start.AudioInSampleRate = 16000
	if err := h.p.QueueFrame(start, frames.Downstream); err != nil {
		t.Fatal(err)
	}
	return h
}

func (h *turnHarness) feed(t *testing.T, states ...VADState) {
	t.Helper()
	for _, s := range states {
		data := make([]byte, scriptedChunkSamples*2)
		data[0] = byte(s)
		if err := h.p.QueueFrame(frames.NewAudioFrame(data, 16000, 1), frames.Downstream); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTurnWorkerEmitsMetricsAndStopsTurn(t *testing.T) {
	h := newTurnHarness(t)
	h.feed(t, VADStateStarting, VADStateSpeaking, VADStateStopping, VADStateQuiet)
	<-h.turn.entered
	h.turn.release <- struct{}{}

	eventually(t, "UserStoppedSpeakingFrame", func() bool { return h.sink.count("UserStoppedSpeakingFrame") == 1 })
	if n := h.sink.count("UserStartedSpeakingFrame"); n != 1 {
		t.Fatalf("got %d UserStartedSpeakingFrame, want 1", n)
	}
	metrics := h.sink.turnMetrics()
	if len(metrics) != 1 || !metrics[0].IsComplete || metrics[0].Probability != 0.9 || metrics[0].E2EProcessingTime != 12*time.Millisecond {
		t.Fatalf("turn metrics = %+v", metrics)
	}

	h.turn.mu.Lock()
	speech := append([]bool(nil), h.turn.speech...)
	h.turn.mu.Unlock()
	want := []bool{false, true, true, false}
	if len(speech) != len(want) {
		t.Fatalf("AppendAudio speech flags = %v, want %v", speech, want)
	}
	for i := range want {
		if speech[i] != want[i] {
			t.Fatalf("AppendAudio speech flags = %v, want %v", speech, want)
		}
	}
}

func TestTurnWorkerDiscardsResultWhenUserResumesDuringInference(t *testing.T) {
	h := newTurnHarness(t)
	h.feed(t, VADStateSpeaking, VADStateQuiet)
	<-h.turn.entered
	h.feed(t, VADStateSpeaking)
	eventually(t, "second UserStartedSpeakingFrame", func() bool { return h.sink.count("UserStartedSpeakingFrame") == 2 })
	h.turn.release <- struct{}{}

	eventually(t, "turn metrics", func() bool { return len(h.sink.turnMetrics()) == 1 })
	time.Sleep(30 * time.Millisecond)
	if n := h.sink.count("UserStoppedSpeakingFrame"); n != 0 {
		t.Fatalf("got %d UserStoppedSpeakingFrame after the user resumed, want 0", n)
	}

	h.feed(t, VADStateQuiet)
	<-h.turn.entered
	h.turn.release <- struct{}{}
	eventually(t, "UserStoppedSpeakingFrame for the resumed turn", func() bool {
		return h.sink.count("UserStoppedSpeakingFrame") == 1
	})
}

type ctxTurnAnalyzer struct {
	mu        sync.Mutex
	triggered bool
	clears    int
	entered   chan struct{}
	cancelled chan struct{}
}

func (a *ctxTurnAnalyzer) AppendAudio(_ []byte, isSpeech bool) turn.EndOfTurnState {
	a.mu.Lock()
	if isSpeech {
		a.triggered = true
	}
	a.mu.Unlock()
	return turn.TurnIncomplete
}

func (a *ctxTurnAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, *turn.TurnMetrics, error) {
	return a.AnalyzeEndOfTurnContext(context.Background())
}

func (a *ctxTurnAnalyzer) AnalyzeEndOfTurnContext(ctx context.Context) (turn.EndOfTurnState, *turn.TurnMetrics, error) {
	a.entered <- struct{}{}
	<-ctx.Done()
	a.cancelled <- struct{}{}
	return turn.TurnIncomplete, nil, ctx.Err()
}

func (a *ctxTurnAnalyzer) SpeechTriggered() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.triggered
}

func (a *ctxTurnAnalyzer) SetSampleRate(int) {}

func (a *ctxTurnAnalyzer) Clear() {
	a.mu.Lock()
	a.triggered = false
	a.clears++
	a.mu.Unlock()
}

func (a *ctxTurnAnalyzer) clearCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.clears
}

func newCtxTurnProcessor(t *testing.T, idle time.Duration) (*VADInputProcessor, *ctxTurnAnalyzer, *frameCollector) {
	t.Helper()
	a := &ctxTurnAnalyzer{entered: make(chan struct{}, 8), cancelled: make(chan struct{}, 8)}
	sink := &frameCollector{}
	p := NewVADInputProcessor(&scriptedAnalyzer{params: DefaultVADParams()}, WithTurnAnalyzer(a), WithAudioIdleTimeout(idle))
	p.Link(sink)
	ctx, cancel := context.WithCancel(context.Background())
	if err := p.Start(ctx); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cancel()
		_ = p.Stop()
	})
	start := frames.NewStartFrame()
	start.AudioInSampleRate = 16000
	if err := p.QueueFrame(start, frames.Downstream); err != nil {
		t.Fatal(err)
	}
	return p, a, sink
}

func feedStates(t *testing.T, p *VADInputProcessor, states ...VADState) {
	t.Helper()
	for _, s := range states {
		data := make([]byte, scriptedChunkSamples*2)
		data[0] = byte(s)
		if err := p.QueueFrame(frames.NewAudioFrame(data, 16000, 1), frames.Downstream); err != nil {
			t.Fatal(err)
		}
	}
}

func awaitSignal(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func TestTurnInferenceCancelledWhenUserResumes(t *testing.T) {
	p, a, sink := newCtxTurnProcessor(t, 0)
	feedStates(t, p, VADStateSpeaking, VADStateQuiet)
	awaitSignal(t, a.entered, "inference to start")

	resumed := time.Now()
	feedStates(t, p, VADStateSpeaking)
	awaitSignal(t, a.cancelled, "in-flight inference to be cancelled")
	if d := time.Since(resumed); d > 500*time.Millisecond {
		t.Fatalf("inference cancelled %v after the user resumed", d)
	}
	time.Sleep(30 * time.Millisecond)
	if n := sink.count("UserStoppedSpeakingFrame"); n != 0 {
		t.Fatalf("got %d UserStoppedSpeakingFrame for a cancelled inference", n)
	}
}

func TestStopClearsTurnAnalyzerAndCancelsInference(t *testing.T) {
	p, a, _ := newCtxTurnProcessor(t, 0)
	feedStates(t, p, VADStateSpeaking, VADStateQuiet)
	awaitSignal(t, a.entered, "inference to start")

	if err := p.QueueFrame(frames.NewEndFrame(), frames.Downstream); err != nil {
		t.Fatal(err)
	}
	awaitSignal(t, a.cancelled, "inference to be cancelled on stop")
	eventually(t, "turn analyzer cleared on stop", func() bool { return a.clearCount() == 1 && !a.SpeechTriggered() })
}

func TestAudioIdleClearsTurnAnalyzerAndEndsTurn(t *testing.T) {
	p, a, sink := newCtxTurnProcessor(t, 60*time.Millisecond)
	feedStates(t, p, VADStateSpeaking)
	eventually(t, "UserStartedSpeakingFrame", func() bool { return sink.count("UserStartedSpeakingFrame") == 1 })

	eventually(t, "turn analyzer cleared on audio idle", func() bool { return a.clearCount() == 1 && !a.SpeechTriggered() })
	eventually(t, "UserStoppedSpeakingFrame on audio idle", func() bool { return sink.count("UserStoppedSpeakingFrame") == 1 })
}
