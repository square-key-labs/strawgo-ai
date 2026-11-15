package src

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

func startTask(tb testing.TB, procs ...processors.FrameProcessor) (*pipeline.PipelineTask, func()) {
	tb.Helper()
	task := pipeline.NewPipelineTask(pipeline.NewPipeline(procs))
	started := make(chan struct{})
	task.OnStarted(func() { close(started) })
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() { errc <- task.Run(ctx) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		tb.Fatal("pipeline did not start")
	}
	return task, func() {
		_ = task.QueueFrame(frames.NewEndFrame())
		select {
		case <-task.Done():
		case <-time.After(5 * time.Second):
			tb.Error("pipeline did not finish after EndFrame")
		}
		cancel()
		<-errc
	}
}

func passthroughs(n int) []processors.FrameProcessor {
	out := make([]processors.FrameProcessor, n)
	for i := range out {
		out[i] = processors.NewPassthroughProcessor(fmt.Sprintf("p%d", i), false)
	}
	return out
}

func BenchmarkPipelineAudioFrames(b *testing.B) {
	for _, depth := range []int{1, 8} {
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			task, stop := startTask(b, passthroughs(depth)...)
			defer stop()
			var got atomic.Int64
			done := make(chan struct{})
			target := int64(b.N)
			task.OnFrameReachedDownstream(func(frames.Frame) {
				if got.Add(1) == target {
					close(done)
				}
			}, &frames.AudioFrame{})
			pcm := make([]byte, 640)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := task.QueueFrame(frames.NewAudioFrame(pcm, 16000, 1)); err != nil {
					b.Fatal(err)
				}
			}
			select {
			case <-done:
			case <-time.After(30 * time.Second):
				b.Fatalf("only %d/%d frames reached the sink", got.Load(), b.N)
			}
		})
	}
}

func BenchmarkContextTurn(b *testing.B) {
	base := []llm.Message{llm.System("You are a helpful assistant.")}
	for i := 0; i < 50; i++ {
		base = append(base, llm.User("question"), llm.Assistant("answer"))
	}
	c := llm.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Set(base)
		c.Add(llm.User("one more"))
		snap := c.Snapshot()
		c.Add(llm.Assistant("reply"))
		if len(snap.Messages) != len(base)+1 {
			b.Fatal("bad snapshot")
		}
	}
}

func BenchmarkTwilioSerializeAudio(b *testing.B) {
	s := serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{StreamSID: "MZ1"})
	pcm := make([]byte, 24000*2/50)
	var dst []serializers.Message
	var err error
	b.SetBytes(int64(len(pcm)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst, err = s.SerializeAudio(dst[:0], pcm, 24000)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTwilioDeserializeMedia(b *testing.B) {
	s := serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{StreamSID: "MZ1"})
	in, err := s.SerializeAudio(nil, make([]byte, 320), 8000)
	if err != nil {
		b.Fatal(err)
	}
	msg := serializers.Message{Type: in[0].Type, Data: append([]byte(nil), in[0].Data...)}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.Deserialize(msg); err != nil {
			b.Fatal(err)
		}
	}
}

func TestUserAggregatorConcurrentProducers(t *testing.T) {
	c := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(c, aggregators.ContextAggregatorPairParams{})
	task, stop := startTask(t, pair.User(), processors.NewPassthroughProcessor("llm", false), pair.Assistant())

	var wg sync.WaitGroup
	const iterations = 200
	produce := func(fn func(i int) error) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				if err := fn(i); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	produce(func(int) error {
		if err := task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2)); err != nil {
			return err
		}
		return task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2))
	})
	produce(func(i int) error {
		return task.QueueFrame(frames.NewTranscriptionFrame(fmt.Sprintf("hello %d", i), true))
	})
	produce(func(int) error {
		if err := task.QueueFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream); err != nil {
			return err
		}
		return task.QueueFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
	})
	produce(func(i int) error {
		if i%20 != 0 {
			return nil
		}
		return task.QueueFrame(frames.NewInterruptionFrame())
	})
	wg.Wait()
	stop()
	for _, m := range c.Messages() {
		if m.Role != llm.RoleUser && m.Role != llm.RoleAssistant {
			t.Fatalf("unexpected role %q in context", m.Role)
		}
	}
}
