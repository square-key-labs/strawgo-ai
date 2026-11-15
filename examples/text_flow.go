//go:build ignore

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/observers"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
)

type console struct {
	*processors.BaseProcessor
	replied chan struct{}
}

func newConsole() *console {
	c := &console{replied: make(chan struct{}, 1)}
	c.BaseProcessor = processors.NewBaseProcessor("Console", c)
	return c
}

func (c *console) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.LLMTextFrame:
		fmt.Print(f.Text)
	case *frames.LLMFullResponseEndFrame:
		fmt.Println()
		select {
		case c.replied <- struct{}{}:
		default:
		}
	case *frames.ErrorFrame:
		fmt.Fprintf(os.Stderr, "error: %v\n", f)
	}
	return c.PushFrame(frame, direction)
}

func main() {
	logger.SetLevel(logger.WARN)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required (set OPENAI_BASE_URL to use any OpenAI-compatible server)")
	}
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       apiKey,
		BaseURL:      os.Getenv("OPENAI_BASE_URL"),
		Model:        envOr("OPENAI_MODEL", "gpt-4.1-mini"),
		SystemPrompt: "You are a concise, friendly assistant.",
	})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	out := newConsole()
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), llmSvc, out, pair.Assistant()})

	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	cfg.EnableUsageMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)
	if os.Getenv("SHOW_METRICS") != "" {
		obs := pipeline.NewTaskObserver()
		obs.AddObserver(observers.NewMetricsLogObserver())
		task.SetObserver(obs)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		in := bufio.NewScanner(os.Stdin)
		for fmt.Print("> "); in.Scan(); fmt.Print("> ") {
			text := strings.TrimSpace(in.Text())
			if text == "" {
				continue
			}
			if err := task.QueueFrame(frames.NewLLMMessagesAppendFrame([]llm.Message{llm.User(text)}, true)); err != nil {
				return
			}
			select {
			case <-out.replied:
			case <-task.Done():
				return
			}
		}
		_ = task.StopWhenDone()
	}()

	if err := task.Run(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nconversation had %d messages\n", conversation.Len())
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
