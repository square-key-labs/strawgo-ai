package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/observers"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

var tools = llm.Tools{Functions: []llm.FunctionSchema{{
	Name:        "get_time",
	Description: "Current local time of the service.",
	Properties:  map[string]any{},
}}}

func main() {
	logger.Init()
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		log.Fatal("GEMINI_API_KEY is required")
	}
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: ":8080",
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: func(ctx context.Context, s *transports.Session) error { return runCall(ctx, s, apiKey) },
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := srv.ListenAndServe(ctx); err != nil {
		log.Fatal(err)
	}
}

func runCall(ctx context.Context, s *transports.Session, apiKey string) error {
	live := gemini.NewLiveService(gemini.LiveConfig{
		APIKey:            apiKey,
		Model:             "gemini-2.5-flash-native-audio-preview-09-2025",
		SystemInstruction: "You are a helpful voice assistant on a phone call. Keep responses brief and conversational.",
		Voice:             "Puck",
		Tools:             tools,
	})
	live.RegisterFunction("get_time", func(context.Context, llm.FunctionCallParams) (llm.FunctionResult, error) {
		return llm.FunctionResult{Value: map[string]any{"time": time.Now().Format(time.Kitchen)}}, nil
	})

	conversation := llm.NewContext(llm.User("Greet the caller and ask how you can help."))
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), pair.User(), live, s.Output(), pair.Assistant(),
	})
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)

	obs := pipeline.NewTaskObserver()
	obs.AddObserver(observers.NewTranscriptionLogObserver())
	obs.AddObserver(observers.NewMetricsLogObserver())
	task.SetObserver(obs)

	if err := task.QueueFrame(frames.NewLLMRunFrame()); err != nil {
		return err
	}
	err := task.Run(ctx)
	for _, m := range conversation.Messages() {
		log.Printf("%s: %s", m.Role, m.Text())
	}
	return err
}
