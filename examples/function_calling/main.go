package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
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
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/groq"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

const systemPrompt = `You are the phone assistant of a small weather service.
Use get_weather to answer weather questions and end_call when the caller says goodbye.
Keep every answer to one short spoken sentence.`

var tools = llm.Tools{Functions: []llm.FunctionSchema{
	{
		Name:        "get_weather",
		Description: "Current weather for a city.",
		Properties: map[string]any{
			"city": map[string]any{"type": "string", "description": "City name, e.g. Paris"},
			"unit": map[string]any{"type": "string", "enum": []string{"celsius", "fahrenheit"}},
		},
		Required: []string{"city"},
	},
	{
		Name:        "end_call",
		Description: "Hang up after the caller has said goodbye.",
		Properties:  map[string]any{},
	},
}}

type bot struct {
	deepgramKey   string
	groqKey       string
	elevenLabsKey string
	voiceID       string
}

func main() {
	logger.Init()
	b := bot{
		deepgramKey:   mustEnv("DEEPGRAM_API_KEY"),
		groqKey:       mustEnv("GROQ_API_KEY"),
		elevenLabsKey: mustEnv("ELEVENLABS_API_KEY"),
		voiceID:       envOr("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
	}
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: envOr("ADDR", ":8080"),
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: b.runCall,
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

func (b bot) runCall(ctx context.Context, s *transports.Session) error {
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: b.deepgramKey, Model: "nova-3", Language: "en"})
	llmSvc := groq.NewLLMService(groq.LLMConfig{APIKey: b.groqKey, Model: "llama-3.3-70b-versatile", SystemPrompt: systemPrompt})
	tts := elevenlabs.NewTTSService(elevenlabs.TTSConfig{APIKey: b.elevenLabsKey, VoiceID: b.voiceID, Model: "eleven_flash_v2_5", UseStreaming: true})

	conversation := llm.NewContext()
	conversation.SetTools(tools)
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)

	llmSvc.RegisterFunction("get_weather", getWeather)
	llmSvc.RegisterFunction("end_call", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		if err := task.QueueFrame(frames.NewTTSSpeakFrame("Thanks for calling, goodbye!")); err != nil {
			return llm.FunctionResult{}, err
		}
		if err := task.QueueFrame(frames.NewEndFrame()); err != nil {
			return llm.FunctionResult{}, err
		}
		runLLM := false
		return llm.FunctionResult{Value: "call ended", RunLLM: &runLLM}, nil
	}, llm.FunctionOptions{CancelOnInterruption: false})

	latency := observers.NewUserBotLatencyObserver()
	latency.OnLatencyBreakdown(func(bd observers.LatencyBreakdown) {
		log.Printf("response latency %v:\n  %s", bd.Total.Round(time.Millisecond), strings.Join(bd.Lines(true), "\n  "))
	})
	obs := pipeline.NewTaskObserver()
	obs.AddObserver(latency)
	task.SetObserver(obs)

	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Weather line, which city are you interested in?")); err != nil {
		return err
	}
	return task.Run(ctx)
}

func getWeather(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
	city, _ := p.Arguments["city"].(string)
	if city == "" {
		return llm.FunctionResult{}, fmt.Errorf("city is required")
	}
	unit, _ := p.Arguments["unit"].(string)
	temp, symbol := 21, "C"
	if unit == "fahrenheit" {
		temp, symbol = 70, "F"
	}
	select {
	case <-ctx.Done():
		return llm.FunctionResult{}, ctx.Err()
	case <-time.After(150 * time.Millisecond):
	}
	return llm.FunctionResult{Value: map[string]any{
		"city":        city,
		"temperature": fmt.Sprintf("%d°%s", temp, symbol),
		"conditions":  "partly cloudy",
	}}, nil
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s is required", key)
	}
	return v
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
