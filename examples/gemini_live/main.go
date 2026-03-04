package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

// Gemini Live S2S example
// Demonstrates "fat processor" pattern - Gemini Live replaces STT+LLM+TTS pipeline
func main() {
	geminiKey := os.Getenv("GEMINI_API_KEY")

	if geminiKey == "" {
		log.Fatal("Missing GEMINI_API_KEY environment variable")
	}

	twilioSerializer := serializers.NewTwilioFrameSerializer("", "")

	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       8080,
		Path:       "/media",
		Serializer: twilioSerializer,
	})

	live := gemini.NewLiveService(gemini.LiveConfig{
		APIKey:            geminiKey,
		Model:             "gemini-2.0-flash-exp",
		SystemInstruction: "You are a helpful voice assistant. Keep responses brief and conversational.",
		Voice:             "Puck",
		InputMIMEType:     "audio/pcm;rate=8000",
		OutputSampleRate:  8000,
		OutputChannels:    1,
	})

	turnStrategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(2*time.Second, true),
		},
	}

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),
		live,
		transport.Output(),
	})

	config := &pipeline.PipelineTaskConfig{
		AllowInterruptions: true,
		TurnStrategies:     turnStrategies,
	}

	task := pipeline.NewPipelineTaskWithConfig(pipe, config)

	task.OnStarted(func() {
		fmt.Println("✓ Gemini Live S2S bot started")
		fmt.Println("✓ Twilio webhook listening on http://localhost:8080/media")
		fmt.Println("✓ Using Gemini Live (replaces STT+LLM+TTS)")
		fmt.Println("\nConfigure your Twilio phone number webhook to:")
		fmt.Println("  http://YOUR_SERVER:8080/media")
		fmt.Println("\nPress Ctrl+C to stop")
	})

	task.OnError(func(err error) {
		log.Printf("Pipeline error: %v", err)
	})

	task.OnFinished(func() {
		fmt.Println("\n✓ Pipeline stopped gracefully")
	})

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		if err := transport.Start(ctx); err != nil {
			log.Printf("Transport error: %v", err)
		}
	}()

	go func() {
		if err := task.Run(ctx); err != nil {
			log.Printf("Pipeline error: %v", err)
		}
	}()

	<-sigChan
	fmt.Println("\nShutting down...")
	cancel()
}
