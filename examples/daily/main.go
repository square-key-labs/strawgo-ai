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
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports/daily"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

// Daily WebRTC voice bot example
// Demonstrates peer-to-peer audio streaming with Daily.co platform
func main() {
	// Get API keys from environment
	dailyKey := os.Getenv("DAILY_API_KEY")
	deepgramKey := os.Getenv("DEEPGRAM_API_KEY")
	elevenLabsKey := os.Getenv("ELEVENLABS_API_KEY")
	elevenLabsVoice := os.Getenv("ELEVENLABS_VOICE_ID")
	openaiKey := os.Getenv("OPENAI_API_KEY")

	if dailyKey == "" || deepgramKey == "" || elevenLabsKey == "" || openaiKey == "" {
		log.Fatal("Missing required API keys. Set DAILY_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, OPENAI_API_KEY")
	}

	if elevenLabsVoice == "" {
		elevenLabsVoice = "21m00Tcm4TlvDq8ikWAM" // Default voice
	}

	roomName := os.Getenv("DAILY_ROOM_NAME")
	if roomName == "" {
		roomName = "strawgo-voice-bot" // Default room name
	}

	// Create Daily transport
	transport := daily.NewDailyTransport(daily.DailyConfig{
		APIKey:             dailyKey,
		RoomName:           roomName,
		CreateRoomIfAbsent: true,
		SampleRate:         48000,
		Channels:           1,
		OnParticipantJoin: func(participantID string) {
			log.Printf("Participant joined: %s", participantID)
		},
		OnParticipantLeave: func(participantID string) {
			log.Printf("Participant left: %s", participantID)
		},
	})

	// Create AI services
	deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   deepgramKey,
		Language: "en",
		Model:    "nova-2",
		Encoding: "opus", // Daily uses Opus codec
	})

	// Create shared LLM context with system prompt
	llmContext := services.NewLLMContext(`You are a helpful voice assistant. Keep responses brief and conversational.`)
	turnStrategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(2*time.Second, true),
		},
		MuteStrategies: []user_mute.UserMuteStrategy{
			user_mute.NewFirstSpeechUserMuteStrategy(true),
		},
	}

	// Create aggregators for context management
	userAgg := aggregators.NewLLMUserAggregator(llmContext, turnStrategies)
	assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, aggregators.DefaultAssistantAggregatorParams())

	openaiLLM := openai.NewLLMService(openai.LLMConfig{
		APIKey:      openaiKey,
		Model:       "gpt-4-turbo-preview",
		Temperature: 0.7,
	})

	elevenLabsTTS := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       elevenLabsKey,
		VoiceID:      elevenLabsVoice,
		Model:        "eleven_turbo_v2",
		OutputFormat: "pcm_24000",
		UseStreaming: true,
	})

	// Build pipeline: Transport → STT → Aggregator → LLM → TTS → Transport
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),
		deepgramSTT,
		userAgg,
		openaiLLM,
		elevenLabsTTS,
		transport.Output(),
		assistantAgg,
	})

	// Configure pipeline with turn strategies
	config := &pipeline.PipelineTaskConfig{
		AllowInterruptions: true,
		TurnStrategies:     turnStrategies,
	}

	task := pipeline.NewPipelineTaskWithConfig(pipe, config)

	// Setup event handlers
	task.OnStarted(func() {
		fmt.Println("✓ Daily WebRTC voice bot started")
		fmt.Printf("✓ Room: %s\n", roomName)
		fmt.Println("✓ Join the room at: https://strawgo.daily.co/" + roomName)
		fmt.Println("\nPress Ctrl+C to stop")
	})

	task.OnError(func(err error) {
		log.Printf("Pipeline error: %v", err)
	})

	task.OnFinished(func() {
		fmt.Println("\n✓ Pipeline stopped gracefully")
	})

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Connect to Daily room
	go func() {
		if err := transport.Connect(ctx); err != nil {
			log.Printf("Transport error: %v", err)
		}
	}()

	// Run pipeline in background
	go func() {
		if err := task.Run(ctx); err != nil {
			log.Printf("Pipeline error: %v", err)
		}
	}()

	// Wait for shutdown signal
	<-sigChan
	fmt.Println("\nShutting down...")
	cancel()
}
