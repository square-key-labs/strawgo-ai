//go:build ignore

package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Printf("Starting Audio-Based Interruption Example...")

	// Get API keys from environment
	deepgramAPIKey := os.Getenv("DEEPGRAM_API_KEY")
	elevenLabsAPIKey := os.Getenv("ELEVENLABS_API_KEY")
	geminiAPIKey := os.Getenv("GEMINI_API_KEY")

	if deepgramAPIKey == "" || elevenLabsAPIKey == "" || geminiAPIKey == "" {
		log.Fatal("Missing API keys. Set DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, and GEMINI_API_KEY")
	}

	turnStrategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(true),
			user_start.NewMinWordsUserTurnStartStrategy(2, true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(700*time.Millisecond, true),
		},
		MuteStrategies: []user_mute.UserMuteStrategy{
			user_mute.NewFirstSpeechUserMuteStrategy(true),
			user_mute.NewFunctionCallUserMuteStrategy(true),
		},
	}

	// =============================================================================
	// SERVICES SETUP
	// =============================================================================

	// STT: Deepgram
	sttService := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   deepgramAPIKey,
		Language: "en-US",
		Model:    "nova-2",
		Encoding: "mulaw", // For Asterisk compatibility
	})

	// TTS: ElevenLabs
	ttsService := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       elevenLabsAPIKey,
		VoiceID:      "21m00Tcm4TlvDq8ikWAM", // Rachel voice
		Model:        "eleven_turbo_v2",
		OutputFormat: "ulaw_8000", // For Asterisk
		UseStreaming: true,
	})

	// LLM: Gemini
	llmService := gemini.NewLLMService(gemini.LLMConfig{
		APIKey: geminiAPIKey,
		Model:  "gemini-2.0-flash-exp",
	})

	// =============================================================================
	// CONTEXT AND AGGREGATORS
	// =============================================================================

	// Create LLM context with system message
	llmContext := services.NewLLMContext("You are a helpful assistant. Keep responses concise and natural.")

	// User aggregator - handles user input and AUDIO-BASED INTERRUPTION
	userAgg := aggregators.NewLLMUserAggregator(
		llmContext,
		turnStrategies,
	)

	// Assistant aggregator - handles bot responses
	assistantAgg := aggregators.NewLLMAssistantAggregator(
		llmContext,
		aggregators.DefaultAssistantAggregatorParams(),
	)

	// =============================================================================
	// TRANSPORT SETUP
	// =============================================================================

	// WebSocket transport with Asterisk serializer
	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port: 8080,
		Path: "/ws",
		Serializer: serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
			Codec:      "mulaw",
			SampleRate: 8000,
		}),
	})

	// =============================================================================
	// PIPELINE CONFIGURATION WITH AUDIO-BASED INTERRUPTION
	// =============================================================================

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),
		sttService,
		userAgg,
		llmService,
		assistantAgg,
		ttsService,
		transport.Output(),
	})

	task := pipeline.NewPipelineTaskWithConfig(
		pipe,
		&pipeline.PipelineTaskConfig{
			AllowInterruptions: true,
			TurnStrategies:     turnStrategies,
		},
	)

	// =============================================================================
	// RUN PIPELINE
	// =============================================================================

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutting down...")
		cancel()
	}()

	// Start WebSocket server
	go func() {
		if err := transport.Start(ctx); err != nil {
			log.Printf("Transport error: %v", err)
		}
	}()

	// Run pipeline
	log.Printf("✅ Pipeline ready with turn strategies")
	log.Printf("📊 Strategy: VAD + min words start, speech-timeout stop, first-speech mute")
	log.Printf("🎤 Connect to ws://localhost:8080/ws")
	log.Printf("🔴 Interruptions trigger when turn-start strategies detect user speech")

	if err := task.Run(ctx); err != nil {
		log.Fatalf("Pipeline error: %v", err)
	}

	log.Println("Pipeline stopped")
}
