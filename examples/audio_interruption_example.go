package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/interruptions"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/transports"
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

	// =============================================================================
	// AUDIO-BASED INTERRUPTION STRATEGIES
	// =============================================================================

	// Option 1: Volume-Based Strategy (Simple, Fast)
	// Detects interruption when user's voice exceeds volume threshold
	volumeStrategy := interruptions.NewVolumeInterruptionStrategy(
		&interruptions.VolumeInterruptionStrategyParams{
			Threshold:  0.02,  // RMS volume threshold (lower = more sensitive)
			WindowSize: 10,    // Analyze last 10 frames (~200ms)
			MinFrames:  3,     // Need 3 frames above threshold to trigger
		},
	)

	// Option 2: VAD-Based Strategy (Robust, Accurate)
	// Detects interruption using voice activity detection (energy + zero-crossing rate)
	// vadStrategy := interruptions.NewVADBasedInterruptionStrategy(
	// 	&interruptions.VADBasedInterruptionStrategyParams{
	// 		MinDuration:     300 * time.Millisecond, // Minimum speech duration
	// 		EnergyThreshold: 0.02,                   // Energy threshold
	// 		ZeroCrossRate:   0.1,                    // Zero-crossing rate threshold
	// 	},
	// )

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
	context := services.NewLLMContext([]services.LLMMessage{
		{
			Role:    "system",
			Content: "You are a helpful assistant. Keep responses concise and natural.",
		},
	})

	// User aggregator - handles user input and AUDIO-BASED INTERRUPTION
	userAgg := aggregators.NewLLMUserAggregator(
		context,
		aggregators.DefaultUserAggregatorParams(),
	)

	// Assistant aggregator - handles bot responses
	assistantAgg := aggregators.NewLLMAssistantAggregator(
		context,
		aggregators.DefaultAssistantAggregatorParams(),
	)

	// =============================================================================
	// TRANSPORT SETUP
	// =============================================================================

	// WebSocket transport with Asterisk serializer
	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       8080,
		Path:       "/ws",
		Serializer: serializers.NewAsteriskSerializer(),
	})

	// =============================================================================
	// PIPELINE CONFIGURATION WITH AUDIO-BASED INTERRUPTION
	// =============================================================================

	task, err := pipeline.NewPipelineTask(
		&pipeline.PipelineTaskConfig{
			// CRITICAL: Enable interruptions
			AllowInterruptions: true,

			// AUDIO-BASED STRATEGIES (No text-based strategies!)
			InterruptionStrategies: []interruptions.InterruptionStrategy{
				volumeStrategy, // Use volume-based detection
				// vadStrategy,  // OR use VAD-based detection
				// You can use BOTH for OR logic (either triggers interruption)
			},
		},
		[]pipeline.FrameProcessor{
			transport.Input(),   // WebSocket input
			sttService,          // Speech-to-text
			userAgg,             // User aggregator (HANDLES AUDIO-BASED INTERRUPTION)
			llmService,          // LLM processing
			assistantAgg,        // Assistant aggregator
			ttsService,          // Text-to-speech
			transport.Output(),  // WebSocket output
		},
	)

	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}

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
	log.Printf("✅ Pipeline ready with AUDIO-BASED interruption!")
	log.Printf("📊 Strategy: Volume-based (threshold=0.02, window=10, min_frames=3)")
	log.Printf("🎤 Connect to ws://localhost:8080/ws")
	log.Printf("🔴 Interruptions will trigger on audio volume (no text needed!)")

	if err := task.Run(ctx); err != nil {
		log.Fatalf("Pipeline error: %v", err)
	}

	log.Println("Pipeline stopped")
}
