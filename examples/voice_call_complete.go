package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/interruptions"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

// Complete voice call example using PCM pipeline with audio converters
// This approach provides maximum flexibility for audio processing
func main() {
	// Get API keys from environment
	deepgramKey := os.Getenv("DEEPGRAM_API_KEY")
	elevenLabsKey := os.Getenv("ELEVENLABS_API_KEY")
	elevenLabsVoice := os.Getenv("ELEVENLABS_VOICE_ID")
	openaiKey := os.Getenv("OPENAI_API_KEY")

	if deepgramKey == "" || elevenLabsKey == "" || openaiKey == "" {
		log.Fatal("Missing required API keys. Check .env file")
	}

	if elevenLabsVoice == "" {
		elevenLabsVoice = "21m00Tcm4TlvDq8ikWAM" // Default voice
	}

	// Create Twilio serializer (handles Twilio Media Streams protocol)
	twilioSerializer := serializers.NewTwilioFrameSerializer("", "")

	// Create WebSocket transport with Twilio serializer
	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       8080,
		Path:       "/media",
		Serializer: twilioSerializer,
	})

	// Create audio converters for PCM pipeline
	// Convert mulaw (from Twilio) -> PCM (for Deepgram)
	inputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
		InputSampleRate:  8000,
		InputCodec:       "mulaw",
		OutputSampleRate: 16000,
		OutputCodec:      "linear16",
	})

	// Convert PCM (from ElevenLabs) -> mulaw (for Twilio)
	outputConverter := audio.NewAudioConverterProcessor(audio.AudioConverterConfig{
		InputSampleRate:  24000,
		InputCodec:       "linear16",
		OutputSampleRate: 8000,
		OutputCodec:      "mulaw",
	})

	// Create AI services with PCM settings
	deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   deepgramKey,
		Language: "en",
		Model:    "nova-2",
		Encoding: "linear16", // PCM format
	})

	// Create shared LLM context with system prompt
	llmContext := services.NewLLMContext(`You are a helpful voice assistant. Keep responses brief and conversational,
as this is a phone conversation. Speak naturally and be concise.`)

	// Create aggregators for context management
	userAgg := aggregators.NewLLMUserAggregator(llmContext, aggregators.DefaultUserAggregatorParams())
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
		OutputFormat: "pcm_24000", // PCM output
		UseStreaming: true,
	})

	// Build pipeline WITH aggregators and interruptions
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),  // WebSocket input with Twilio serializer
		inputConverter,     // mulaw -> PCM conversion
		deepgramSTT,        // PCM processing → TranscriptionFrame
		userAgg,            // User aggregator → LLMContextFrame (with interruptions!)
		openaiLLM,          // LLM → TextFrame
		elevenLabsTTS,      // TTS → TTSAudioFrame (emits TTSStarted/Stopped)
		outputConverter,    // PCM -> mulaw conversion
		transport.Output(), // WebSocket output with Twilio serializer
		assistantAgg,       // Assistant aggregator (updates context)
	})

	// Configure interruptions
	config := &pipeline.PipelineTaskConfig{
		AllowInterruptions: true,
		InterruptionStrategies: []interruptions.InterruptionStrategy{
			interruptions.NewMinWordsInterruptionStrategy(3), // Interrupt after 3 words
		},
	}

	// Create task with interruption config
	task := pipeline.NewPipelineTaskWithConfig(pipe, config)

	// Setup event handlers
	task.OnStarted(func() {
		fmt.Println("✓ Pipeline started successfully")
		fmt.Println("✓ Twilio webhook listening on http://localhost:8080/twilio")
		fmt.Println("✓ Using PCM pipeline with audio conversions")
		fmt.Println("✓ LLM aggregators enabled with context management")
		fmt.Println("✓ Interruptions enabled (3+ words triggers interruption)")
		fmt.Println("\nConfigure your Twilio phone number webhook to:")
		fmt.Println("  http://YOUR_SERVER:8080/twilio")
		fmt.Println("\nPress Ctrl+C to stop")
		fmt.Println("\nTry saying 'Hey wait stop' while the bot is speaking to test interruptions!")
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

	// Start WebSocket transport
	go func() {
		if err := transport.Start(ctx); err != nil {
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
