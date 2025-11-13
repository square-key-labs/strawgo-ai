package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

// Voice call example using mulaw passthrough (zero conversions!)
// This approach provides the best performance for telephony applications
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
	twilioSerializer := serializers.NewTwilioFrameSerializer("", "") // StreamSid/CallSid set on connection

	// Create WebSocket transport with Twilio serializer
	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       8080,
		Path:       "/media", // Twilio Media Streams endpoint
		Serializer: twilioSerializer,
	})

	// Create AI services with mulaw support (zero conversions!)
	deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   deepgramKey,
		Language: "en",
		Model:    "nova-2",
		Encoding: "mulaw", // Accept mulaw directly - no conversion needed!
	})

	openaiLLM := openai.NewLLMService(openai.LLMConfig{
		APIKey:      openaiKey,
		Model:       "gpt-4-turbo-preview",
		Temperature: 0.7,
		SystemPrompt: `You are a helpful voice assistant. Keep responses brief and conversational,
as this is a phone conversation. Speak naturally and be concise.`,
	})

	elevenLabsTTS := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       elevenLabsKey,
		VoiceID:      elevenLabsVoice,
		Model:        "eleven_turbo_v2",
		OutputFormat: "ulaw_8000", // Output mulaw directly - no conversion needed!
		UseStreaming: true,
	})

	// Build pipeline with NO audio converters - mulaw stays mulaw throughout!
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),  // WebSocket input with Twilio serializer
		deepgramSTT,        // Accepts mulaw input
		openaiLLM,          // Text processing
		elevenLabsTTS,      // Outputs mulaw
		transport.Output(), // WebSocket output with Twilio serializer
	})

	// Create and configure task
	task := pipeline.NewPipelineTask(pipe)

	// Setup event handlers
	task.OnStarted(func() {
		fmt.Println("✓ Pipeline started successfully")
		fmt.Println("✓ Twilio webhook listening on http://localhost:8080/twilio")
		fmt.Println("✓ Using mulaw passthrough (zero audio conversions)")
		fmt.Println("\nConfigure your Twilio phone number webhook to:")
		fmt.Println("  http://YOUR_SERVER:8080/twilio")
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
