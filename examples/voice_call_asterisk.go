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
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

// Voice call example using Asterisk WebSocket with codec passthrough
// Supports both mulaw (North America) and alaw (Europe) with zero audio conversions
// This demonstrates integration with Asterisk PBX systems using native telephony codecs
func main() {
	// Get API keys from environment
	deepgramKey := os.Getenv("DEEPGRAM_API_KEY")
	elevenLabsKey := os.Getenv("ELEVENLABS_API_KEY")
	elevenLabsVoice := os.Getenv("ELEVENLABS_VOICE_ID")
	geminiKey := os.Getenv("GEMINI_API_KEY")

	if deepgramKey == "" || elevenLabsKey == "" || geminiKey == "" {
		log.Fatal("Missing required API keys. Check .env file")
	}

	if elevenLabsVoice == "" {
		elevenLabsVoice = "21m00Tcm4TlvDq8ikWAM" // Default voice
	}

	// Configure codec for Asterisk
	// Use "mulaw" for North America/Twilio, "alaw" for Europe/Telnyx
	asteriskCodec := "mulaw" // Change to "alaw" for European deployments

	// Create Asterisk serializer (handles Asterisk WebSocket protocol)
	// Using binary mode for raw codec frames with passthrough
	asteriskSerializer := serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
		ChannelID:  "", // Will be set from Asterisk messages
		UseBinary:  true,
		Codec:      asteriskCodec, // Configurable codec
		SampleRate: 8000,          // Telephony standard
	})

	// Create WebSocket transport with Asterisk serializer
	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       8080,
		Path:       "/asterisk", // Asterisk WebSocket endpoint
		Serializer: asteriskSerializer,
	})

	// Create AI services with codec passthrough (zero conversions!)
	// Deepgram supports mulaw and alaw directly
	deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   deepgramKey,
		Language: "en",
		Model:    "nova-2",
		Encoding: asteriskCodec, // Passthrough: use same codec as Asterisk
	})

	// Using Gemini instead of OpenAI for this example
	geminiLLM := gemini.NewLLMService(gemini.LLMConfig{
		APIKey:      geminiKey,
		Model:       "gemini-pro",
		Temperature: 0.7,
		SystemPrompt: `You are a helpful voice assistant for an Asterisk phone system.
Keep responses brief and conversational. Speak naturally and be concise.`,
	})

	// ElevenLabs supports both ulaw_8000 and alaw_8000
	// Map codec to ElevenLabs format
	ttsFormat := "ulaw_8000"
	if asteriskCodec == "alaw" {
		ttsFormat = "alaw_8000"
	}

	elevenLabsTTS := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       elevenLabsKey,
		VoiceID:      elevenLabsVoice,
		Model:        "eleven_turbo_v2",
		OutputFormat: ttsFormat, // Passthrough: output same codec as Asterisk
		UseStreaming: true,
	})

	// Build pipeline with NO audio converters
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),  // WebSocket input with Asterisk serializer
		deepgramSTT,
		geminiLLM,
		elevenLabsTTS,
		transport.Output(), // WebSocket output with Asterisk serializer
	})

	// Create and configure task
	task := pipeline.NewPipelineTask(pipe)

	// Setup event handlers
	task.OnStarted(func() {
		fmt.Println("✓ Pipeline started successfully")
		fmt.Println("✓ Asterisk WebSocket listening on ws://localhost:8080/asterisk")
		fmt.Printf("✓ Using %s codec passthrough (zero audio conversions)\n", asteriskCodec)
		fmt.Printf("✓ Pipeline: Asterisk (%s) → Deepgram (%s) → Gemini → ElevenLabs (%s) → Asterisk (%s)\n",
			asteriskCodec, asteriskCodec, ttsFormat, asteriskCodec)
		fmt.Println("\nConfigure your Asterisk dialplan:")
		fmt.Println("  exten => _X.,1,Answer()")
		fmt.Println("  same => n,Stasis(strawgo)")
		fmt.Println("\nAnd in ari.conf:")
		fmt.Println("  [strawgo]")
		fmt.Println("  type=ws")
		fmt.Println("  url=ws://YOUR_SERVER:8080/asterisk")
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
