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

	// Configure fallback codec for Asterisk (auto-detected from MEDIA_START)
	// Use "mulaw" for North America, "alaw" for Europe
	// This is only used as fallback if MEDIA_START message is not received
	asteriskCodec := "mulaw" // Change to "alaw" for European deployments

	// Create Asterisk serializer (handles Asterisk WebSocket protocol)
	// Codec will be auto-detected from MEDIA_START control message
	asteriskSerializer := serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
		ChannelID:  "", // Will be set from MEDIA_START message
		Codec:      asteriskCodec, // Fallback codec if auto-detection fails
		SampleRate: 8000,          // Fallback sample rate
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
		fmt.Println("✓ Codec will be auto-detected from MEDIA_START message")
		fmt.Printf("✓ Fallback codec: %s (if auto-detection fails)\n", asteriskCodec)
		fmt.Println("✓ Zero audio conversions with codec passthrough")
		fmt.Println("\nConfigure your Asterisk dialplan:")
		fmt.Println("  exten => _X.,1,Answer()")
		fmt.Println("  same => n,ExternalMedia(ws://YOUR_SERVER:8080/asterisk,c(ulaw))")
		fmt.Println("\nSupported codecs:")
		fmt.Println("  • ulaw (North America, Japan)")
		fmt.Println("  • alaw (Europe, rest of world)")
		fmt.Println("  • slin/slin16 (linear PCM)")
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
