package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

// Voice call example with SileroVAD integration
// Demonstrates Voice Activity Detection for detecting when user starts/stops speaking
// Uses Asterisk WebSocket with codec passthrough + Silero VAD model
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
	asteriskCodec := "mulaw" // Change to "alaw" for European deployments

	// Create Asterisk serializer
	asteriskSerializer := serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
		ChannelID:  "",
		Codec:      asteriskCodec,
		SampleRate: 8000,
	})

	// Create WebSocket transport with Asterisk serializer
	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       8080,
		Path:       "/asterisk",
		Serializer: asteriskSerializer,
	})

	// ========================================
	// SILERO VAD CONFIGURATION
	// ========================================
	// Configure VAD parameters
	vadParams := vad.VADParams{
		Confidence: 0.7,  // Voice confidence threshold (0.0-1.0)
		StartSecs:  0.2,  // Delay before detecting speech start
		StopSecs:   0.8,  // Delay before detecting speech end
		MinVolume:  0.6,  // Minimum volume threshold
	}

	// Create Silero VAD analyzer
	// Model is automatically loaded from embedded package data (no path needed!)
	sileroAnalyzer, err := vad.NewSileroVADAnalyzer(8000, vadParams)
	if err != nil {
		log.Fatalf("Failed to create SileroVAD analyzer: %v", err)
	}

	// Create VAD input processor
	vadProcessor := vad.NewVADInputProcessor(sileroAnalyzer)

	fmt.Println("✓ SileroVAD initialized")
	fmt.Printf("  • Confidence threshold: %.2f\n", vadParams.Confidence)
	fmt.Printf("  • Start delay: %.2fs\n", vadParams.StartSecs)
	fmt.Printf("  • Stop delay: %.2fs\n", vadParams.StopSecs)
	fmt.Printf("  • Min volume: %.2f\n\n", vadParams.MinVolume)

	// ========================================
	// AI SERVICES CONFIGURATION
	// ========================================

	// Deepgram STT with codec passthrough
	deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   deepgramKey,
		Language: "en",
		Model:    "nova-2",
		Encoding: asteriskCodec,
	})

	// Gemini LLM
	geminiLLM := gemini.NewLLMService(gemini.LLMConfig{
		APIKey:      geminiKey,
		Model:       "gemini-pro",
		Temperature: 0.7,
		SystemPrompt: `You are a helpful voice assistant with VAD support.
You can detect when users start and stop speaking.
Keep responses brief and conversational. Speak naturally and be concise.`,
	})

	// ElevenLabs TTS with codec passthrough
	ttsFormat := "ulaw_8000"
	if asteriskCodec == "alaw" {
		ttsFormat = "alaw_8000"
	}

	elevenLabsTTS := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
		APIKey:       elevenLabsKey,
		VoiceID:      elevenLabsVoice,
		Model:        "eleven_turbo_v2",
		OutputFormat: ttsFormat,
		UseStreaming: true,
	})

	// ========================================
	// PIPELINE CONSTRUCTION
	// ========================================
	// Build pipeline with VAD processor after WebSocket input
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		transport.Input(),  // WebSocket input (receives audio from Asterisk)
		vadProcessor,       // ← SILERO VAD (detects when user speaks)
		deepgramSTT,        // Speech-to-Text
		geminiLLM,          // LLM processing
		elevenLabsTTS,      // Text-to-Speech
		transport.Output(), // WebSocket output (sends audio to Asterisk)
	})

	// Create and configure task
	task := pipeline.NewPipelineTask(pipe)

	// Setup event handlers
	task.OnStarted(func() {
		fmt.Println("✓ Pipeline started successfully")
		fmt.Println("✓ Asterisk WebSocket listening on ws://localhost:8080/asterisk")
		fmt.Printf("✓ Codec: %s with passthrough (zero conversions)\n", asteriskCodec)
		fmt.Println("✓ SileroVAD enabled for voice activity detection")
		fmt.Println("\nVAD will emit:")
		fmt.Println("  • UserStartedSpeakingFrame when speech detected")
		fmt.Println("  • UserStoppedSpeakingFrame when speech ends")
		fmt.Println("\nConfigure your Asterisk dialplan:")
		fmt.Println("  exten => _X.,1,Answer()")
		fmt.Println("  same => n,ExternalMedia(ws://YOUR_SERVER:8080/asterisk,c(ulaw))")
		fmt.Println("\nPress Ctrl+C to stop")
	})

	task.OnError(func(err error) {
		log.Printf("Pipeline error: %v", err)
	})

	task.OnFinished(func() {
		fmt.Println("\n✓ Pipeline stopped gracefully")
		// Cleanup VAD resources
		if err := sileroAnalyzer.Cleanup(); err != nil {
			log.Printf("VAD cleanup error: %v", err)
		}
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
