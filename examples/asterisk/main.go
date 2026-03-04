package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	// "github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type Config struct {
	DeepgramAPIKey  string
	GeminiAPIKey    string
	CartesiaAPIKey  string
	CartesiaVoiceID string
	// ElevenLabsAPIKey             string
	// ElevenLabsVoiceID            string
	SystemPrompt         string
	WebSocketPort        int
	WebSocketPath        string
	AllowInterruptions   bool
	InterruptionMinWords int
	Codec                string
	EnableVAD            bool
	VADConfidence        float32
	VADStartSecs         float32
	VADStopSecs          float32
	VADMinVolume         float32
	// Smart Turn configuration
	EnableSmartTurn     bool
	SmartTurnMode       string  // "local" for ONNX, "fal" for Fal.ai hosted, "http" for custom HTTP endpoint
	SmartTurnAPIKey     string  // Fal.ai API key for hosted Smart Turn
	SmartTurnModelPath  string  // Path to ONNX model file for local mode
	SmartTurnStopSecs   float64 // Silence timeout before auto-completing turn (default: 3.0)
	SmartTurnCPUThreads int     // Number of CPU threads for local ONNX inference (default: 1)
}

func loadConfig() *Config {
	return &Config{
		DeepgramAPIKey:  getEnv("DEEPGRAM_API_KEY", ""),
		GeminiAPIKey:    getEnv("GEMINI_API_KEY", ""),
		CartesiaAPIKey:  getEnv("CARTESIA_API_KEY", ""),
		CartesiaVoiceID: getEnv("CARTESIA_VOICE_ID", "faf0731e-dfb9-4cfc-8119-259a79b27e12"), // Barbershop Man
		// ElevenLabsAPIKey:             getEnv("ELEVENLABS_API_KEY", ""),
		// ElevenLabsVoiceID:            getEnv("ELEVENLABS_VOICE_ID", "ZUrEGyu8GFMwnHbvLhv2"),
		SystemPrompt:         getEnv("SYSTEM_PROMPT", "You are a helpful AI assistant. Keep your responses brief and conversational."),
		WebSocketPort:        getEnvInt("WEBSOCKET_PORT", 6969),
		WebSocketPath:        getEnv("WEBSOCKET_PATH", "/ws"),
		AllowInterruptions:   getEnvBool("ALLOW_INTERRUPTIONS", true),
		InterruptionMinWords: getEnvInt("INTERRUPTION_MIN_WORDS", 2),
		Codec:                getEnv("CODEC", "alaw"),
		EnableVAD:            getEnvBool("ENABLE_VAD", true),
		VADConfidence:        getEnvFloat32("VAD_CONFIDENCE", 0.7),
		VADStartSecs:         getEnvFloat32("VAD_START_SECS", 0.2),
		VADStopSecs:          getEnvFloat32("VAD_STOP_SECS", 0.8),  // Allow natural pauses without splitting utterances
		VADMinVolume:         getEnvFloat32("VAD_MIN_VOLUME", 0.1), // Lower default catches most speech
		// Smart Turn - ML-based end-of-turn detection
		EnableSmartTurn:     getEnvBool("ENABLE_SMART_TURN", false),
		SmartTurnMode:       getEnv("SMART_TURN_MODE", "local"),         // "local", "fal", or "http"
		SmartTurnAPIKey:     getEnv("SMART_TURN_API_KEY", ""),           // Fal.ai API key (for fal/http modes)
		SmartTurnModelPath:  getEnv("SMART_TURN_MODEL_PATH", ""),        // ONNX model path (for local mode)
		SmartTurnStopSecs:   getEnvFloat64("SMART_TURN_STOP_SECS", 3.0), // Silence timeout
		SmartTurnCPUThreads: getEnvInt("SMART_TURN_CPU_THREADS", 1),     // CPU threads for local ONNX
	}
}

func getEnvFloat64(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if result, err := strconv.ParseFloat(value, 64); err == nil {
			return result
		}
	}
	return defaultValue
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var result int
		if _, err := fmt.Sscanf(value, "%d", &result); err == nil {
			return result
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if result, err := strconv.ParseBool(value); err == nil {
			return result
		}
	}
	return defaultValue
}

func getEnvFloat32(key string, defaultValue float32) float32 {
	if value := os.Getenv(key); value != "" {
		if result, err := strconv.ParseFloat(value, 32); err == nil {
			return float32(result)
		}
	}
	return defaultValue
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	config := loadConfig()

	if err := validateConfig(config); err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	asteriskSerializer := serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
		Codec:      config.Codec,
		SampleRate: 8000,
	})

	transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
		Port:       config.WebSocketPort,
		Path:       config.WebSocketPath,
		Serializer: asteriskSerializer,
	})

	deepgramSTT := deepgram.NewSTTService(deepgram.STTConfig{
		APIKey:   config.DeepgramAPIKey,
		Language: "multi",
		Model:    "nova-3",
		Encoding: config.Codec,
	})

	llmContext := services.NewLLMContext(config.SystemPrompt)
	turnStrategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(true),
			user_start.NewMinWordsUserTurnStartStrategy(config.InterruptionMinWords, true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(time.Duration(config.VADStopSecs*1000)*time.Millisecond, true),
		},
		MuteStrategies: []user_mute.UserMuteStrategy{
			user_mute.NewFirstSpeechUserMuteStrategy(true),
			user_mute.NewFunctionCallUserMuteStrategy(true),
		},
	}

	userAgg := aggregators.NewLLMUserAggregator(llmContext, turnStrategies)
	assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, aggregators.DefaultAssistantAggregatorParams())

	geminiLLM := gemini.NewLLMService(gemini.LLMConfig{
		APIKey:      config.GeminiAPIKey,
		Model:       "gemini-2.5-flash-lite",
		Temperature: 0.7,
	})

	// Configure TTS encoding based on codec
	var ttsEncoding string
	var ttsSampleRate int
	switch config.Codec {
	case "alaw":
		ttsEncoding = "pcm_alaw"
		ttsSampleRate = 8000
	case "ulaw":
		ttsEncoding = "pcm_mulaw"
		ttsSampleRate = 8000
	case "linear16":
		ttsEncoding = "pcm_s16le"
		ttsSampleRate = 16000
	default:
		ttsEncoding = "pcm_alaw"
		ttsSampleRate = 8000
	}

	// Cartesia TTS (Sonic-3)
	cartesiaTTS := cartesia.NewTTSService(cartesia.TTSConfig{
		APIKey:             config.CartesiaAPIKey,
		VoiceID:            config.CartesiaVoiceID,
		Model:              "sonic-3",
		CartesiaVersion:    "2025-04-16",
		Language:           "en",
		SampleRate:         ttsSampleRate,
		Encoding:           ttsEncoding,
		Container:          "raw",
		AggregateSentences: true,
	})

	// ElevenLabs TTS (disabled)
	// ttsFormat := "ulaw_8000"
	// if config.Codec == "alaw" {
	// 	ttsFormat = "alaw_8000"
	// }
	// elevenLabsTTS := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
	// 	APIKey:       config.ElevenLabsAPIKey,
	// 	VoiceID:      config.ElevenLabsVoiceID,
	// 	Model:        "eleven_turbo_v2_5",
	// 	OutputFormat: ttsFormat,
	// 	UseStreaming: true,
	// })

	// Initialize VAD if enabled
	var vadProcessor *vad.VADInputProcessor
	var vadAnalyzer *vad.SileroVADAnalyzer
	if config.EnableVAD {
		vadParams := vad.VADParams{
			Confidence: config.VADConfidence,
			StartSecs:  config.VADStartSecs,
			StopSecs:   config.VADStopSecs,
			MinVolume:  config.VADMinVolume,
		}

		var err error
		vadAnalyzer, err = vad.NewSileroVADAnalyzer(8000, vadParams)
		if err != nil {
			log.Fatalf("Failed to create SileroVAD analyzer: %v", err)
		}

		// Initialize Smart Turn if enabled
		var turnAnalyzer turn.TurnAnalyzer
		if config.EnableSmartTurn {
			smartTurnParams := &turn.SmartTurnParams{
				StopSecs:        config.SmartTurnStopSecs,
				PreSpeechMs:     0,
				MaxDurationSecs: 8.0,
			}

			switch config.SmartTurnMode {
			case "local":
				// Use local ONNX inference (recommended - fast, no network required)
				localTurn, err := turn.NewLocalSmartTurn(turn.LocalSmartTurnConfig{
					ModelPath: config.SmartTurnModelPath,
					CPUCount:  config.SmartTurnCPUThreads,
					Params:    smartTurnParams,
				})
				if err != nil {
					log.Fatalf("Failed to initialize Local Smart Turn: %v", err)
				}
				turnAnalyzer = localTurn
				log.Printf("Smart Turn initialized (local ONNX, cpu_threads=%d, stop_secs=%.1f)",
					config.SmartTurnCPUThreads, config.SmartTurnStopSecs)

			case "fal":
				// Use Fal.ai hosted Smart Turn
				if config.SmartTurnAPIKey == "" {
					log.Fatalf("SMART_TURN_API_KEY is required for fal mode")
				}
				turnAnalyzer = turn.NewFalSmartTurn(turn.FalSmartTurnConfig{
					APIKey: config.SmartTurnAPIKey,
					Params: smartTurnParams,
				})
				log.Printf("Smart Turn initialized (Fal.ai hosted, stop_secs=%.1f)", config.SmartTurnStopSecs)

			case "http":
				// Use HTTP Smart Turn with custom URL
				turnAnalyzer = turn.NewHTTPSmartTurn(turn.HTTPSmartTurnConfig{
					APIKey: config.SmartTurnAPIKey,
					Params: smartTurnParams,
				})
				log.Printf("Smart Turn initialized (HTTP, stop_secs=%.1f)", config.SmartTurnStopSecs)

			default:
				log.Fatalf("Invalid SMART_TURN_MODE: %s (must be 'local', 'fal', or 'http')", config.SmartTurnMode)
			}
		}

		// Create VAD processor with optional Smart Turn
		if turnAnalyzer != nil {
			vadProcessor = vad.NewVADInputProcessorWithTurn(vadAnalyzer, turnAnalyzer)
		} else {
			vadProcessor = vad.NewVADInputProcessor(vadAnalyzer)
		}

		log.Printf("SileroVAD initialized (confidence=%.2f, start=%.2fs, stop=%.2fs, minVolume=%.2f)",
			vadParams.Confidence, vadParams.StartSecs, vadParams.StopSecs, vadParams.MinVolume)
	}

	// Build pipeline with optional VAD
	// SentenceAggregator buffers LLM tokens and emits complete sentences
	// This ensures interruptions occur at sentence boundaries
	sentenceAgg := aggregators.NewSentenceAggregator()

	processors := []processors.FrameProcessor{
		transport.Input(),
	}
	if config.EnableVAD && vadProcessor != nil {
		processors = append(processors, vadProcessor)
	}
	processors = append(processors,
		deepgramSTT,
		userAgg,
		geminiLLM,
		sentenceAgg, // Split LLM output into sentences for clean interruptions
		cartesiaTTS,
		transport.Output(),
		assistantAgg,
	)

	pipe := pipeline.NewPipeline(processors)

	taskConfig := &pipeline.PipelineTaskConfig{
		AllowInterruptions: config.AllowInterruptions,
		TurnStrategies:     turnStrategies,
	}

	task := pipeline.NewPipelineTaskWithConfig(pipe, taskConfig)

	task.OnStarted(func() {
		log.Printf("Pipeline started successfully")
		log.Printf("WebSocket listening on ws://localhost:%d%s", config.WebSocketPort, config.WebSocketPath)
		log.Printf("Using %s codec passthrough (no audio conversions)", config.Codec)
		if config.EnableVAD {
			log.Printf("SileroVAD enabled (confidence=%.2f, start=%.2fs, stop=%.2fs)",
				config.VADConfidence, config.VADStartSecs, config.VADStopSecs)
			if config.EnableSmartTurn {
				log.Printf("Smart Turn enabled (mode=%s, stop_secs=%.1f)", config.SmartTurnMode, config.SmartTurnStopSecs)
			}
		} else {
			log.Printf("VAD disabled")
		}
		if config.AllowInterruptions {
			log.Printf("Interruptions enabled (min words: %d)", config.InterruptionMinWords)
		} else {
			log.Printf("Interruptions disabled")
		}
	})

	task.OnError(func(err error) {
		log.Printf("Pipeline error: %v", err)
	})

	task.OnFinished(func() {
		log.Println("Pipeline stopped gracefully")
		// Cleanup VAD resources if enabled
		if config.EnableVAD && vadAnalyzer != nil {
			if err := vadAnalyzer.Cleanup(); err != nil {
				log.Printf("VAD cleanup error: %v", err)
			} else {
				log.Println("VAD resources cleaned up")
			}
		}
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
	log.Println("Shutting down...")
	cancel()
}

func validateConfig(config *Config) error {
	if config.DeepgramAPIKey == "" {
		return fmt.Errorf("DEEPGRAM_API_KEY is required")
	}
	if config.GeminiAPIKey == "" {
		return fmt.Errorf("GEMINI_API_KEY is required")
	}
	if config.CartesiaAPIKey == "" {
		return fmt.Errorf("CARTESIA_API_KEY is required")
	}
	if config.CartesiaVoiceID == "" {
		return fmt.Errorf("CARTESIA_VOICE_ID is required")
	}
	// ElevenLabs validation (disabled)
	// if config.ElevenLabsAPIKey == "" {
	// 	return fmt.Errorf("ELEVENLABS_API_KEY is required")
	// }
	// if config.ElevenLabsVoiceID == "" {
	// 	return fmt.Errorf("ELEVENLABS_VOICE_ID is required")
	// }
	return nil
}
