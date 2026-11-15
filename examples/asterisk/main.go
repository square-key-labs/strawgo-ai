package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/turn"
	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/observers"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/transports"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

type config struct {
	deepgramKey     string
	geminiKey       string
	cartesiaKey     string
	cartesiaVoice   string
	systemPrompt    string
	codec           string
	onnxSocket      string
	smartTurn       bool
	minWords        int
	speechTimeout   time.Duration
	allowInterrupts bool
}

func loadConfig() config {
	return config{
		deepgramKey:     mustEnv("DEEPGRAM_API_KEY"),
		geminiKey:       mustEnv("GEMINI_API_KEY"),
		cartesiaKey:     mustEnv("CARTESIA_API_KEY"),
		cartesiaVoice:   envOr("CARTESIA_VOICE_ID", "faf0731e-dfb9-4cfc-8119-259a79b27e12"),
		systemPrompt:    envOr("SYSTEM_PROMPT", "You are a helpful phone assistant. Keep your responses brief and conversational."),
		codec:           os.Getenv("CODEC"),
		onnxSocket:      envOr("ONNX_WORKER_SOCK", fmt.Sprintf("/tmp/onnx-worker-%d.sock", os.Getpid())),
		smartTurn:       envBool("ENABLE_SMART_TURN", false),
		minWords:        envInt("INTERRUPTION_MIN_WORDS", 2),
		speechTimeout:   time.Duration(envInt("SPEECH_TIMEOUT_MS", 800)) * time.Millisecond,
		allowInterrupts: envBool("ALLOW_INTERRUPTIONS", true),
	}
}

func main() {
	logger.Init()
	cfg := loadConfig()
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: envOr("ADDR", ":6969"),
		Path: envOr("WEBSOCKET_PATH", "/ws"),
		NewSerializer: func(r *http.Request) serializers.FrameSerializer {
			return serializers.NewAsteriskFrameSerializer(serializers.AsteriskSerializerConfig{
				ChannelID: r.URL.Query().Get("channel"),
				Codec:     cfg.codec,
			})
		},
		Handler: cfg.runCall,
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := srv.ListenAndServe(ctx); err != nil {
		log.Fatal(err)
	}
}

func (cfg config) runCall(ctx context.Context, s *transports.Session) error {
	info := s.CallInfo()
	log.Printf("channel %s connected (%s @ %d Hz)", info.CallID, info.Encoding, info.SampleRate)

	analyzer, err := vad.NewSileroVADAnalyzer(16000, vad.VADParams{Confidence: 0.7, StartSecs: 0.2, StopSecs: 0.2, MinVolume: 0.1}, cfg.onnxSocket)
	if err != nil {
		return err
	}
	defer analyzer.Cleanup()
	vadProc := vad.NewVADInputProcessor(analyzer)
	if cfg.smartTurn {
		smart, err := turn.NewOnnxSmartTurn(turn.OnnxSmartTurnConfig{SockPath: cfg.onnxSocket, Params: turn.DefaultSmartTurnParams()})
		if err != nil {
			return err
		}
		defer smart.Close()
		vadProc = vad.NewVADInputProcessorWithTurn(analyzer, smart)
	}

	strategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(cfg.allowInterrupts),
			user_start.NewMinWordsUserTurnStartStrategy(cfg.minWords, cfg.allowInterrupts),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(cfg.speechTimeout),
		},
		MuteStrategies: []user_mute.UserMuteStrategy{
			user_mute.NewFunctionCallUserMuteStrategy(),
		},
	}

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: cfg.deepgramKey, Model: "nova-3", Language: "multi"})
	llmSvc := gemini.NewLLMService(gemini.LLMConfig{APIKey: cfg.geminiKey, Model: "gemini-2.5-flash-lite", SystemPrompt: cfg.systemPrompt, Temperature: 0.7})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: cfg.cartesiaKey, VoiceID: cfg.cartesiaVoice, Model: "sonic-3", Language: "en"})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{
		User: aggregators.UserAggregatorParams{Strategies: strategies},
	})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), vadProc, stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	taskCfg := pipeline.DefaultPipelineTaskConfig()
	taskCfg.AllowInterruptions = cfg.allowInterrupts
	taskCfg.EnableMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, taskCfg)

	obs := pipeline.NewTaskObserver()
	obs.AddObserver(observers.NewMetricsLogObserver())
	obs.AddObserver(observers.NewTranscriptionLogObserver())
	task.SetObserver(obs)

	err = task.Run(ctx)
	log.Printf("channel %s finished with %d context messages", info.CallID, conversation.Len())
	return err
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s is required", key)
	}
	return v
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if n, err := strconv.Atoi(os.Getenv(key)); err == nil {
		return n
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	if b, err := strconv.ParseBool(os.Getenv(key)); err == nil {
		return b
	}
	return fallback
}
