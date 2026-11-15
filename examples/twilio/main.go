package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/pipeline/observers"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

const systemPrompt = `You are a friendly voice assistant on a phone call.
Answer in one or two short sentences. Never use lists, markdown or emojis.`

type bot struct {
	deepgramKey string
	openaiKey   string
	cartesiaKey string
	voiceID     string
	vadSocket   string
}

func main() {
	logger.Init()
	b := bot{
		deepgramKey: mustEnv("DEEPGRAM_API_KEY"),
		openaiKey:   mustEnv("OPENAI_API_KEY"),
		cartesiaKey: mustEnv("CARTESIA_API_KEY"),
		voiceID:     envOr("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
		vadSocket:   os.Getenv("ONNX_WORKER_SOCK"),
	}
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: envOr("ADDR", ":8080"),
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: b.runCall,
		Params:  transports.TransportParams{SessionTimeout: 10 * time.Minute},
	})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	log.Printf("point your TwiML <Stream url=\"wss://YOUR_HOST/media\"/> at this server")
	if err := srv.ListenAndServe(ctx); err != nil {
		log.Fatal(err)
	}
}

func (b bot) runCall(ctx context.Context, s *transports.Session) error {
	info := s.CallInfo()
	log.Printf("call %s started (stream %s)", info.CallID, info.StreamID)

	procs := []processors.FrameProcessor{s.Input()}
	if b.vadSocket != "" {
		analyzer, err := vad.NewSileroVADAnalyzer(16000, vad.DefaultVADParams(), b.vadSocket)
		if err != nil {
			return err
		}
		defer analyzer.Cleanup()
		procs = append(procs, vad.NewVADInputProcessor(analyzer))
	}

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: b.deepgramKey, Model: "nova-3", Language: "en", SmartFormat: true})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: b.openaiKey, Model: "gpt-4.1-mini", SystemPrompt: systemPrompt})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: b.cartesiaKey, VoiceID: b.voiceID, Model: "sonic-2", Language: "en"})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})

	procs = append(procs, stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant())

	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	cfg.EnableUsageMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipeline.NewPipeline(procs), cfg)

	latency := observers.NewUserBotLatencyObserver()
	latency.OnLatencyMeasured(func(d time.Duration) {
		log.Printf("call %s: user stopped speaking -> bot audio in %v", info.CallID, d.Round(time.Millisecond))
	})
	turnsObs := observers.NewTurnTrackingObserver()
	turnsObs.OnTurnEnded(func(turn int, d time.Duration, interrupted bool) {
		log.Printf("call %s: turn %d took %v (interrupted=%v)", info.CallID, turn, d.Round(time.Millisecond), interrupted)
	})
	obs := pipeline.NewTaskObserver()
	obs.AddObserver(observers.NewMetricsLogObserver())
	obs.AddObserver(latency)
	obs.AddObserver(turnsObs)
	task.SetObserver(obs)

	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hi there! How can I help you today?")); err != nil {
		return err
	}
	err := task.Run(ctx)
	log.Printf("call %s ended after %d messages", info.CallID, conversation.Len())
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
