# StrawGo

StrawGo is a Go library for real-time voice and text agents. An application is a pipeline of frame processors (transport input, VAD, STT, context aggregators, LLM, TTS, transport output) driven by a `PipelineTask`. Telephony providers are supported through WebSocket media-stream serializers; browsers through a WebRTC handler.

Module: `github.com/square-key-labs/strawgo-ai` (Go 1.25 or newer, see `go.mod`).

Status: the engine, service bases, transports and most providers were rebuilt after v0.0.12 and have not been released. Nothing in this rebuild has been run against live vendor endpoints; see the capability table for what each provider has actually been tested with.

## Quickstart

A Twilio Media Streams bot: Deepgram STT, OpenAI LLM, Cartesia TTS. This is `examples/twilio/main.go` with the optional Silero VAD and the latency observers removed.

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
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

func main() {
	logger.Init()
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: ":8080",
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: runCall,
		Params:  transports.TransportParams{SessionTimeout: 10 * time.Minute},
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

func runCall(ctx context.Context, s *transports.Session) error {
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en", SmartFormat: true})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: systemPrompt})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: os.Getenv("CARTESIA_API_KEY"), VoiceID: os.Getenv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	})
	cfg := pipeline.DefaultPipelineTaskConfig()
	cfg.EnableMetrics = true
	task := pipeline.NewPipelineTaskWithConfig(pipe, cfg)

	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hi there! How can I help you today?")); err != nil {
		return err
	}
	return task.Run(ctx)
}
```

Point a TwiML `<Stream url="wss://YOUR_HOST/media"/>` at the server. `transports.NewServer` accepts one WebSocket per call and runs the handler with a fresh `*transports.Session`; each call builds its own pipeline.

Runnable examples:

| Path | What it shows |
| --- | --- |
| [examples/twilio](examples/twilio/main.go) | Twilio bot, optional Silero VAD, latency and turn observers |
| [examples/function_calling](examples/function_calling/main.go) | Tool calls with `RegisterFunction`, ending the call from a function |
| [examples/asterisk](examples/asterisk/main.go) | Asterisk WebSocket channel, explicit turn strategies, optional Smart Turn |
| [examples/gemini_live](examples/gemini_live/main.go) | Gemini Live speech-to-speech with a tool |
| [examples/text_flow.go](examples/text_flow.go) | Text-only console chat (`go run examples/text_flow.go`) |

## Documentation

- [Architecture](docs/architecture.md): frames, processor queues, interruptions, task lifecycle, context ownership, function calling, metrics and tracing.
- [Writing a provider](docs/writing-a-provider.md): LLM, STT and TTS provider interfaces.
- [Transports](docs/transports.md): WebSocket server and sessions, serializers, WebRTC, RTVI.
- [Flows](docs/flows.md): node-based conversation flows.
- [Migrating from v0.0.12](docs/migration-v0.0.12.md): breaking API changes.
- [CHANGELOG](CHANGELOG.md), [glossary](CONTEXT.md), [ADRs](docs/adr/).

## Capabilities

Status values:

- **mock-tested only**: unit tests in the package run against fakes or local test servers. Never run against the live vendor API.
- **untested**: compiles; the package has no tests.
- **not available**: absent, or present but not usable.

### LLM

All LLM services are `*llmservice.Service` built from an `llm.Provider`.

| Provider | Package | Status |
| --- | --- | --- |
| OpenAI Chat Completions | `services/openai` (`NewLLMService`) | untested |
| OpenAI Responses API | `services/openai` (`NewResponsesLLMService`) | untested |
| Azure OpenAI | `services/azureopenai` | untested |
| Groq | `services/groq` | mock-tested only |
| Ollama | `services/ollama` | mock-tested only |
| Together, Cerebras, DeepSeek, Fireworks, Mistral, NVIDIA, OpenRouter, Perplexity, Qwen, SambaNova, xAI | one package each under `services/` (OpenAI-compatible, via `services/openaicompat`) | untested |
| Anthropic | `services/anthropic` | mock-tested only |
| Google Gemini | `services/gemini` (`NewLLMService`) | mock-tested only |
| Google Vertex AI (Gemini) | `services/vertex` | mock-tested only |
| AWS Bedrock (Converse) | `services/aws/bedrock` | untested |
| Sarvam | `services/sarvam` (`NewLLMService`, OpenAI-compatible) | untested |
| MCP tools (stdio, SSE, streamable HTTP) | `services/mcp` | untested |

### STT

| Provider | Package | Mode | Status |
| --- | --- | --- | --- |
| Deepgram | `services/deepgram` (`NewSTTService`) | streaming | mock-tested only |
| Deepgram Flux | `services/deepgram` (`NewFluxSTTService`) | streaming | mock-tested only (package tests do not cover Flux specifically) |
| AssemblyAI | `services/assemblyai` | streaming | mock-tested only |
| Azure Speech | `services/azure` | streaming | mock-tested only |
| OpenAI Realtime transcription | `services/openai_realtime` (`NewSTTService`) | streaming | mock-tested only |
| OpenAI transcription | `services/openai` (`NewSTTService`, `NewRealtimeSTTService`) | segmented / streaming | untested |
| Whisper (OpenAI API) | `services/whisper` | segmented | mock-tested only |
| Groq Whisper | `services/groq` (`NewSTTService`) | segmented | untested |
| ElevenLabs | `services/elevenlabs` (`NewSTTService`, `NewRealtimeSTTService`) | segmented / streaming | untested |
| Cartesia | `services/cartesia` (`NewSTTService`) | streaming | untested |
| Google Cloud Speech | `services/google` (`NewSTTService`) | segmented | untested |
| AWS Transcribe | `services/aws/transcribe` | streaming | untested |
| Gladia, Soniox, Sarvam | `services/gladia`, `services/soniox`, `services/sarvam` | streaming | untested |
| fal | `services/fal` | segmented | untested |
| Speechmatics | | | not available |

### TTS

| Provider | Package | Mode | Status |
| --- | --- | --- | --- |
| Cartesia | `services/cartesia` (`NewTTSService` / `NewHTTPTTSService`) | WebSocket / HTTP | WebSocket mock-tested only, HTTP untested |
| ElevenLabs | `services/elevenlabs` (`NewTTSService` / `NewHTTPTTSService`) | WebSocket / HTTP | `NewTTSService` mock-tested only, `NewHTTPTTSService` untested |
| Deepgram | `services/deepgram` (`NewTTSService` / `NewHTTPTTSService`) | WebSocket / HTTP | WebSocket mock-tested only, HTTP untested |
| Azure Speech | `services/azure` | HTTP | mock-tested only |
| Google Cloud TTS | `services/google` | HTTP | mock-tested only |
| OpenAI | `services/openai` (`NewTTSService`) | HTTP | untested |
| Groq | `services/groq` (`NewTTSService`) | HTTP | untested |
| AWS Polly | `services/aws/polly` | HTTP | untested |
| Rime, Neuphonic, Inworld, Sarvam | `services/rime`, `services/neuphonic`, `services/inworld`, `services/sarvam` | WebSocket and HTTP variants | untested |
| Async | `services/asyncai` (`NewHTTPTTSService`) | HTTP | untested |
| Fish Audio, Resemble, Speechify | `services/fish`, `services/resemble`, `services/speechify` | WebSocket | untested |
| MiniMax | `services/minimax` | HTTP | untested |
| Piper, XTTS (self-hosted servers) | `services/piper`, `services/xtts` | HTTP | untested |

### Speech-to-speech

| Provider | Package | Status |
| --- | --- | --- |
| OpenAI Realtime | `services/openai_realtime` (`NewLLMService`) | mock-tested only |
| Gemini Live | `services/gemini` (`NewLiveService`) | mock-tested only |
| AWS Nova Sonic | | not available |

### Transports

| Transport | Package | Status |
| --- | --- | --- |
| WebSocket server, one `Session` per connection | `transports` (`NewServer`) | tested in-repo with local WebSocket connections |
| WebSocket client | `transports` (`Dial`) | tested in-repo with local WebSocket connections |
| WebRTC (Pion), SmallWebRTC-style offer/answer over HTTP | `transports/webrtc` | untested; G.711 (PCMU/PCMA) only by default |
| WebRTC Opus codec | `transports/webrtc`, build tag `opus` | not available: never compiled, needs libopus |
| Daily | | not available |
| LiveKit | | not available |
| Video (any transport) | | not available |

### Telephony and client serializers

| Serializer | Status |
| --- | --- |
| Twilio Media Streams (playback ack via `mark`, REST hangup) | playback-ack mock-tested only; hangup untested |
| Asterisk WebSocket channel (playback ack) | playback-ack mock-tested only |
| Telnyx, Plivo, Exotel, Vonage, Genesys AudioHook, Jambonz | untested |
| Protobuf (Pipecat client wire format) | untested |
| RTVI client JSON (`RTVIClientSerializer`) | untested |

### Audio

| Component | Package | Status |
| --- | --- | --- |
| Streaming polyphase resampler | `audio/resample` | untested |
| G.711 mu-law / A-law | `audio/g711` | untested |
| PCM helpers, WAV encode | `audio/pcm`, `audio/wav` | untested |
| DTMF tone generation | `audio/dtmf` | untested |
| Input filters (noise gate, chains) | `audio/filter` | untested |
| Output mixer (sound files) | `audio/mixer` | untested |
| Silero VAD (via the Rust `onnx-worker` over a Unix socket) | `audio/vad` | tested against a fake worker socket only |
| Recording, per-turn audio | `processors/audiobuffer` | untested |

### Turn-taking

| Component | Package | Status |
| --- | --- | --- |
| Start strategies: VAD, transcription, min-words | `turns/user_start` | tested in-repo |
| Stop strategies: speech timeout, turn analyzer | `turns/user_stop` | tested in-repo |
| Mute strategies: first speech, mute until first bot reply completes, function call, always | `turns/user_mute` | tested in-repo |
| Smart Turn (ONNX via `onnx-worker`, HTTP, fal) | `audio/turn` | ONNX client tested against a fake worker socket only; HTTP and fal untested |
| User idle detection | `processors/aggregators` | tested in-repo |
| DTMF aggregation | `processors/aggregators` (`NewDTMFAggregator`) | untested |

### Flows

| Component | Package | Status |
| --- | --- | --- |
| Node-based flows (static Go nodes and JSON flow configs), Mermaid export | `flows` | untested |

### Extensions

| Component | Package | Status |
| --- | --- | --- |
| IVR navigator | `extensions/ivr` | untested |
| Voicemail detector | `extensions/voicemail` | untested |

### Observability

| Component | Package | Status |
| --- | --- | --- |
| Typed metrics frames (TTFB, processing, LLM usage, TTS characters, STT seconds) | `frames` | untested as a unit; exercised by observer tests |
| Observers: user-to-bot latency, turn tracking, turn metrics, startup timing, metrics/transcription/LLM/debug loggers | `pipeline/observers` | latency, turn metrics and startup timing tested in-repo; loggers and turn tracking untested |
| OpenTelemetry tracing (conversation, turn and service spans) | `tracing` | untested |
| RTVI server protocol (processor and observer) | `processors/rtvi` | untested |
| Transcript processor | `processors/transcript` | untested |

## Performance

No performance figures are claimed here. Measure on your own hardware:

```bash
go test -run '^$' -bench . -benchmem ./src/
```

This runs the frame-throughput, context and Twilio serializer benchmarks in `src/benchmark_test.go`. `cmd/loadtest` load-tests the ONNX worker (`go run ./cmd/loadtest -h` for flags); it needs a running `onnx-worker`.

## Development

```bash
go build ./...
go vet ./...
go test -race ./...
```

Silero VAD and the ONNX Smart Turn analyzer talk to the Rust `onnx-worker` (in [onnx-worker](onnx-worker/)) over a Unix socket; see `src/worker` for the supervisor and `scripts/install_onnxruntime.sh`.

## License

MIT, see [LICENSE](LICENSE).
