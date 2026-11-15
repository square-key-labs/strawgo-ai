# Strawgo

Strawgo is a Go library for real-time voice and text agents: programs that listen to a caller, understand them, and answer out loud. An application is a pipeline of frame processors (transport input, VAD, STT, context aggregators, LLM, TTS, transport output) driven by a `PipelineTask`. Telephony providers connect through WebSocket media-stream serializers; browsers through a WebRTC handler.

Module: `github.com/square-key-labs/strawgo-ai` (Go 1.25.9 or newer, see `go.mod`).

Status: the engine, service bases, transports and most providers were rebuilt after v0.0.12, and the rebuild is on `main` only. The tagged releases `v0.1.0` to `v0.1.3` were cut from the older code (their `VERSION` file says `0.0.12`) and have the old API, so `go get` without a version gives you that older API. Install from `main` until a newer release is tagged. Nothing in this rebuild has been run against live vendor endpoints; see [Capabilities](#capabilities) for what each part has been tested with.

## Quickstart

Run the text chat example (needs only an OpenAI key, or any OpenAI-compatible server via `OPENAI_BASE_URL`):

```sh
git clone https://github.com/square-key-labs/strawgo-ai && cd strawgo-ai
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
go run examples/text_flow.go
```

A phone bot in your own module: Twilio Media Streams in, Deepgram STT, OpenAI LLM, Cartesia TTS out. Save it as `main.go`, then run `go get github.com/square-key-labs/strawgo-ai@main && go mod tidy` in the module folder (after the file exists, so that `go mod tidy` keeps the requirement).

```go
package main

import (
	"context"
	"log"
	"net/http"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

func main() {
	srv, err := transports.NewServer(transports.ServerConfig{
		Addr: ":8080",
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{})
		},
		Handler: runCall,
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Fatal(srv.ListenAndServe(context.Background()))
}

func runCall(ctx context.Context, s *transports.Session) error {
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a friendly phone assistant. Answer in one short sentence."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: os.Getenv("CARTESIA_API_KEY"), VoiceID: os.Getenv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})

	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})
	task := pipeline.NewPipelineTask(pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant(),
	}))
	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Hi there! How can I help you today?")); err != nil {
		return err
	}
	return task.Run(ctx)
}
```

Point a TwiML `<Connect><Stream url="wss://YOUR_HOST/media"/></Connect>` at the server. Each call gets its own `*transports.Session` and its own pipeline. [Your first phone bot](docs/getting-started/first-phone-bot.md) walks through this program, the tunnel and the Twilio setup step by step.

Runnable examples:

| Path | What it shows |
| --- | --- |
| [examples/twilio](examples/twilio/main.go) | Twilio bot, optional Silero VAD, latency and turn observers |
| [examples/function_calling](examples/function_calling/main.go) | Tool calls with `RegisterFunction`, ending the call from a function |
| [examples/asterisk](examples/asterisk/main.go) | Asterisk WebSocket channel, explicit turn strategies, optional Smart Turn |
| [examples/gemini_live](examples/gemini_live/main.go) | Gemini Live speech-to-speech with a tool |
| [examples/text_flow.go](examples/text_flow.go) | Text-only console chat (`go run examples/text_flow.go`) |

## Documentation

The full documentation is in [docs/](docs/README.md).

- Getting started: [Introduction](docs/getting-started/introduction.md), [Installation](docs/getting-started/installation.md), [Quickstart](docs/getting-started/quickstart.md), [First phone bot](docs/getting-started/first-phone-bot.md), [Core concepts](docs/getting-started/core-concepts.md), [Glossary](docs/getting-started/glossary.md)
- Guides: [Function calling](docs/guides/function-calling.md), [Turn-taking](docs/guides/turn-taking.md), [Flows](docs/guides/flows.md), [Transports overview](docs/guides/transports-overview.md), [WebRTC](docs/guides/webrtc.md), [Deployment](docs/guides/deployment.md), [Troubleshooting](docs/guides/troubleshooting.md), and [more](docs/README.md#guides)
- Reference: [Frames](docs/reference/frames.md), [Pipeline](docs/reference/pipeline.md), [Processors](docs/reference/processors.md), and [more](docs/README.md#reference)
- Providers: [Provider matrix](docs/providers/README.md), [Writing a custom provider](docs/providers/custom.md)
- [Cookbook](docs/cookbook/README.md): complete programs for common bots
- Project: [Architecture](docs/architecture.md), [Migrating from v0.0.12](docs/migration-v0.0.12.md), [CHANGELOG](CHANGELOG.md), [domain glossary](CONTEXT.md), [ADRs](docs/adr/)

## Capabilities

The [provider and component status table](docs/providers/README.md) is the single source of truth for what exists and how each part is tested. Status values there: **mock-tested** (tests against a local mock server or fake peer), **unit-tested** (tests without a network mock), **untested** (no tests), **not available**. Nothing has been verified against a live vendor API.

| Category | What exists | Status summary | Details |
| --- | --- | --- | --- |
| LLM | OpenAI (Chat Completions and Responses), Anthropic, Gemini, Vertex AI, AWS Bedrock, Groq, Ollama, Azure OpenAI and many OpenAI-compatible vendors | mixed: some mock-tested, Vertex and Bedrock unit-tested, most OpenAI-compatible vendors untested | [LLM providers](docs/providers/README.md#llm) |
| STT | Deepgram (incl. Flux), AssemblyAI, Azure, OpenAI, Whisper, Groq, ElevenLabs, Cartesia, Google, AWS Transcribe, Gladia, Soniox, Sarvam, fal | mixed; Speechmatics not available | [STT providers](docs/providers/README.md#stt) |
| TTS | Cartesia, ElevenLabs, Deepgram, Azure, Google, OpenAI, Groq, AWS Polly, Rime, Neuphonic, Inworld, Sarvam, Async, Fish, Resemble, Speechify, MiniMax, Piper, XTTS | mixed: a few mock-tested, most untested | [TTS providers](docs/providers/README.md#tts) |
| Speech-to-speech | OpenAI Realtime, Gemini Live | mock-tested; AWS Nova Sonic not available | [Speech-to-speech](docs/providers/README.md#speech-to-speech) |
| Transports | WebSocket server and client, WebRTC (G.711) | mock-tested; Opus, Daily, LiveKit and video not available | [Transports overview](docs/guides/transports-overview.md) |
| Telephony serializers | Twilio, Telnyx, Plivo, Exotel, Vonage, Asterisk, Genesys AudioHook, Jambonz, Protobuf, RTVI | unit-tested; REST hang-up mock-tested | [Telephony providers](docs/guides/telephony-providers.md) |
| Audio and turn-taking | resampler, G.711, mixer, filters, DTMF, Silero VAD, Smart Turn, turn strategies | unit-tested or mock-tested; DTMF aggregator untested | [Turn-taking](docs/guides/turn-taking.md), [Audio](docs/guides/audio.md) |
| Flows, extensions, MCP | node-based flows, IVR navigator, voicemail detector, MCP client | flows unit-tested; IVR, voicemail and MCP untested | [Flows](docs/guides/flows.md) |
| Observability | metrics frames, observers, OpenTelemetry tracing, RTVI | some observers and RTVI unit-tested; loggers, turn tracking and tracing untested | [Observability](docs/guides/observability.md) |

The [framework components table](docs/providers/README.md#framework-components) gives the per-package status.

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
