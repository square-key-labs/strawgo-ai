# Installation

This page gets Strawgo into a Go module, sets up API keys, and optionally installs the `onnx-worker` sidecar that runs Silero VAD and Smart Turn. At the end you check that everything builds and that a bot answers you.

## Before you begin

- Go 1.25.9 or newer. The module's `go.mod` declares `go 1.25.9`. Go 1.21 and newer download that toolchain automatically the first time you build, unless `GOTOOLCHAIN=local` is set; Go 1.20 and older cannot build the module. Check with `go version`.
- An account and API key for at least one LLM provider. The [quickstart](quickstart.md) uses OpenAI, or any server that speaks the OpenAI Chat Completions API.
- Optional: a Rust toolchain (`cargo`), only if you want to build `onnx-worker` from source.

## Add Strawgo to your module

The module path is `github.com/square-key-labs/strawgo-ai`. Packages live under `src/`, so imports look like `github.com/square-key-labs/strawgo-ai/src/pipeline`.

> [!NOTE]
> The API in these docs is in `v0.2.0` and newer. The tags `v0.1.0` to `v0.1.3` were cut from the older code and have the old API described in [Migrating from v0.0.12](../migration-v0.0.12.md). If your module already requires one of them, the command below upgrades it. Check which tags exist with `git ls-remote --tags https://github.com/square-key-labs/strawgo-ai`.

Create a module:

```sh
mkdir mybot && cd mybot
go mod init example.com/mybot
```

Write your program first, for example the one in the [quickstart](quickstart.md). Then, in the module folder, fetch the latest Strawgo release and add its indirect dependencies (protobuf, Pion, the AWS SDK and others) to `go.sum`, in one command:

```sh
go get github.com/square-key-labs/strawgo-ai@latest && go mod tidy
```

Run them in this order, after the program file exists. `go mod tidy` removes requirements that no code imports yet, so running it before you have a program drops Strawgo again. If you skip `go mod tidy`, `go build` stops with a `missing go.sum entry` error. If the build reports `undefined: transports.NewServer` or a missing Strawgo package, see [Troubleshooting](../guides/troubleshooting.md#undefined-transportsnewserver-or-a-missing-strawgo-package).

## Set your API keys

Strawgo never reads API keys by itself. Every service takes its key as a config field, such as `openai.LLMConfig{APIKey: ...}`. The examples in these docs read keys from environment variables with `os.Getenv`, so you export them in your shell:

```sh
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
export CARTESIA_API_KEY=YOUR_CARTESIA_API_KEY
```

These are the variable names the examples use. You only need the ones for the services your program creates.

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI LLM in the quickstart and the phone bot |
| `OPENAI_BASE_URL` | Quickstart and `examples/text_flow.go`: points the OpenAI service at another OpenAI-compatible server |
| `OPENAI_MODEL` | `examples/text_flow.go` only: the model name (default `gpt-4.1-mini`) |
| `SHOW_METRICS` | `examples/text_flow.go` only: any non-empty value logs metrics |
| `DEEPGRAM_API_KEY` | Deepgram STT |
| `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` | Cartesia TTS |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Twilio serializer, to hang up the call through Twilio's REST API |
| `GROQ_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | `examples/function_calling` |
| `GEMINI_API_KEY` | `examples/gemini_live` |
| `ONNX_WORKER_SOCK` | Path of the `onnx-worker` Unix socket, when you use Silero VAD (the phone bot, `examples/twilio`, `examples/asterisk`) |
| `ADDR` | Listen address of `examples/twilio` and `examples/function_calling` (default `:8080`) and `examples/asterisk` (default `:6969`) |
| `WEBSOCKET_PATH`, `CODEC`, `SYSTEM_PROMPT`, `SPEECH_TIMEOUT_MS`, `INTERRUPTION_MIN_WORDS`, `ALLOW_INTERRUPTIONS`, `ENABLE_SMART_TURN` | `examples/asterisk` only; see the top of its `main.go` for defaults |

The logger is the one part of Strawgo that reads the environment. `logger.Init()` reads `LOG_LEVEL` (`DEBUG`, `INFO`, `WARN`, `ERROR`; default `INFO`) and `LOG_COLOR` (`false` or `0` turns colour off).

## Optional: install the onnx-worker sidecar

Silero VAD (voice activity detection: deciding whether a chunk of audio contains speech) and the ONNX Smart Turn model (deciding whether the caller has finished their turn) do not run inside your Go process. They run in `onnx-worker`, a small Rust program in the [`onnx-worker/`](../../onnx-worker/) folder of the repository. Your Go code talks to it over a Unix socket.

You do not need it to get started. Without it, Strawgo starts a user turn from the STT transcription instead of from VAD. Add it when you want faster, more reliable speech detection on calls.

The worker needs two model files and a socket path. It takes them as flags:

```sh
onnx-worker --vad-model ~/.cache/strawgo/models/silero_vad.onnx --turn-model ~/.cache/strawgo/models/smart-turn-v3.1-cpu.onnx --socket /tmp/onnx-worker.sock
```

Both model flags are required, even if you only use VAD. An optional `--threads N` sets the inference thread count (default: the number of CPUs, between 1 and 4).

You can get the worker and its models in three ways.

### Option 1: let Go start it for you

`worker.Start` in `src/worker` does everything:

1. It finds the binary. With an empty path it looks for `onnx-worker` on your `PATH`, then in `~/.cache/strawgo/bin/onnx-worker`, and otherwise downloads it from the latest GitHub release of the repository into that cache path.
2. It downloads the two models from Hugging Face into `~/.cache/strawgo/models/` if they are not there yet: `silero_vad.onnx` from `onnx-community/silero-vad`, and `smart-turn-v3.1-cpu.onnx` from `pipecat-ai/smart-turn-v3`.
3. It starts the worker on a socket in your temp directory, waits up to 15 seconds for the socket to appear, and restarts the worker up to three times if it crashes.

This program starts the worker, connects a Silero VAD analyzer to it, and runs one inference on silence:

```go
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
	"github.com/square-key-labs/strawgo-ai/src/worker"
)

func main() {
	sup, err := worker.Start(os.Getenv("ONNX_WORKER_BIN"))
	if err != nil {
		log.Fatal(err)
	}
	defer sup.Stop()

	analyzer, err := vad.NewSileroVADAnalyzer(16000, vad.DefaultVADParams(), sup.SocketPath())
	if err != nil {
		log.Fatal(err)
	}
	defer analyzer.Cleanup()

	silence := make([]byte, analyzer.NumFramesRequired()*2)
	fmt.Printf("worker socket: %s\n", sup.SocketPath())
	fmt.Printf("voice confidence on silence: %.2f\n", analyzer.VoiceConfidence(silence))
}
```

`worker.Start` takes a binary path. The program passes `ONNX_WORKER_BIN`, so you can point it at a binary you built; when the variable is empty, the lookup and download described above happen. `sup.SocketPath()` is the socket to give to `vad.NewSileroVADAnalyzer` and `turn.NewOnnxSmartTurn`. `NumFramesRequired()` is the number of 16-bit samples per inference (512 at 16 kHz), so the buffer is twice that many bytes. A confidence near zero means the worker is up and the model loaded.

Pre-built binaries exist for Linux amd64, Linux arm64 and macOS arm64 (Apple Silicon). There is none for Intel Macs or Windows; build from source there.

> [!NOTE]
> The downloaded binary comes from the latest GitHub release. If your module pins an older Strawgo version or a commit from `main`, the binary can differ from your Go code; if VAD misbehaves, build the worker from the same commit as your Go code.

### Option 2: build from source

```sh
git clone https://github.com/square-key-labs/strawgo-ai
cd strawgo-ai/onnx-worker
cargo build --release
```

The binary is `onnx-worker/target/release/onnx-worker`. The build downloads the ONNX Runtime library automatically through the `ort` crate, so the first build is slow and you do not install ONNX Runtime yourself. Put the binary on your `PATH`, or pass its path to `worker.Start`.

### Option 3: run it as a separate process

In production you may prefer to run the worker under your process manager and share one socket across calls. Start it with the command shown at the top of this section, then give your bot the socket path. The [phone bot](first-phone-bot.md) and the repository examples read it from `ONNX_WORKER_SOCK`. You can fetch the models with `curl`:

```sh
mkdir -p ~/.cache/strawgo/models
curl -L -o ~/.cache/strawgo/models/silero_vad.onnx https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx
curl -L -o ~/.cache/strawgo/models/smart-turn-v3.1-cpu.onnx https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.1-cpu.onnx
```

The [deployment guide](../guides/deployment.md) covers running the worker next to your bot in production.

> [!NOTE]
> The repository also contains `scripts/install_onnxruntime.sh`. The Rust worker does not need it, because the `ort` crate fetches ONNX Runtime at build time. You can ignore it.

## Verify the install

Clone the repository and build everything. This checks that your Go toolchain is new enough:

```sh
git clone https://github.com/square-key-labs/strawgo-ai
cd strawgo-ai
go build ./...
```

Then run the text chat example. It needs only an LLM key:

```sh
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
go run examples/text_flow.go
```

You see a `>` prompt. Type a message and press Enter; the reply streams in below. Press Ctrl+D to end the conversation, or Ctrl+C to quit. To use another OpenAI-compatible server, such as a local Ollama, also set `OPENAI_BASE_URL` (for Ollama, `http://localhost:11434/v1`) and `OPENAI_MODEL`.

## Next steps

- [Build your own text bot in the quickstart](quickstart.md)
- [Build a phone bot](first-phone-bot.md)
- [Learn the core concepts](core-concepts.md)
