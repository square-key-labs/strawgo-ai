# Deployment

This guide takes a working bot to production: how the process handles many calls, how to shut down without dropping calls, a health endpoint, a Docker image, the ONNX worker for voice activity detection, secrets, scaling, security, and how to size machines.

## Before you begin

- A bot that works locally, for example from [Twilio](telephony-twilio.md).
- Familiarity with [Transports overview](transports-overview.md).
- Terms used on this page:
  - A **session** is one call or one browser connection.
  - **VAD** (voice activity detection) decides when the caller is speaking. Strawgo's Silero VAD and smart-turn models run in a separate process, the **ONNX worker**.
  - **Draining** means refusing new calls while letting current calls finish.

## Process model

One Strawgo process serves many concurrent calls:

- The WebSocket server (or WebRTC handler) accepts each connection and runs your handler on its own goroutine.
- Your handler builds a separate pipeline per call. Each processor runs its own goroutines. Nothing is shared between calls unless you share it.
- The ONNX worker is one extra process per host (or per container). Every call's VAD and smart-turn analyzer connects to it over a Unix socket.
- External services (STT, LLM, TTS) are reached over the network per call.

So a deployment is: N identical bot processes behind a load balancer, each with one ONNX worker next to it if you use Silero VAD or smart turn.

## A server with health checks and draining

This program adds what a bare example leaves out: a liveness and readiness endpoint, graceful draining on `SIGTERM`, an in-process supervisor for the ONNX worker, and fail-fast configuration.

```go
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/vad"
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
	"github.com/square-key-labs/strawgo-ai/src/worker"
)

type config struct {
	addr         string
	drainTimeout time.Duration
	twilioSID    string
	twilioToken  string
	deepgramKey  string
	openaiKey    string
	cartesiaKey  string
	voiceID      string
	workerBinary string
}

type bot struct {
	cfg       config
	vadSocket string
}

func main() {
	logger.Init()
	cfg := config{
		addr:         envOr("ADDR", ":8080"),
		drainTimeout: time.Duration(envInt("DRAIN_TIMEOUT_SECONDS", 600)) * time.Second,
		twilioSID:    mustEnv("TWILIO_ACCOUNT_SID"),
		twilioToken:  mustEnv("TWILIO_AUTH_TOKEN"),
		deepgramKey:  mustEnv("DEEPGRAM_API_KEY"),
		openaiKey:    mustEnv("OPENAI_API_KEY"),
		cartesiaKey:  mustEnv("CARTESIA_API_KEY"),
		voiceID:      mustEnv("CARTESIA_VOICE_ID"),
		workerBinary: os.Getenv("ONNX_WORKER_BIN"),
	}

	b := &bot{cfg: cfg, vadSocket: os.Getenv("ONNX_WORKER_SOCK")}
	if b.vadSocket == "" && cfg.workerBinary != "" {
		sup, err := worker.Start(cfg.workerBinary)
		if err != nil {
			log.Fatal(err)
		}
		defer sup.Stop()
		b.vadSocket = sup.SocketPath()
	}

	media, err := transports.NewServer(transports.ServerConfig{
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: cfg.twilioSID,
				AuthToken:  cfg.twilioToken,
			})
		},
		Handler: b.runCall,
		Params:  transports.TransportParams{SessionTimeout: 15 * time.Minute},
	})
	if err != nil {
		log.Fatal(err)
	}

	var draining atomic.Bool
	mux := http.NewServeMux()
	mux.Handle("/media", media)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Active-Calls", strconv.Itoa(media.ActiveSessions()))
		if draining.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	httpSrv := &http.Server{Addr: cfg.addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		draining.Store(true)
		log.Printf("draining %d active calls (up to %v)", media.ActiveSessions(), cfg.drainTimeout)
		drainCtx, cancel := context.WithTimeout(context.Background(), cfg.drainTimeout)
		defer cancel()
		if err := media.ShutdownContext(drainCtx); err != nil {
			log.Printf("drain timed out; remaining calls were closed")
		}
		shutdownCtx, cancelHTTP := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancelHTTP()
		_ = httpSrv.Shutdown(shutdownCtx)
	}()

	log.Printf("listening on %s", cfg.addr)
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func (b *bot) runCall(ctx context.Context, s *transports.Session) error {
	procs := []processors.FrameProcessor{s.Input()}
	if b.vadSocket != "" {
		analyzer, err := vad.NewSileroVADAnalyzer(16000, vad.DefaultVADParams(), b.vadSocket)
		if err != nil {
			return err
		}
		defer analyzer.Cleanup()
		procs = append(procs, vad.NewVADInputProcessor(analyzer))
	}
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: b.cfg.deepgramKey, Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: b.cfg.openaiKey, Model: "gpt-4.1-mini", SystemPrompt: "You are a helpful phone assistant. Keep answers short."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: b.cfg.cartesiaKey, VoiceID: b.cfg.voiceID, Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})
	procs = append(procs, stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant())
	return pipeline.NewPipelineTask(pipeline.NewPipeline(procs)).Run(ctx)
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
```

## Graceful shutdown

When your orchestrator stops an instance, you want current calls to finish and new calls to go elsewhere. The program above does this in four steps:

1. `SIGTERM` cancels `ctx`.
2. `draining` flips, so `/readyz` returns `503` and the load balancer stops sending new calls.
3. `media.ShutdownContext(drainCtx)` stops accepting new WebSocket upgrades (they get `503 server shutting down`) and waits for every active call to end on its own. If `drainCtx` expires first, it closes the remaining sessions, which cancels their pipelines and hangs up through the provider, and returns the context's error.
4. The HTTP server shuts down.

Set your platform's termination grace period (for example Kubernetes `terminationGracePeriodSeconds`) longer than `DRAIN_TIMEOUT_SECONDS`, and that longer than your longest expected call.

Other shutdown tools:

| Tool | Effect |
| --- | --- |
| `Server.ListenAndServe(ctx)` | Cancelling `ctx` closes every session immediately. Fine for development; it does not drain. |
| `Server.Shutdown()` | Closes every session now. |
| `Server.ShutdownContext(ctx)` | Drains as described above. |
| `Server.ActiveSessions()` | Number of calls in progress. |
| `webrtc.Handler.Shutdown(ctx)` | Refuses new offers, closes every peer and waits for handlers, up to `ctx`. It does not drain. |
| `PipelineTask.Cancel()` | Ends one call's pipeline now. |
| `PipelineTask.QueueFrame(frames.NewEndFrame())` | Ends one call after queued speech finishes. |

To end calls gracefully during a drain (for example, say "we are restarting, please call back"), keep a registry of running tasks in your handler and queue a `TTSSpeakFrame` and an `EndFrame` on each.

## Health endpoint

`transports.Server` serves only its WebSocket path, so mount it on your own `http.ServeMux` (as above) to add other routes:

- `/healthz` (liveness) answers `200` while the process is up.
- `/readyz` (readiness) answers `503` while draining, so no new calls arrive. It also reports the active call count in a header, which is handy for dashboards and autoscaling.

If you run the ONNX worker as a sidecar, add a readiness check that the socket exists (`os.Stat(socketPath)`).

## Docker image

This multi-stage Dockerfile builds your bot (a Go module whose `main` package is at the repository root), builds the ONNX worker from Strawgo's source, downloads the two ONNX models, and runs as a non-root user. Set `STRAWGO_REF` to the Strawgo version in your `go.mod`.

```dockerfile
FROM golang:1.25 AS bot
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/bot .

FROM rust:1-bookworm AS worker
ARG STRAWGO_REF=main
RUN git clone --depth 1 --branch "${STRAWGO_REF}" https://github.com/square-key-labs/strawgo-ai.git /strawgo
WORKDIR /strawgo/onnx-worker
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 \
 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 bot
RUN mkdir -p /home/bot/.cache/strawgo/models \
 && curl -fsSL -o /home/bot/.cache/strawgo/models/silero_vad.onnx https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx \
 && curl -fsSL -o /home/bot/.cache/strawgo/models/smart-turn-v3.1-cpu.onnx https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.1-cpu.onnx \
 && chown -R bot:bot /home/bot/.cache
COPY --from=bot /out/bot /usr/local/bin/bot
COPY --from=worker /strawgo/onnx-worker/target/release/onnx-worker /usr/local/bin/onnx-worker
USER bot
ENV ONNX_WORKER_BIN=/usr/local/bin/onnx-worker
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/bot"]
```

> [!WARNING]
> This Dockerfile has not been built as part of writing this guide. Build it once in CI before relying on it, and pin the base images and model URLs to versions you have tested.

How the pieces fit:

- `worker.Start(path)` (called when `ONNX_WORKER_BIN` is set) looks for the models in `~/.cache/strawgo/models/` and downloads them only if missing. The image pre-downloads them, so containers start without network access to Hugging Face.
- If you pass an empty path to `worker.Start`, it looks for `onnx-worker` on `PATH`, then in `~/.cache/strawgo/bin/`, then downloads a release binary from GitHub. Prebuilt binaries exist for Linux amd64 and arm64 and macOS arm64 only.
- The bot needs only `ca-certificates` to reach provider APIs; `curl` is used for the model download and can be removed after it.

Build and run:

```bash
docker build -t my-voice-bot --build-arg STRAWGO_REF=main .
docker run --rm -p 8080:8080 --env-file .env my-voice-bot
```

## Run the ONNX worker sidecar

Instead of supervising the worker from your bot, you can run it as its own process or container. The command line is:

```bash
onnx-worker --vad-model /models/silero_vad.onnx --turn-model /models/smart-turn-v3.1-cpu.onnx --socket /run/strawgo/onnx-worker.sock --threads 4
```

| Flag | Required | Meaning |
| --- | --- | --- |
| `--vad-model` | yes | Path to the Silero VAD ONNX model. |
| `--turn-model` | yes | Path to the Smart Turn ONNX model. |
| `--socket` | yes | Unix socket path to listen on. An existing file at that path is removed first. |
| `--threads` | no | Inference threads; defaults to the number of CPUs, capped at 4. |

Logging follows `RUST_LOG` (default `info`). Set `ONNX_WORKER_SOCK` in your bot to the same socket path, as the program above reads it. Both containers must share the socket's directory, for example with a shared volume:

```yaml
services:
  bot:
    image: my-voice-bot
    environment:
      ONNX_WORKER_SOCK: /run/strawgo/onnx-worker.sock
    env_file: .env
    ports:
      - "8080:8080"
    volumes:
      - worker-socket:/run/strawgo
  onnx-worker:
    image: my-voice-bot
    entrypoint:
      - /usr/local/bin/onnx-worker
      - --vad-model
      - /home/bot/.cache/strawgo/models/silero_vad.onnx
      - --turn-model
      - /home/bot/.cache/strawgo/models/smart-turn-v3.1-cpu.onnx
      - --socket
      - /run/strawgo/onnx-worker.sock
    volumes:
      - worker-socket:/run/strawgo
volumes:
  worker-socket:
```

Supervised versus sidecar:

| | `worker.Start` in the bot | Sidecar |
| --- | --- | --- |
| Restarts on crash | Up to 3 times (after 0.5 s, 1 s, 2 s), then gives up | Your orchestrator's restart policy |
| Socket path | `$TMPDIR/onnx-worker-<pid>.sock` | Your choice |
| Logs | Mixed into the bot's stdout and stderr | Separate |
| Shutdown | `Supervisor.Stop()` sends `SIGTERM`, then kills after 2 s | Your orchestrator |

`examples/twilio` and `examples/asterisk` read the socket path from `ONNX_WORKER_SOCK`; that variable is a convention of the examples, not something the library reads.

## Environment and secrets

- The library itself reads only two environment variables, both in `logger.Init()`: `LOG_LEVEL` (`DEBUG`, `INFO`, `WARN`, `ERROR`; default `INFO`) and `LOG_COLOR` (`false` or `0` turns colours off, which you want for log collectors).
- Everything else (API keys, Twilio credentials, socket paths) is read by your program. Read and check all of it at startup, as `mustEnv` does, so a missing key fails the deployment instead of the first call.
- Keep secrets in your platform's secret store and inject them as environment variables or files. Do not bake them into the image, and do not commit `.env` files. The repository's `.env.example` shows the names the examples use.
- For Twilio auto hang-up, prefer an API key (`APIKeySID` + `APIKeySecret`) over the account Auth Token, so you can rotate it on its own.
- Do not log API keys or full configuration structs. At `DEBUG` level Strawgo logs frame and protocol details, so review debug output before shipping it to a shared log store.

## Scaling and load balancing

Voice sessions are long-lived connections, which changes a few load-balancer defaults:

- **WebSocket support.** The load balancer must pass `Upgrade` and `Connection` headers through. Most do with WebSocket enabled.
- **Idle timeouts.** Set the idle timeout above the session ping interval (`TransportParams.PingInterval`, default 20 s). Strawgo pings every interval, so a 60 s idle timeout is enough to keep a quiet call alive.
- **No stickiness for telephony.** Each call is one WebSocket connection to one instance, so any instance can take any call.
- **Stickiness for WebRTC trickle ICE.** Peer connections live in the memory of the instance that answered the offer. A `PATCH` with candidates, or a renegotiation `POST` with a `pc_id`, must reach the same instance. Either route by `pc_id`, use sticky sessions, or avoid trickle (the page in [WebRTC](webrtc.md) waits for gathering and never sends a `PATCH`).
- **WebRTC media is UDP.** Each instance needs UDP ports reachable from browsers, or a TURN server that relays for it. Use a Pion `SettingEngine` (`Config.SettingEngine`) to set a port range and the public IP when behind NAT.
- **Scale on calls, not CPU alone.** Calls are long and their cost is uneven. Use the active-call count (`ActiveSessions()`) as a scaling signal, together with CPU.
- **Draining during deploys.** Roll instances one at a time and let each drain, as described above.

## Security checklist

- [ ] **TLS everywhere.** Serve `wss://` and `https://`, usually by terminating TLS at the load balancer. Providers require `wss://`.
- [ ] **Authenticate who connects.** By default `ServerConfig.CheckOrigin` accepts every connection. Strawgo does not validate Twilio's `X-Twilio-Signature`, Genesys AudioHook API keys, or any other provider signature. Add a check (see below).
- [ ] **Restrict browser origins.** For WebRTC and browser WebSocket clients, check the `Origin` header and authenticate the user (for example a short-lived token in `request_data` or a header) before creating a session. Wrap `webrtc.Handler` in your own middleware, or call `HandleOffer` yourself after checking.
- [ ] **Limit call length.** Set `SessionTimeout` (WebSocket and WebRTC) so a stuck or abusive call ends.
- [ ] **Treat caller input as untrusted.** Callers can say, type (RTVI `send-text`) or key in anything. Do not give the LLM tools that do damage without confirmation, and validate tool arguments in your handlers.
- [ ] **Protect recordings and transcripts.** Store them encrypted with access controls, and tell callers they are recorded. Keep `tracing.Config.RecordText` off unless your trace store is cleared for personal data.
- [ ] **Keep secrets out of logs and images** (see above).
- [ ] **Run as non-root** with a read-only root filesystem where possible; the Dockerfile above uses a dedicated user.

### Where to add a Twilio signature check

Twilio signs the WebSocket upgrade request with an `X-Twilio-Signature` header: an HMAC-SHA1 of the full stream URL, keyed with your Auth Token, base64-encoded. This check is not part of Strawgo and has not been tested against Twilio; confirm it against Twilio's request-validation documentation and test it with a real call.

```go
func twilioSignatureCheck(authToken, publicBaseURL string) func(*http.Request) bool {
	return func(r *http.Request) bool {
		mac := hmac.New(sha1.New, []byte(authToken))
		mac.Write([]byte(publicBaseURL + r.URL.RequestURI()))
		expected := base64.StdEncoding.EncodeToString(mac.Sum(nil))
		return hmac.Equal([]byte(expected), []byte(r.Header.Get("X-Twilio-Signature")))
	}
}
```

Plug it in as the server's `CheckOrigin`:

```go
func newCheckedServer(handler transports.SessionHandler) (*transports.Server, error) {
	return transports.NewServer(transports.ServerConfig{
		Path:        "/media",
		CheckOrigin: twilioSignatureCheck(os.Getenv("TWILIO_AUTH_TOKEN"), "wss://YOUR_HOST"),
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Handler: handler,
	})
}
```

`publicBaseURL` must be exactly the scheme and host Twilio connected to (the URL in your TwiML), because a proxy in front of the bot changes what the bot sees in `r.Host`. When `CheckOrigin` returns false, the upgrade fails with `403 Forbidden` and your handler never runs. You can do the same check as HTTP middleware around the server instead, which also lets you log rejected requests.

## Resource sizing

No sizing numbers are published, because cost per call depends on your providers, codecs, VAD and turn settings. Measure instead:

1. **Build a load test that looks like production.** Use your real pipeline with your real providers (or local fakes) and drive N concurrent calls, for example with a [WebSocket client](websocket-client.md) bot playing recorded audio into your bot.
2. **Watch per-instance numbers** as N grows: CPU, resident memory, goroutine count (`runtime.NumGoroutine()`), `ActiveSessions()`, and the user-to-bot latency from `UserBotLatencyObserver` (see [Observability](observability.md)). The knee where latency starts to rise is your per-instance limit; run below it.
3. **Profile hot spots** with Go's standard `net/http/pprof` on an internal port.
4. **Measure the ONNX worker separately** with `go run ./cmd/loadtest` (see [Observability: measure performance](observability.md#measure-performance)), and tune `--threads`.
5. **Repeat after every change** to providers, audio settings or Strawgo versions.

## Next steps

- [Observability](observability.md): metrics, latency and tracing in production.
- [Troubleshooting](troubleshooting.md)
- [WebRTC](webrtc.md): browser calls and TURN.
- [Telephony providers](telephony-providers.md)
