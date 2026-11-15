# WebSocket client

Most transports wait for someone to connect to your bot. `transports.Dial` does the opposite: your bot connects out to a WebSocket endpoint and runs a pipeline over that connection. This page shows when to use it and how.

## Before you begin

- Read [Transports overview](transports-overview.md). `Dial` returns the same `*transports.Session` that the WebSocket server gives your handler.
- Know which message format the remote endpoint speaks, because you must pick the serializer yourself. A **serializer** translates WebSocket messages into Strawgo frames and back. A **frame** is one unit of data in the pipeline, such as a chunk of audio.

## When to use it

- A media gateway or contact-center platform expects the bot to connect to it, rather than connecting to the bot.
- You run a test harness that drives one bot from another bot.
- A backend service streams audio to whoever connects, and your bot should listen and respond.

If the remote side connects to you, use `transports.NewServer` instead.

## A complete program

This bot connects to a WebSocket endpoint that speaks the Pipecat protobuf frame format, sends an auth header, and runs a voice pipeline until either side ends the conversation.

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

func main() {
	logger.Init()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	header := http.Header{}
	header.Set("Authorization", "Bearer "+mustEnv("GATEWAY_TOKEN"))
	dialCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	sess, err := transports.Dial(dialCtx, transports.ClientConfig{
		URL:        mustEnv("GATEWAY_URL"),
		Header:     header,
		Serializer: serializers.NewProtobufFrameSerializer(),
		Params: transports.TransportParams{
			AudioOutSampleRate: 16000,
			SessionTimeout:     30 * time.Minute,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		_ = sess.Close()
		sess.Wait()
	}()

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: mustEnv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: mustEnv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a helpful assistant. Keep answers short."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: mustEnv("CARTESIA_API_KEY"), VoiceID: mustEnv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		sess.Input(), stt, pair.User(), llmSvc, tts, sess.Output(), pair.Assistant(),
	})
	if err := pipeline.NewPipelineTask(pipe).Run(ctx); err != nil {
		log.Fatal(err)
	}
	log.Printf("conversation over")
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s is required", key)
	}
	return v
}
```

Run it with the endpoint and token in the environment:

```bash
export GATEWAY_URL=wss://gateway.example.com/bot
export GATEWAY_TOKEN=YOUR_GATEWAY_TOKEN
export DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export CARTESIA_API_KEY=YOUR_CARTESIA_API_KEY
export CARTESIA_VOICE_ID=YOUR_CARTESIA_VOICE_ID
go run .
```

### How it works

1. `transports.Dial` opens the connection with `Dialer` (default `websocket.DefaultDialer`) and the given `Header`. It fails if the connection cannot be opened before `ctx` expires.
2. It starts the session's reader right away, but frames are held until the pipeline's `StartFrame` reaches `sess.Input()`, so nothing is lost while you build the pipeline.
3. The pipeline is built exactly as in a server handler.
4. If the remote side closes the connection, input cancels the pipeline and `Run` returns. If your pipeline ends first (an `EndFrame`), output finishes sending audio and closes the connection.
5. The deferred `Close` and `Wait` release the connection and its goroutines in every case.

## `ClientConfig`

| Field | Required | Meaning |
| --- | --- | --- |
| `URL` | yes | `ws://` or `wss://` URL. |
| `Serializer` | yes | One serializer instance for this connection. |
| `Header` | no | Extra HTTP headers for the upgrade request, such as `Authorization`. |
| `Params` | no | The same `TransportParams` as the server. |
| `Dialer` | no | A `*websocket.Dialer` from `github.com/gorilla/websocket`, for proxies, TLS settings or timeouts. |

Differences from a server session:

- There is no handshake, even if the serializer has one, so `CallInfo()` is empty and `Request()` is `nil`.
- The output rate comes from the serializer's wire rate if it has one, then `Params.AudioOutSampleRate`, then the task's output rate. Set `AudioOutSampleRate` to what the remote side expects when the serializer has no fixed rate, as the program above does.

## Driving another bot: the RTVI client serializer

`serializers.NewRTVIClientSerializer()` makes your bot act as an [RTVI](rtvi.md) client. It sends audio as `raw-audio` JSON messages and turns the remote bot's RTVI messages into frames:

| Remote message | Frame in your pipeline |
| --- | --- |
| `bot-llm-started`, `bot-llm-text`, `bot-llm-stopped` | `LLMFullResponseStartFrame`, `LLMTextFrame`, `LLMFullResponseEndFrame` |
| `bot-tts-text` | `TTSTextFrame` |
| `bot-started-speaking`, `bot-stopped-speaking` | `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame` |
| `bot-interrupted` | `InterruptionFrame` |
| `user-transcription` | `TranscriptionFrame` or `InterimTranscriptionFrame` |
| `llm-function-call-in-progress`, `llm-function-call-stopped` | function call frames |
| `bot-llm-marker` | `LLMMarkerResponseFrame` |

It only sends outgoing transport messages that are RTVI messages (`"label": "rtvi-ai"`). This is useful for test harnesses that play recorded user audio into a bot under test and check what it says:

```go
func dialBotUnderTest(ctx context.Context, url string) (*transports.Session, error) {
	return transports.Dial(ctx, transports.ClientConfig{
		URL:        url,
		Serializer: serializers.NewRTVIClientSerializer(),
		Params:     transports.TransportParams{AudioOutSampleRate: 16000},
	})
}
```

The remote bot must accept RTVI JSON over a plain WebSocket text connection. A Strawgo bot using `NewProtobufFrameSerializer` does not (it expects protobuf), so check what your bot under test speaks.

## Next steps

- [Transports overview](transports-overview.md)
- [Telephony providers](telephony-providers.md): the serializers you can pass to `Dial`.
- [RTVI](rtvi.md)
- [Transports reference](../reference/transports.md)
