# Writing a custom provider

This page shows you how to add an LLM, STT or TTS vendor that Strawgo does not support yet. You get six complete skeletons, one for each kind of integration, and a checklist for each.

LLM, STT and TTS services each share one implementation. Your integration supplies only the vendor-specific I/O, and the shared service handles frames, interruptions, settings updates, metrics and lifecycle.

- **LLM** (large language model): you implement `llm.Provider` and wrap it in `llmservice.Service`.
- **STT** (speech-to-text): you implement `stt.StreamingProvider` or `stt.SegmentedProvider` and embed `*stt.Base`.
- **TTS** (text-to-speech): you implement `tts.StreamingProvider` or `tts.HTTPProvider` and embed `*tts.Base`.

A **frame** is a typed message that flows through the **pipeline**, the chain of processors that makes up a bot. An **interruption** (barge-in) is when the caller talks over the bot; the pipeline then cancels the work in flight.

| You are adding | Shared service | You implement | Real provider to copy |
| --- | --- | --- | --- |
| LLM with an OpenAI-compatible API | `openaicompat.NewService` | nothing but defaults | `src/services/groq/llm.go` |
| LLM with any other API | `llmservice.New` | `llm.Provider` | `src/services/anthropic/llm.go` |
| STT over a websocket | `stt.NewStreaming` | `stt.StreamingProvider` | `src/services/deepgram/stt.go` |
| STT, one HTTP request per utterance | `stt.NewSegmented` | `stt.SegmentedProvider` | `src/services/whisper/stt.go` |
| TTS over a websocket | `tts.NewStreaming` | `tts.StreamingProvider` | `src/services/cartesia/tts.go` |
| TTS, one HTTP request per sentence | `tts.NewHTTP` | `tts.HTTPProvider` | `src/services/openai/tts.go` |

Rules for every kind:

- Honour the `ctx` you are given. For per-request calls it is the processor's current context, which is cancelled on interruption.
- Send API keys in headers, not in URLs, whenever the vendor accepts a header. Never log them.
- Keep the audio and text hot paths free of per-frame goroutines and avoidable allocations.
- Test against a mock server (`net/http/httptest`, or a websocket upgrader inside one). Nothing in this repository is tested against live vendor APIs, and the [provider matrix](README.md) says so for every row; add your provider there with an honest status.

## LLM with an OpenAI-compatible API

If the vendor speaks the OpenAI Chat Completions protocol, reuse `openaicompat`. This is the whole of `src/services/groq/llm.go`, apart from the constants:

```go
package main

import (
	"os"

	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "Example", BaseURL: "https://llm.example.com/v1", Model: "example-large"}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults)
}

func main() {
	_ = NewLLMService(LLMConfig{APIKey: os.Getenv("EXAMPLE_API_KEY")})
}
```

When the server differs from OpenAI in small ways, pass `openaicompat.Option` values as the third argument of `NewService`. The [LLM providers page](llm.md#any-other-openai-compatible-server) lists them; `src/services/mistral/llm.go` shows several at once.

Checklist:

- [ ] `Defaults.Name` is set; it becomes the processor name in logs and metrics.
- [ ] Options match the server: no `developer` role, no `stream_options`, a different max-tokens or seed field.
- [ ] A test drives `NewProvider(cfg).Stream` against an `httptest` server that returns a recorded SSE stream.

## LLM with any other API

Implement `llm.Provider`:

```text
type Provider interface {
	Name() string
	Stream(ctx context.Context, snapshot Snapshot, settings Settings, emit func(Event)) error
}
```

- `snapshot` is an immutable view of the conversation: `Messages`, `Tools`, `ToolChoice`. Do not modify it, and never write to the `*llm.Context`; the context aggregators own it.
- `settings` is the service's current `llm.Settings` (model, system instruction, temperature, top-p, max tokens, seed, response format, provider-specific `Extra`). Use `settings.SystemInstruction` as the system prompt.
- Call `emit` from the goroutine running `Stream`, and never after `Stream` returns. The events are `llm.TextDelta`, `llm.ThoughtDelta`, `llm.ToolCall` (one complete call with the whole JSON arguments), `llm.UsageEvent`, `llm.ResponseInfo`, `llm.Refusal` and `llm.Done`.
- Return `ctx.Err()` (or any error wrapping `context.Canceled`) when cancelled; the service treats it as a normal interruption. Any other error becomes a non-fatal `ErrorFrame`.

The service emits `LLMFullResponseStartFrame` and `LLMFullResponseEndFrame`, `LLMTextFrame`s, thought frames, TTFB, processing and usage metrics, applies `LLMUpdateSettingsFrame`, and runs function calls.

This provider talks to a hypothetical non-streaming JSON endpoint. It maps tools, tool calls and tool results, so function calling works.

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

type Provider struct {
	apiKey string
	url    string
	client *http.Client
}

type wireCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type wireMessage struct {
	Role       string     `json:"role"`
	Text       string     `json:"text,omitempty"`
	ToolCalls  []wireCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type wireTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type wireRequest struct {
	Model    string        `json:"model"`
	System   string        `json:"system,omitempty"`
	Messages []wireMessage `json:"messages"`
	Tools    []wireTool    `json:"tools,omitempty"`
}

type wireResponse struct {
	Text         string     `json:"text"`
	ToolCalls    []wireCall `json:"tool_calls"`
	StopReason   string     `json:"stop_reason"`
	InputTokens  int        `json:"input_tokens"`
	OutputTokens int        `json:"output_tokens"`
}

func NewProvider(apiKey string) *Provider {
	return &Provider{apiKey: apiKey, url: "https://llm.example.com/v1/generate", client: &http.Client{}}
}

func (p *Provider) Name() string { return "ExampleLLM" }

func (p *Provider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	body, err := json.Marshal(buildRequest(snapshot, settings))
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("example LLM: HTTP %d", resp.StatusCode)
	}
	var out wireResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	if out.Text != "" {
		emit(llm.TextDelta{Text: out.Text})
	}
	for _, c := range out.ToolCalls {
		emit(llm.ToolCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
	}
	emit(llm.UsageEvent{Usage: llm.Usage{
		PromptTokens:     out.InputTokens,
		CompletionTokens: out.OutputTokens,
		TotalTokens:      out.InputTokens + out.OutputTokens,
	}})
	stop := llm.StopReasonStop
	if len(out.ToolCalls) > 0 {
		stop = llm.StopReasonToolCalls
	}
	emit(llm.Done{StopReason: stop, RawStopReason: out.StopReason})
	return nil
}

func buildRequest(snapshot llm.Snapshot, settings llm.Settings) wireRequest {
	req := wireRequest{Model: settings.Model, System: settings.SystemInstruction}
	for _, m := range snapshot.Messages {
		if r, ok := m.ToolResult(); ok {
			req.Messages = append(req.Messages, wireMessage{Role: "tool", Text: r.Content, ToolCallID: r.ToolCallID})
			continue
		}
		wm := wireMessage{Role: string(m.Role), Text: m.Text()}
		for _, c := range m.ToolCalls() {
			wm.ToolCalls = append(wm.ToolCalls, wireCall{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
		}
		if wm.Text != "" || len(wm.ToolCalls) > 0 {
			req.Messages = append(req.Messages, wm)
		}
	}
	for _, f := range snapshot.Tools.Functions {
		req.Tools = append(req.Tools, wireTool{Name: f.Name, Description: f.Description, Parameters: f.JSONSchema()})
	}
	return req
}

func NewLLMService(apiKey, model string) *llmservice.Service {
	return llmservice.New(NewProvider(apiKey), llmservice.WithSettings(llm.Settings{Model: model}))
}

func main() {
	_ = NewLLMService(os.Getenv("EXAMPLE_API_KEY"), "example-large")
}
```

Test it against a mock server by calling `Stream` directly. Put this in `provider_test.go` next to the provider:

```go
package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func TestStreamEmitsTextAndUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"m1"`) {
			http.Error(w, "bad model", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(w, `{"text":"hi","stop_reason":"end","input_tokens":3,"output_tokens":1}`)
	}))
	defer srv.Close()

	p := NewProvider("test-key")
	p.url = srv.URL
	var text string
	var usage llm.Usage
	err := p.Stream(context.Background(), llm.Snapshot{Messages: []llm.Message{llm.User("hello")}}, llm.Settings{Model: "m1"}, func(ev llm.Event) {
		switch e := ev.(type) {
		case llm.TextDelta:
			text += e.Text
		case llm.UsageEvent:
			usage = e.Usage
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if text != "hi" || usage.TotalTokens != 4 {
		t.Fatalf("text=%q usage=%+v", text, usage)
	}
}
```

`llmservice.New` options are `WithName`, `WithSettings`, `WithSequentialFunctionCalls` and `WithFunctionCallTimeout`. A production provider also streams (see `openaicompat.NewSSEReader`), maps image and audio parts, and turns vendor errors into typed errors; `src/services/anthropic/llm.go` does all of that.

Checklist:

- [ ] The API key is in a header.
- [ ] The request uses `http.NewRequestWithContext(ctx, …)`, and a cancelled `ctx` returns `ctx.Err()`.
- [ ] Usage is emitted as `llm.UsageEvent` so usage metrics work.
- [ ] `emit` is never called after `Stream` returns.
- [ ] The provider never writes to the context; it only reads `snapshot`.
- [ ] Tool schemas, tool calls and tool results are mapped, or the README row says tools are not supported.
- [ ] Tests run against an `httptest` server.

## STT over a websocket

Implement `stt.StreamingProvider` and embed `*stt.Base`:

```text
type StreamingProvider interface {
	Connect(ctx context.Context, sink Sink) error
	SendAudio(audio []byte)
	Disconnect(graceful bool)
}

type Sink interface {
	Transcript(t Transcript)
	Push(frame frames.Frame, direction frames.FrameDirection)
	Error(err error, fatal bool)
}
```

- `Connect` is called once when the pipeline starts, with the processor's lifetime context.
- `SendAudio` is called on the audio hot path with 16-bit little-endian mono PCM at `SampleRate()`. Queue the bytes and return; do not block.
- Report results with `sink.Transcript(stt.Transcript{Text, Final, Finalized, Language})`. Non-final results become `InterimTranscriptionFrame`s. Set `Finalized` when the vendor confirms the end of the utterance; that stops the TTFB clock.
- `sink.Push` sends any other frame, for example vendor speech events.
- `Disconnect(true)` is called on `EndFrame`, `Disconnect(false)` on `CancelFrame`.

Two optional interfaces are picked up automatically:

- `stt.Finalizer` (`Finalize()`): called when the user stops speaking. Ask the vendor to flush a final result.
- `stt.SettingsApplier` (`ApplySettings(frames.STTSettings) error`): called with the merged settings after an update. Most providers reconnect here.

`reconnect.New` (`src/services/reconnect`) gives you the websocket session most streaming providers use: one writer goroutine, a bounded send queue (512 messages by default), keepalive, reconnect with exponential backoff (250 ms to 10 s, 10 attempts by default), and a graceful close message. `SendDroppable` lets old audio be evicted when the queue is full; `Send` never drops.

```go
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync/atomic"

	"github.com/gorilla/websocket"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

type STTService struct {
	*stt.Base
	apiKey  string
	session atomic.Pointer[reconnect.Session]
}

type result struct {
	Text      string `json:"text"`
	IsFinal   bool   `json:"is_final"`
	EndOfTurn bool   `json:"end_of_turn"`
	Language  string `json:"language"`
}

func NewSTTService(apiKey string) *STTService {
	s := &STTService{apiKey: apiKey}
	s.Base = stt.NewStreaming("ExampleSTT", s, stt.Options{
		Provider: "example",
		Settings: frames.STTSettings{Model: "example-rt", Language: "en"},
	})
	return s
}

func (s *STTService) endpoint() (string, http.Header) {
	settings := s.Settings()
	q := url.Values{}
	q.Set("model", settings.Model)
	q.Set("language", settings.Language)
	q.Set("sample_rate", strconv.Itoa(s.SampleRate()))
	h := http.Header{}
	h.Set("Authorization", "Bearer "+s.apiKey)
	return "wss://stt.example.com/v1/listen?" + q.Encode(), h
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:     "ExampleSTT",
		Endpoint: s.endpoint,
		OnMessage: func(_ int, data []byte) {
			var r result
			if json.Unmarshal(data, &r) == nil {
				sink.Transcript(stt.Transcript{Text: r.Text, Final: r.IsFinal, Finalized: r.EndOfTurn, Language: r.Language})
			}
		},
		OnError:      sink.Error,
		CloseMessage: &reconnect.Message{Type: websocket.TextMessage, Data: []byte(`{"type":"close"}`)},
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *STTService) SendAudio(audio []byte) {
	if session := s.session.Load(); session != nil {
		session.SendDroppable(websocket.BinaryMessage, audio)
	}
}

func (s *STTService) Finalize() {
	if session := s.session.Load(); session != nil {
		session.Send(websocket.TextMessage, []byte(`{"type":"finalize"}`))
	}
}

func (s *STTService) ApplySettings(frames.STTSettings) error {
	if session := s.session.Load(); session != nil {
		session.Reconnect()
	}
	return nil
}

func (s *STTService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func main() {
	_ = NewSTTService(os.Getenv("EXAMPLE_API_KEY"))
}
```

`endpoint` is called on every (re)connect, so a settings update followed by `Reconnect` picks up the new model and language. Compare `src/services/deepgram/stt.go`, which adds keepalive messages and mu-law input.

Checklist:

- [ ] Auth goes in the handshake headers returned by `Endpoint`.
- [ ] `SendAudio` never blocks; audio uses `SendDroppable`.
- [ ] `Finalized` is set only when the vendor confirms the end of an utterance.
- [ ] `ApplySettings` reconnects (or sends a vendor configure message) so runtime changes take effect.
- [ ] Errors go to `sink.Error`; pass `fatal` only when the call cannot continue.
- [ ] `stt.Options.Provider` is set, so the base announces the service with an `STTMetadataFrame`.
- [ ] Tests run against an `httptest` server that upgrades to a websocket, like `src/services/deepgram/stt_test.go`.

## STT with one HTTP request per utterance

Implement `stt.SegmentedProvider`:

```text
type SegmentedProvider interface {
	Transcribe(ctx context.Context, wav []byte) (Transcript, error)
}
```

The base buffers audio between "user started speaking" and "user stopped speaking", with `PreRoll` (500 ms) before and `TrailingSilence` (500 ms) after, capped at `MaxSegment` (30 s). It wraps the audio in a WAV container and calls `Transcribe` from a worker goroutine. Return one final transcript. A segmented service needs speaking frames, so the pipeline must contain a VAD or another turn source.

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

type BatchSTT struct {
	*stt.Base
	apiKey string
	client *http.Client
}

func NewBatchSTT(apiKey string) *BatchSTT {
	s := &BatchSTT{apiKey: apiKey, client: &http.Client{Timeout: 30 * time.Second}}
	s.Base = stt.NewSegmented("ExampleBatchSTT", s, stt.Options{
		Provider: "example-batch",
		Settings: frames.STTSettings{Model: "example-batch", Language: "en"},
		TTFSP99:  2 * time.Second,
	})
	return s
}

func (s *BatchSTT) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	settings := s.Settings()
	url := "https://stt.example.com/v1/transcribe?model=" + settings.Model + "&language=" + settings.Language
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(wav))
	if err != nil {
		return stt.Transcript{}, err
	}
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Header.Set("Content-Type", "audio/wav")
	resp, err := s.client.Do(req)
	if err != nil {
		return stt.Transcript{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return stt.Transcript{}, fmt.Errorf("example STT: HTTP %d", resp.StatusCode)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return stt.Transcript{}, err
	}
	return stt.Transcript{Text: out.Text, Final: true, Finalized: true, Language: settings.Language}, nil
}

func main() {
	_ = NewBatchSTT(os.Getenv("EXAMPLE_API_KEY"))
}
```

`TTFSP99` is your estimate of the vendor's 99th-percentile time to final transcript. The speech-timeout stop strategy uses it as its transcript wait; measure it rather than guess. Reading `s.Settings()` inside `Transcribe` makes runtime updates apply from the next utterance without an `ApplySettings` hook.

Checklist:

- [ ] Auth in headers; `ctx` passed to the request.
- [ ] Settings read per request, not cached.
- [ ] `Final` and `Finalized` both true on the returned transcript.
- [ ] `MaxSegment` matches the vendor's upload limit (Google uses 55 s).
- [ ] Tests run against an `httptest` server, like `src/services/whisper/stt_test.go`.

## TTS over a websocket

Implement `tts.StreamingProvider` and embed `*tts.Base`:

```text
type StreamingProvider interface {
	Connect(ctx context.Context, sink Sink) error
	Disconnect(graceful bool)
	Speak(contextID, text string) error
	Flush(contextID string) error
	Cancel(contextID string) error
}

type Sink interface {
	Audio(contextID string, audio []byte)
	Words(contextID string, words []Word)
	Done(contextID string)
	Error(err error, fatal bool)
}
```

- `Connect` is called at start with the lifetime context; keep the connection for the whole call.
- The base opens one context ID per LLM response. `Speak` sends one sentence (or one token with `tts.AggregateTokens`) for a context; `Flush` marks the end of that context's text; `Cancel` is called on interruption for every context still open. None of them may block on the network.
- From the read loop, call `sink.Audio` with PCM for a context, `sink.Words` with word timestamps when `Options.WordTimestamps` is set, and `sink.Done` when the vendor has finished the context. Audio for cancelled contexts is discarded by the base.

```go
package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/reconnect"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

type TTSService struct {
	*tts.Base
	apiKey  string
	session atomic.Pointer[reconnect.Session]
}

type speakMessage struct {
	Type       string `json:"type"`
	ContextID  string `json:"context_id"`
	Text       string `json:"text,omitempty"`
	Voice      string `json:"voice,omitempty"`
	Model      string `json:"model,omitempty"`
	SampleRate int    `json:"sample_rate,omitempty"`
}

type serverMessage struct {
	Type      string `json:"type"`
	ContextID string `json:"context_id"`
	Audio     string `json:"audio"`
	Error     string `json:"error"`
}

func NewTTSService(apiKey, voice string) *TTSService {
	s := &TTSService{apiKey: apiKey}
	s.Base = tts.NewStreaming("ExampleTTS", s, tts.Options{
		Settings: frames.TTSSettings{Model: "example-voice-1", Voice: voice},
		Filters:  []tts.TextFilter{tts.MarkdownFilter{}},
	})
	return s
}

func (s *TTSService) endpoint() (string, http.Header) {
	h := http.Header{}
	h.Set("Authorization", "Bearer "+s.apiKey)
	return "wss://tts.example.com/v1/stream", h
}

func (s *TTSService) Connect(ctx context.Context, sink tts.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name:      "ExampleTTS",
		Endpoint:  s.endpoint,
		OnMessage: func(_ int, data []byte) { s.onMessage(sink, data) },
		OnError:   sink.Error,
	})
	s.session.Store(session)
	session.Start(ctx)
	return nil
}

func (s *TTSService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func (s *TTSService) send(m speakMessage) error {
	session := s.session.Load()
	if session == nil {
		return errors.New("example TTS not connected")
	}
	return session.SendJSON(m)
}

func (s *TTSService) Speak(contextID, text string) error {
	settings := s.Settings()
	return s.send(speakMessage{
		Type:       "speak",
		ContextID:  contextID,
		Text:       text,
		Voice:      settings.Voice,
		Model:      settings.Model,
		SampleRate: s.SampleRate(),
	})
}

func (s *TTSService) Flush(contextID string) error {
	return s.send(speakMessage{Type: "flush", ContextID: contextID})
}

func (s *TTSService) Cancel(contextID string) error {
	return s.send(speakMessage{Type: "cancel", ContextID: contextID})
}

func (s *TTSService) onMessage(sink tts.Sink, data []byte) {
	var m serverMessage
	if err := json.Unmarshal(data, &m); err != nil {
		return
	}
	switch m.Type {
	case "audio":
		audio, err := base64.StdEncoding.DecodeString(m.Audio)
		if err != nil {
			sink.Error(err, false)
			return
		}
		sink.Audio(m.ContextID, audio)
	case "done":
		sink.Done(m.ContextID)
	case "error":
		sink.Error(errors.New("example TTS: "+m.Error), false)
		if m.ContextID != "" {
			sink.Done(m.ContextID)
		}
	}
}

func main() {
	_ = NewTTSService(os.Getenv("EXAMPLE_API_KEY"), "narrator")
}
```

Audio is 16-bit little-endian mono PCM at `SampleRate()`: `tts.Options.SampleRate` if set, else the pipeline's `AudioOutSampleRate`, else 24 kHz. A vendor that produces another encoding sets `Options.Codec` (for example `"mulaw"`) and `BytesPerSample`, or calls `SetAudioFormat` once it knows.

Checklist:

- [ ] Auth goes in the handshake headers.
- [ ] `Speak`, `Flush` and `Cancel` only enqueue; `SendJSON` never waits on the network.
- [ ] Every context ends with `sink.Done`, including on vendor errors, or the base waits until `StopTimeout` (2 s by default).
- [ ] `Cancel` tells the vendor to stop, so audio for an interrupted reply stops arriving.
- [ ] Settings are read in `Speak`, or `ApplySettings` reconnects.
- [ ] `WordTimestamps: true` only if you call `sink.Words`.
- [ ] Tests run against a mock websocket server, like `src/services/cartesia/tts_test.go`.

## TTS with one HTTP request per sentence

Implement `tts.HTTPProvider`:

```text
type HTTPProvider interface {
	Synthesize(ctx context.Context, req Request) (io.ReadCloser, error)
}
```

`Synthesize` is called once per sentence with the context for that reply, which is cancelled on interruption. Return as soon as the response headers arrive; the base reads the body in chunks and closes it. `req` carries `ContextID`, `Text`, `Settings` and the target `SampleRate`.

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

type HTTPTTS struct {
	*tts.Base
	apiKey string
	client *http.Client
}

func NewHTTPTTS(apiKey, voice string) *HTTPTTS {
	s := &HTTPTTS{apiKey: apiKey, client: &http.Client{}}
	s.Base = tts.NewHTTP("ExampleHTTPTTS", s, tts.Options{
		Settings: frames.TTSSettings{Model: "example-voice-1", Voice: voice},
		Filters:  []tts.TextFilter{tts.MarkdownFilter{}},
	})
	return s
}

func (s *HTTPTTS) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	payload, err := json.Marshal(map[string]any{
		"text":        req.Text,
		"model":       req.Settings.Model,
		"voice":       req.Settings.Voice,
		"encoding":    "pcm_s16le",
		"sample_rate": req.SampleRate,
	})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://tts.example.com/v1/speech", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("example TTS: HTTP %d", resp.StatusCode)
	}
	return resp.Body, nil
}

func main() {
	_ = NewHTTPTTS(os.Getenv("EXAMPLE_API_KEY"), "narrator")
}
```

If the vendor returns WAV, wrap the body with `tts.NewReadCloser(tts.SkipWAVHeader(resp.Body), resp.Body)`. If it cannot produce the requested rate, resample with `src/audio/resample`, as `src/services/openai/tts.go` does.

Checklist:

- [ ] Auth in headers; `ctx` passed to the request.
- [ ] Settings come from `req.Settings`, so runtime updates apply on the next sentence.
- [ ] Non-200 responses close the body and return an error.
- [ ] The returned audio matches `req.SampleRate` and the codec in `tts.Options`.
- [ ] Tests run against an `httptest` server, like `src/services/google/tts_test.go`.

## Options reference

`stt.Options`: `Provider`, `Settings` (`frames.STTSettings{Model, Language, Extra}`), `SampleRate`, `BytesPerSample` (default 2), `TTFSP99` (default 1 s), `TTFBTimeout` (default 2 s), `DropAudio` (do not forward `AudioFrame`s past the service), and for segmented services `PreRoll`, `TrailingSilence`, `MaxSegment`, `SegmentQueue`.

`tts.Options`: `Settings` (`frames.TTSSettings{Model, Voice, Language, Speed, Extra}`), `SampleRate`, `Codec`, `BytesPerSample`, `Aggregation` (`tts.AggregateSentences` or `tts.AggregateTokens`), `WordTimestamps`, `Filters`, `StopTimeout` (default 2 s), `ChunkDuration` (default 40 ms). `tts.SettingsApplier` works as for STT.

The full API is in the [STT and TTS bases reference](../reference/stt-tts-bases.md) and the [LLM reference](../reference/llm.md).

## Next steps

- [Provider matrix](README.md): add your provider's row with an honest status.
- [LLM providers](llm.md), [STT providers](stt.md), [TTS providers](tts.md): see how the shipped integrations are configured.
- [Processors reference](../reference/processors.md): write a processor that is not a service.
