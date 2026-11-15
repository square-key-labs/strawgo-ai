# Writing a provider

LLM, STT and TTS services share one implementation each; a vendor integration only supplies the vendor-specific part.

| Kind | Shared service | You implement | Templates in the tree |
| --- | --- | --- | --- |
| LLM | `llmservice.Service` (`src/services/llmservice`) | `llm.Provider` | `src/services/anthropic`, `src/services/gemini`, `src/services/openaicompat` |
| STT, streaming | `stt.Base` from `stt.NewStreaming` (`src/services/stt`) | `stt.StreamingProvider` | `src/services/deepgram/stt.go`, `src/services/assemblyai` |
| STT, per utterance | `stt.Base` from `stt.NewSegmented` | `stt.SegmentedProvider` | `src/services/whisper` |
| TTS, streaming | `tts.Base` from `tts.NewStreaming` (`src/services/tts`) | `tts.StreamingProvider` | `src/services/cartesia/tts.go`, `src/services/deepgram/tts.go` |
| TTS, HTTP | `tts.Base` from `tts.NewHTTP` | `tts.HTTPProvider` | `src/services/openai/tts.go`, `src/services/google/tts.go` |

The shared services handle frames, interruptions, settings updates, metrics and lifecycle. Providers do I/O only. Rules for all of them:

- Honour the `ctx` you are given. For per-request calls it is the processor's epoch context, which is cancelled on interruption.
- Never log or put API keys in URLs when the vendor accepts a header.
- Keep hot paths (audio in, audio out, text deltas) free of per-frame goroutines and avoidable allocations.

## LLM: `llm.Provider`

```
type Provider interface {
	Name() string
	Stream(ctx context.Context, snapshot Snapshot, settings Settings, emit func(Event)) error
}
```

- `snapshot` is an immutable view of the conversation: `Messages`, `Tools`, `ToolChoice`. Do not modify it, and never write to the `*llm.Context`.
- `settings` is the service's current `llm.Settings` (model, system instruction, temperature, top-p, max tokens, seed, provider-specific `Extra`). Use `settings.SystemInstruction` as the system prompt.
- Call `emit` synchronously from the goroutine running `Stream`, and never after `Stream` returns. Events:
  - `llm.TextDelta{Text}`: output text.
  - `llm.ThoughtDelta{Text, Signature}`: reasoning output.
  - `llm.ToolCall{ID, Name, Arguments}`: one complete call; `Arguments` is the whole JSON object.
  - `llm.UsageEvent{Usage}`: token counts.
  - `llm.Done{StopReason}`.
- Return `ctx.Err()` (or any error wrapping `context.Canceled`) when cancelled; the service treats it as a normal interruption. Other errors become a non-fatal `ErrorFrame`.

The service emits `LLMFullResponseStartFrame`/`EndFrame`, `LLMTextFrame`, thought frames, TTFB, processing and usage metrics, applies `LLMUpdateSettingsFrame`, and runs function calls (see [architecture](architecture.md#function-calling)).

### OpenAI-compatible endpoints

If the vendor speaks the OpenAI Chat Completions protocol, reuse `openaicompat`; this is all `src/services/groq/llm.go` contains:

```go
package main

import (
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
)

type LLMConfig = openaicompat.ServiceConfig

var defaults = openaicompat.Defaults{Name: "Example", BaseURL: "https://llm.example.com/v1", Model: "example-large"}

func NewLLMService(cfg LLMConfig) *llmservice.Service {
	return openaicompat.NewService(cfg, defaults)
}

func main() {
	_ = NewLLMService(LLMConfig{APIKey: "key"})
}
```

### Any other API

A provider for a hypothetical non-streaming JSON endpoint:

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

type Provider struct {
	apiKey string
	url    string
	client *http.Client
}

type wireMessage struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

type wireRequest struct {
	Model    string        `json:"model"`
	System   string        `json:"system,omitempty"`
	Messages []wireMessage `json:"messages"`
}

type wireResponse struct {
	Text         string `json:"text"`
	InputTokens  int    `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
}

func (p *Provider) Name() string { return "ExampleLLM" }

func (p *Provider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	req := wireRequest{Model: settings.Model, System: settings.SystemInstruction}
	for _, m := range snapshot.Messages {
		if text := m.Text(); text != "" {
			req.Messages = append(req.Messages, wireMessage{Role: string(m.Role), Text: text})
		}
	}
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("example LLM HTTP %d", resp.StatusCode)
	}
	var out wireResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	emit(llm.TextDelta{Text: out.Text})
	emit(llm.UsageEvent{Usage: llm.Usage{
		PromptTokens:     out.InputTokens,
		CompletionTokens: out.OutputTokens,
		TotalTokens:      out.InputTokens + out.OutputTokens,
	}})
	emit(llm.Done{StopReason: "stop"})
	return nil
}

func NewLLMService(apiKey, model string) *llmservice.Service {
	p := &Provider{apiKey: apiKey, url: "https://llm.example.com/v1/generate", client: &http.Client{}}
	return llmservice.New(p, llmservice.WithSettings(llm.Settings{Model: model}))
}

func main() {
	_ = NewLLMService("key", "example-large")
}
```

`llmservice.New` options: `WithName`, `WithSettings`, `WithSequentialFunctionCalls`, `WithFunctionCallTimeout`. A real provider also maps tool schemas from `snapshot.Tools`, tool-call and tool-result parts (`m.ToolCalls()`, `m.ToolResult()`), images and audio parts; see `src/services/anthropic/llm.go`.

## STT

Both STT bases deliver audio as 16-bit little-endian mono PCM at `SampleRate()`: `stt.Options.SampleRate` if set, else the pipeline's `AudioInSampleRate`, else 16 kHz. They handle `AudioFrame`s, VAD and user speaking frames, `STTMuteFrame`, `STTUpdateSettingsFrame`, TTFB (measured from the end of user speech to the final transcript), usage metrics, and emit `STTMetadataFrame` with `Options.TTFSP99` when `Options.Provider` is set, which the speech-timeout stop strategy uses as its transcript wait.

`stt.Options`: `Provider`, `Settings` (`frames.STTSettings{Model, Language, Extra}`), `SampleRate`, `BytesPerSample`, `TTFSP99`, `TTFBTimeout`, `DropAudio` (do not forward `AudioFrame`s past the STT service), and for segmented services `PreRoll`, `TrailingSilence`, `MaxSegment`, `SegmentQueue`.

Optional interfaces on the provider:

- `stt.Finalizer` (`Finalize()`): called when the user stops speaking; ask the vendor to flush a final result.
- `stt.SettingsApplier` (`ApplySettings(frames.STTSettings) error`): called with the merged settings after `UpdateSettings`.

### Streaming

```
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

`Connect` is called once at start with the processor-lifetime context. `SendAudio` is called on the audio hot path: queue and return, do not block. Report results with `sink.Transcript(stt.Transcript{Text, Final, Finalized, Language})`; set `Finalized` when the vendor confirms the end of the utterance (this stops the TTFB clock). `sink.Push` sends other frames (for example vendor-side speech events).

`reconnect.New` (`src/services/reconnect`) provides the WebSocket session most streaming providers use: one writer goroutine, bounded send queue, keepalive, reconnect with backoff, graceful close message.

```go
package main

import (
	"context"
	"encoding/json"
	"net/http"
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
	Text    string `json:"text"`
	IsFinal bool   `json:"is_final"`
}

func NewSTTService(apiKey string) *STTService {
	s := &STTService{apiKey: apiKey}
	s.Base = stt.NewStreaming("ExampleSTT", s, stt.Options{
		Provider: "example",
		Settings: frames.STTSettings{Model: "example-rt", Language: "en"},
	})
	return s
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	session := reconnect.New(reconnect.Config{
		Name: "ExampleSTT",
		Endpoint: func() (string, http.Header) {
			h := http.Header{}
			h.Set("Authorization", "Bearer "+s.apiKey)
			return "wss://stt.example.com/v1/listen", h
		},
		OnMessage: func(_ int, data []byte) {
			var r result
			if json.Unmarshal(data, &r) == nil {
				sink.Transcript(stt.Transcript{Text: r.Text, Final: r.IsFinal, Finalized: r.IsFinal})
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

func (s *STTService) Disconnect(graceful bool) {
	if session := s.session.Load(); session != nil {
		session.Close(graceful)
	}
}

func main() {
	_ = NewSTTService("key")
}
```

### Per utterance (segmented)

```
type SegmentedProvider interface {
	Transcribe(ctx context.Context, wav []byte) (Transcript, error)
}
```

The base buffers audio between user-started and user-stopped speaking (with `PreRoll` before and `TrailingSilence` after, capped at `MaxSegment`), wraps it in a WAV container and calls `Transcribe` from a worker goroutine. Return one final `stt.Transcript`.

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
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
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://stt.example.com/v1/transcribe", bytes.NewReader(wav))
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
		return stt.Transcript{}, fmt.Errorf("example STT HTTP %d", resp.StatusCode)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return stt.Transcript{}, err
	}
	return stt.Transcript{Text: out.Text, Final: true, Finalized: true, Language: s.Settings().Language}, nil
}

func main() {
	_ = NewBatchSTT("key")
}
```

## TTS

The TTS base consumes `LLMTextFrame`, `TextFrame`, `AggregatedTextFrame` and `TTSSpeakFrame`, aggregates text into sentences (or passes tokens through with `Aggregation: tts.AggregateTokens`), applies `Options.Filters` (for example `tts.MarkdownFilter{}`), opens one context id per LLM response, emits `TTSStartedFrame`/`TTSStoppedFrame`, `TTSTextFrame`, `TTSAudioFrame` in `ChunkDuration` pieces, TTFB, processing and character-usage metrics, and handles interruptions and `TTSUpdateSettingsFrame`.

Audio is 16-bit little-endian mono PCM at `SampleRate()` (`tts.Options.SampleRate`, else the pipeline's `AudioOutSampleRate`, else 24 kHz). A provider that produces another encoding sets `Options.Codec` (for example `"mulaw"`) and `BytesPerSample`, or calls `SetAudioFormat` once it knows.

`tts.Options`: `Settings` (`frames.TTSSettings{Model, Voice, Language, Speed, Extra}`), `SampleRate`, `Codec`, `BytesPerSample`, `Aggregation`, `WordTimestamps`, `Filters`, `StopTimeout`, `ChunkDuration`. `tts.SettingsApplier` works as for STT.

### HTTP

```
type HTTPProvider interface {
	Synthesize(ctx context.Context, req Request) (io.ReadCloser, error)
}
```

`Synthesize` is called once per sentence on the TTS processor's data goroutine, with the epoch context. Return as soon as the response headers arrive; the base reads the body in chunks and closes it. `req` carries `ContextID`, `Text`, `Settings` and the target `SampleRate`.

```go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

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
	s.Base = tts.NewHTTP("ExampleTTS", s, tts.Options{
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
		return nil, fmt.Errorf("example TTS HTTP %d", resp.StatusCode)
	}
	return resp.Body, nil
}

func main() {
	_ = NewHTTPTTS("key", "narrator")
}
```

If the vendor returns WAV, wrap the body with `tts.NewReadCloser(tts.SkipWAVHeader(resp.Body), resp.Body)`. If it cannot produce the requested rate, resample with `src/audio/resample` as `src/services/openai/tts.go` does.

### Streaming

```
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

- `Connect` is called at start with the lifetime context; keep the connection for the whole session (`reconnect.New` again).
- `Speak` sends one sentence (or token) for a context; `Flush` marks the end of that context's text; `Cancel` is called on interruption for every context still open. None of them may block on the network; queue and return.
- From the read loop, call `sink.Audio` with PCM for a context, `sink.Words` with word timestamps when `Options.WordTimestamps` is set, and `sink.Done(contextID)` when the vendor has finished the context. Audio for contexts that were cancelled is discarded by the base.

See `src/services/cartesia/tts.go` for a complete implementation.

## Checklist

- Constructor returns your type embedding `*llmservice.Service`, `*stt.Base` or `*tts.Base`, so it can be placed directly in a pipeline.
- Settings flow through `Settings`/`UpdateSettings`; do not cache them in the provider.
- Tests use `httptest` servers or fakes; say so in the capability table in the README, since nothing in the tree is verified against live vendor endpoints.
