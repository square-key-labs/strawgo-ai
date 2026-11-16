# STT and TTS bases reference

This page documents the `stt` and `tts` base packages that every speech-to-text and text-to-speech provider is built on, and the `reconnect` package that streaming providers use for their WebSocket connections.

**STT** (speech-to-text) turns caller audio into text. **TTS** (text-to-speech) turns the bot's text into audio. A **provider** is the small adapter that talks to one vendor; the base does everything else (frames, metrics, interruptions, ordering). **VAD** (voice activity detection) tells the base when the user starts and stops talking. An **interruption** (barge-in) is the user talking over the bot.

Import paths: `github.com/square-key-labs/strawgo-ai/src/services/stt`, `.../src/services/tts`, `.../src/services/reconnect`.

To write a provider step by step, read [custom providers](../providers/custom.md). This page is the lookup.

## Quick example

The two functions below wrap a batch STT API and an HTTP TTS API. Each provider type implements one method; the base turns it into a full pipeline processor.

```go
type batchSTT struct {
	transcribe func(ctx context.Context, wav []byte) (string, error)
}

func (p batchSTT) Transcribe(ctx context.Context, wav []byte) (stt.Transcript, error) {
	text, err := p.transcribe(ctx, wav)
	if err != nil {
		return stt.Transcript{}, err
	}
	return stt.Transcript{Text: text, Language: "en"}, nil
}

type httpTTS struct {
	client *http.Client
	url    string
}

func (p httpTTS) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	body := strings.NewReader(fmt.Sprintf(`{"text":%q,"voice":%q,"sample_rate":%d}`, req.Text, req.Settings.Voice, req.SampleRate))
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, body)
	if err != nil {
		return nil, err
	}
	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("tts: HTTP %d", resp.StatusCode)
	}
	return resp.Body, nil
}

func newServices(transcribe func(context.Context, []byte) (string, error), ttsURL string) (*stt.Base, *tts.Base) {
	sttSvc := stt.NewSegmented("MySTT", batchSTT{transcribe: transcribe}, stt.Options{Provider: "my-stt"})
	ttsSvc := tts.NewHTTP("MyTTS", httpTTS{client: http.DefaultClient, url: ttsURL}, tts.Options{
		Settings: frames.TTSSettings{Voice: "alloy"},
		Filters:  []tts.TextFilter{tts.MarkdownFilter{}},
	})
	return sttSvc, ttsSvc
}
```

## STT base

```text
func NewStreaming(name string, provider StreamingProvider, opts Options) *Base
func NewSegmented(name string, provider SegmentedProvider, opts Options) *Base
```

Use `NewStreaming` for vendors that take a continuous audio stream and send transcripts back (usually over a WebSocket). Use `NewSegmented` for vendors that transcribe a whole file: the base cuts audio into one WAV per user utterance, using VAD frames, and calls the provider once per segment.

A streaming provider keeps the `Sink` it gets in `Connect` and calls it whenever the vendor sends a result. This skeleton shows the shape; `onVendorResult` is where your WebSocket read loop would report each result:

```go
type vendorSTT struct {
	sink stt.Sink
}

func (p *vendorSTT) Connect(ctx context.Context, sink stt.Sink) error {
	p.sink = sink
	return nil
}

func (p *vendorSTT) SendAudio(audio []byte)   {}
func (p *vendorSTT) Disconnect(graceful bool) {}

func (p *vendorSTT) onVendorResult(text string, final bool) {
	p.sink.Transcript(stt.Transcript{Text: text, Final: final, Language: "en"})
}

func newVendorSTT() *stt.Base {
	return stt.NewStreaming("VendorSTT", &vendorSTT{}, stt.Options{Provider: "vendor"})
}
```

### stt.Options

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Provider` | `string` | none | Provider ID. When set, the base pushes an `STTMetadataFrame` with `TTFSP99` after the `StartFrame`. |
| `Settings` | `frames.STTSettings` | zero | Initial model, language and extra settings. |
| `SampleRate` | `int` | `StartFrame.AudioInSampleRate`, else 16000 (`DefaultSampleRate`) | Input sample rate in Hz. |
| `BytesPerSample` | `int` | 2 | Used to convert audio bytes to seconds for usage metrics. |
| `TTFSP99` | `time.Duration` | 1 s (`DefaultTTFSP99`) | The provider's P99 time from end of speech to final transcript. The default speech-timeout stop strategy waits up to this long for a transcript. |
| `TTFBTimeout` | `time.Duration` | 2 s (`DefaultTTFBTimeout`) | How long after the user stops talking to wait for a provider-finalized transcript before reporting TTFB from the last final one. |
| `DropAudio` | `bool` | false | Consume `AudioFrame`s instead of forwarding them downstream. |
| `PreRoll` | `time.Duration` | 500 ms (`DefaultPreRoll`) | Segmented only: audio kept from before speech started. |
| `TrailingSilence` | `time.Duration` | 500 ms (`DefaultTrailingSilence`) | Segmented only: silence appended to each segment. Negative appends none. |
| `MaxSegment` | `time.Duration` | 30 s (`DefaultMaxSegment`) | Segmented only: a segment is sent when it reaches this length. |
| `SegmentQueue` | `int` | 8 | Segmented only: segments waiting for the provider. When full, the new segment is dropped and a non-fatal error is pushed. |

### Provider interfaces

```text
type StreamingProvider interface {
	Connect(ctx context.Context, sink Sink) error
	SendAudio(audio []byte)
	Disconnect(graceful bool)
}

type SegmentedProvider interface {
	Transcribe(ctx context.Context, wav []byte) (Transcript, error)
}

type Finalizer interface {
	Finalize()
}

type SettingsApplier interface {
	ApplySettings(settings frames.STTSettings) error
}
```

| Method | When the base calls it |
| --- | --- |
| `Connect` | On `StartFrame`, or on the first audio if that comes first. `ctx` is the processor's lifetime. An error is pushed as a non-fatal `ErrorFrame`. |
| `SendAudio` | For each downstream `AudioFrame` while not muted. Must not block for long. |
| `Disconnect` | `true` on `EndFrame` (flush and close), `false` on `CancelFrame`. |
| `Transcribe` | Once per segment, on a worker goroutine, with a WAV file. The result is always treated as final. |
| `Finalize` (optional) | When the user stops speaking. Ask the vendor to finalize now. |
| `ApplySettings` (optional) | After every settings change, with the full merged settings. |

`Transcript` fields: `Text string`, `Final bool` (false for interim results), `Finalized bool` (the vendor confirmed this is the end of the utterance; used for TTFB), `Language string`.

### stt.Sink

The sink is how a streaming provider sends results back. It is safe to call from any goroutine.

| Method | Effect |
| --- | --- |
| `Transcript(t Transcript)` | Empty text is ignored. `Final: false` pushes an `InterimTranscriptionFrame`; `Final: true` pushes a `TranscriptionFrame` and records TTFB and usage metrics. |
| `Push(frame frames.Frame, direction frames.FrameDirection)` | Push any other frame, for example `UserStartedSpeakingFrame` from a vendor's own turn detection. |
| `Error(err error, fatal bool)` | Push an `ErrorFrame` upstream. Fatal errors cancel the task by default. |

### Base methods and frames

| Method | Description |
| --- | --- |
| `Settings() frames.STTSettings` | Current settings. |
| `UpdateSettings(delta frames.STTSettings) error` | Merge and call `ApplySettings` if implemented. |
| `SetLanguage(language string)`, `SetModel(model string)` | `UpdateSettings` with one field; errors are logged. |
| `SampleRate() int` | The input sample rate in use. |
| `Muted() bool` | Whether an `STTMuteFrame` muted the service. |

| Frame | Action |
| --- | --- |
| `AudioFrame` (downstream) | Sent to the provider; forwarded unless `DropAudio`. |
| `VADUserStartedSpeakingFrame`, `UserStartedSpeakingFrame` | Start of an utterance (segmented: begin a segment). |
| `VADUserStoppedSpeakingFrame`, `UserStoppedSpeakingFrame` | End of an utterance: arm TTFB, call `Finalize`, close the segment. |
| `STTMuteFrame` | Set muted; consumed. |
| `STTUpdateSettingsFrame` | Apply if `Service` is nil or names this processor; consumed. |
| `EndFrame`, `CancelFrame` | Disconnect gracefully or immediately. |

Helpers: `WAV(pcm []byte, sampleRate, channels int) []byte` and `WAVHeader(dataLen, sampleRate, channels int) []byte`.

## TTS base

```text
func NewStreaming(name string, provider StreamingProvider, opts Options) *Base
func NewHTTP(name string, provider HTTPProvider, opts Options) *Base
func NewContextID() string
```

Use `NewStreaming` for vendors with a persistent connection that accepts text and streams audio back. Use `NewHTTP` for vendors with one HTTP request per piece of text.

The base groups each LLM response into one **context**, identified by a context ID, and makes sure audio from an interrupted context never plays.

A streaming TTS provider receives text per context and hands audio back through the `Sink`, tagged with the same context ID. This skeleton shows the shape; `onVendorAudio` and `onVendorDone` are where your read loop would report what the vendor sends:

```go
type vendorTTS struct {
	sink tts.Sink
}

func (p *vendorTTS) Connect(ctx context.Context, sink tts.Sink) error {
	p.sink = sink
	return nil
}

func (p *vendorTTS) Disconnect(graceful bool)           {}
func (p *vendorTTS) Speak(contextID, text string) error { return nil }
func (p *vendorTTS) Flush(contextID string) error       { return nil }
func (p *vendorTTS) Cancel(contextID string) error      { return nil }

func (p *vendorTTS) onVendorAudio(contextID string, pcm []byte) { p.sink.Audio(contextID, pcm) }
func (p *vendorTTS) onVendorDone(contextID string)              { p.sink.Done(contextID) }

func newVendorTTS() *tts.Base {
	return tts.NewStreaming("VendorTTS", &vendorTTS{}, tts.Options{})
}
```

### tts.Options

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Settings` | `frames.TTSSettings` | zero | Initial model, voice, language, speed and extra settings. |
| `SampleRate` | `int` | `StartFrame.AudioOutSampleRate`, else 24000 (`DefaultSampleRate`) | Output sample rate in Hz, also sent in `Request.SampleRate`. |
| `Codec` | `string` | `"linear16"` | Codec of the provider's audio, stamped on each `TTSAudioFrame` as `"codec"` metadata (`"linear16"`, `"mulaw"`, `"alaw"`). |
| `BytesPerSample` | `int` | 2 | Bytes per sample of the provider's audio; used to size HTTP chunks. |
| `Aggregation` | `AggregationMode` | `AggregateSentences` | `AggregateSentences` buffers LLM text and speaks whole sentences. `AggregateTokens` sends text as it arrives. |
| `WordTimestamps` | `bool` | false | When true, the base does not emit a `TTSTextFrame` per sentence; the provider reports words with `Sink.Words`. |
| `Filters` | `[]TextFilter` | none | Applied in order to each piece of text before it is sent. Reset on interruption and at the end of each turn. A `transforms.NewVoiceFormatter` chain is a `TextFilter`; see [Text for speech](../guides/text-for-speech.md). |
| `StopTimeout` | `time.Duration` | 2 s (`DefaultStopTimeout`) | A context that was flushed or produced audio is closed (`TTSStoppedFrame`) after this long with no new audio, even if the provider never calls `Done`. |
| `ChunkDuration` | `time.Duration` | 40 ms (`DefaultChunkDuration`) | HTTP only: audio is read and pushed in chunks of this length (at least 320 bytes). |

### Provider interfaces

```text
type StreamingProvider interface {
	Connect(ctx context.Context, sink Sink) error
	Disconnect(graceful bool)
	Speak(contextID, text string) error
	Flush(contextID string) error
	Cancel(contextID string) error
}

type HTTPProvider interface {
	Synthesize(ctx context.Context, req Request) (io.ReadCloser, error)
}

type SettingsApplier interface {
	ApplySettings(settings frames.TTSSettings) error
}

type TextFilter interface {
	Filter(text string) string
	Reset()
}
```

| Method | When the base calls it |
| --- | --- |
| `Connect` | On `StartFrame`. Errors become a non-fatal `ErrorFrame`. |
| `Speak` | For each sentence (or token) of a context. An error becomes a non-fatal `ErrorFrame`. |
| `Flush` | When a context has no more text (end of the LLM response, or after a standalone `TTSSpeakFrame`). If it fails, the context is closed. |
| `Cancel` | For each live context on interruption. Errors are logged at debug level. |
| `Disconnect` | `true` on `EndFrame` after playback drains (up to 10 s), `false` on `CancelFrame`. |
| `Synthesize` | Once per piece of text. Return the raw audio in `Options.Codec` at `req.SampleRate`. Closing the reader ends the request. |
| `ApplySettings` (optional) | After every settings change, with the full merged settings. |

`Request` fields: `ContextID string`, `Text string`, `Settings frames.TTSSettings`, `SampleRate int`.

### tts.Sink

| Method | Effect |
| --- | --- |
| `Audio(contextID string, audio []byte)` | Queue audio for that context. Audio for a cancelled or unknown context is dropped; late audio for a context that `StopTimeout` closed reopens it. The first audio of a context reports TTFB. |
| `Words(contextID string, words []Word)` | Queue a `TTSTextFrame` (type word, metadata `"word_start_time"` in seconds) per `Word{Text string; Start time.Duration}`. |
| `Done(contextID string)` | The vendor finished this context. The base emits `TTSStoppedFrame` once its audio is out. |
| `Error(err error, fatal bool)` | Push an `ErrorFrame` upstream. |

### Base methods and frames

| Method | Description |
| --- | --- |
| `Settings() frames.TTSSettings`, `UpdateSettings(delta frames.TTSSettings) error` | Read and merge settings; calls `ApplySettings` if implemented. |
| `SetVoice`, `SetModel`, `SetLanguage` | `UpdateSettings` with one field; errors are logged. |
| `SetAudioFormat(sampleRate int, codec string)` | Change the output format; zero or empty values are ignored. For providers that learn the format after connecting. |
| `SampleRate() int`, `Codec() string` | The format in use. |
| `AddFilters(filters ...TextFilter)` | Append filters after those in `Options.Filters`. Lets you add a filter to a built-in service, which embeds `*tts.Base`. Safe while the pipeline runs; applies from the next piece of text. A filter with per-stream state (a `transforms` formatter) is copied, here and in `Options.Filters`, so one value can be shared by several services. |

The base only acts on downstream frames; upstream frames pass straight through.

| Frame | Action |
| --- | --- |
| `LLMFullResponseStartFrame` | Close any open context and open a new one. |
| `LLMTextFrame` | Aggregate into the current context, unless `SkipTTS`. |
| `TextFrame` | Inside a response: aggregate. Outside: speak as its own context. `SkipTTS` passes it through. |
| `AggregatedTextFrame` | Speak as one unit, without further aggregation. |
| `TTSSpeakFrame` | Speak as its own context. Filters are reset before and after it. |
| `LLMFullResponseEndFrame` | Speak the remaining text and flush the context. |
| `TTSUpdateSettingsFrame` | Apply if `Service` is nil or names this processor; consumed. |
| `InterruptionFrame` | Drop queued audio, cancel live contexts, reset aggregation and filters, push `TTSStoppedFrame` upstream if something was playing. |
| `EndFrame` | Finish the turn, wait up to 10 s for playback to drain, disconnect gracefully. |
| `CancelFrame` | Drop audio and disconnect immediately. |

For each context the base emits, downstream and in order: `TTSStartedFrame{ContextID}`, `TTSAudioFrame`s (with `ContextID` set), a `TTSTextFrame` per spoken sentence or word, then `TTSStoppedFrame`. It also pushes `TTSStartedFrame` and `TTSStoppedFrame` upstream so the aggregators see them. When it sends text to the provider it pushes a `TTSQueuedTextFrame{Text, TextType, ContextID, PlaybackTracked}` downstream right away; it is a system frame, so it is not held behind audio, and the assistant aggregator uses it to know what was queued but not yet played. `PlaybackTracked` is true for HTTP providers and for streaming providers with `WordTimestamps`, whose `TTSTextFrame`s follow the audio they describe. When a context is closed (the LLM response ended, or a `TTSSpeakFrame` was sent) it pushes `TTSQueuedTextFrame{ContextID, Complete: true}`; an interrupted context is never marked complete. `tts.SplitSentences(text)` splits text into sentences with the same rules the base uses to aggregate LLM text. Other non-system frames are held in order behind the audio that precedes them. With metrics on it reports TTFB, character usage and, for HTTP providers, processing time.

Helpers:

| Helper | Purpose |
| --- | --- |
| `MarkdownFilter{}` | A `TextFilter` that strips links, headings, bullets and emphasis markers. |
| `transforms.NewVoiceFormatter(opts)` | In `src/utils/text/transforms`: a `TextFilter` that rewrites phone numbers, OTPs and ids, money, dates, percentages, units, acronyms and emails into speakable words for `en-US`, `en-IN` or `hi-IN`. Nothing is on unless you add it. See [Text for speech](../guides/text-for-speech.md). |
| `SkipWAVHeader(r io.Reader) io.Reader` | Skip a RIFF/WAVE header, if present, so only sample data is read. |
| `NewReadCloser(r io.Reader, c io.Closer) io.ReadCloser` | Pair a wrapped reader with the original closer. |

## reconnect.Session

`reconnect.Session` is a WebSocket client with a send queue, automatic reconnection with jittered exponential backoff, keepalives and graceful close. Streaming STT and TTS providers use it.

```text
func New(cfg Config) *Session
func (s *Session) Start(ctx context.Context)
func (s *Session) Send(messageType int, data []byte) bool
func (s *Session) SendDroppable(messageType int, data []byte) bool
func (s *Session) SendJSON(v any) error
func (s *Session) Clear()
func (s *Session) Reconnect()
func (s *Session) Close(graceful bool)
func (s *Session) Connected() bool
func (s *Session) Done() <-chan struct{}
func (s *Session) Dropped() uint64
```

This session authenticates on every connection attempt and sends a keepalive after five seconds without writes:

```go
func newVendorSocket(apiKey string, onMessage func(messageType int, data []byte)) *reconnect.Session {
	return reconnect.New(reconnect.Config{
		Name: "VendorSocket",
		Endpoint: func() (string, http.Header) {
			h := http.Header{}
			h.Set("Authorization", "Bearer "+apiKey)
			return "wss://api.example.com/v1/stream", h
		},
		OnMessage:     onMessage,
		KeepaliveIdle: 5 * time.Second,
		Keepalive: func() (reconnect.Message, bool) {
			return reconnect.Message{Type: websocket.TextMessage, Data: []byte(`{"type":"keepalive"}`)}, true
		},
	})
}
```

Call `Start(ctx)` from your provider's `Connect`, and `Close(graceful)` from `Disconnect`.

### reconnect.Config

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Name` | `string` | `"websocket"` | Log prefix. |
| `Endpoint` | `func() (string, http.Header)` | none (required) | Returns the URL and headers. Called before every connection attempt, so it can refresh tokens. |
| `Dialer` | `*websocket.Dialer` | proxy from environment, 10 s handshake timeout | Gorilla WebSocket dialer. |
| `OnOpen` | `func(w Writer) error` | none | Runs after each connect, before queued messages are sent; use it to send a setup message. An error counts as a failed attempt. |
| `OnMessage` | `func(messageType int, data []byte)` | none | Each received message. |
| `OnDisconnect` | `func(err error)` | none | A connection was lost unexpectedly. |
| `OnError` | `func(err error, fatal bool)` | none | Connection failures; `fatal` once reconnection gives up. |
| `CloseMessage` | `*Message` | none | Sent on graceful close; the session then waits for the server to close. |
| `DrainOnClose` | `bool` | false | On graceful close without a `CloseMessage`, still wait for the server to close. |
| `KeepaliveIdle` | `time.Duration` | 0 (off) | Send `Keepalive()` after this long with no writes. |
| `Keepalive` | `func() (Message, bool)` | none | Builds the keepalive message; false skips it. |
| `PingInterval` | `time.Duration` | 0 (off) | WebSocket ping interval. When set, a connection with no reads for 2 × ping + write timeout is treated as dead. |
| `WriteTimeout` | `time.Duration` | 5 s | Per-write deadline. |
| `CloseTimeout` | `time.Duration` | 2 s | How long a graceful close waits. |
| `QueueSize` | `int` | 512 | Messages buffered while sending or disconnected. |
| `Backoff` | `Backoff` | see below | Reconnection policy. |

`Backoff`: `Initial` (default 250 ms), `Max` (default 10 s), `MaxAttempts` (default 10; negative means never give up). Each delay is a random value between half and all of `Initial` doubled per attempt, capped at `Max`. Failures reset after a connection stays up for 5 s.

| Method | Description |
| --- | --- |
| `Start` | Connect in the background. Calling it again does nothing. |
| `Send` | Queue a message. Returns false if the queue is full of non-droppable messages. |
| `SendDroppable` | Queue a message that may be evicted (oldest first) to make room, for example audio. |
| `SendJSON` | Marshal and `Send` as text; returns `reconnect.ErrQueueFull` when full. |
| `Clear` | Discard queued messages, for example on interruption. |
| `Reconnect` | Close the current connection and connect again. |
| `Close(graceful)` | Stop. Graceful flushes the queue and closes politely; otherwise closes at once. Waits for the session to end. |
| `Connected`, `Done`, `Dropped` | Whether connected, a channel closed when the session ends, and the count of dropped messages. |

`Message` is `{Type int; Data []byte}` with Gorilla message types (`websocket.TextMessage`, `websocket.BinaryMessage`). `Writer` has `WriteMessage(messageType int, data []byte) error`.

## Next steps

- [Custom providers](../providers/custom.md): build a provider on these bases, step by step.
- [STT providers](../providers/stt.md) and [TTS providers](../providers/tts.md): the built-in ones.
- [Frames reference](frames.md): the frames these bases emit.
- [Turns reference](turns.md): how transcripts and VAD decide when the user is done.
