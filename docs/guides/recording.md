# Recording

This guide shows how to record a call's audio and collect a transcript. At the end you have a bot that saves `CALLID.wav` (caller on the left channel, bot on the right) and `CALLID.json` (who said what, and when) for every call.

## Before you begin

- A working phone bot, for example from [Twilio](telephony-twilio.md).
- Read [Transports overview](transports-overview.md#place-input-first-and-output-near-the-end) for where processors go in the pipeline.
- Terms used on this page:
  - A **processor** is one step in the pipeline. Frames (chunks of audio, text, or signals) flow through processors in order.
  - **STT** (speech-to-text) produces the caller's words. **TTS** (text-to-speech) produces the bot's voice.
  - **Mono** audio has one channel; **stereo** has two (left and right).
  - A **turn** is one stretch of speech by one side of the conversation.

> [!WARNING]
> Recording calls is regulated in many places. Tell callers they are being recorded and follow the consent rules that apply to you.

## Save the recording and transcript at the end of each call

```go
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/square-key-labs/strawgo-ai/src/audio/wav"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/processors/audiobuffer"
	"github.com/square-key-labs/strawgo-ai/src/processors/transcript"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

var recordingsDir = "recordings"

func main() {
	logger.Init()
	if err := os.MkdirAll(recordingsDir, 0o755); err != nil {
		log.Fatal(err)
	}
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

type wavRecorder struct {
	path   string
	file   *os.File
	writer *wav.Writer
}

func (r *wavRecorder) write(audio []byte, sampleRate, channels int) {
	if r.writer == nil {
		f, err := os.Create(r.path)
		if err != nil {
			log.Printf("recording: %v", err)
			return
		}
		w, err := wav.NewWriter(f, sampleRate, channels)
		if err != nil {
			log.Printf("recording: %v", err)
			_ = f.Close()
			return
		}
		r.file, r.writer = f, w
	}
	if _, err := r.writer.Write(audio); err != nil {
		log.Printf("recording: %v", err)
	}
}

func (r *wavRecorder) close() {
	if r.writer == nil {
		return
	}
	if err := r.writer.Close(); err != nil {
		log.Printf("recording: %v", err)
	}
	if err := r.file.Close(); err != nil {
		log.Printf("recording: %v", err)
	}
	r.writer = nil
}

func runCall(ctx context.Context, s *transports.Session) error {
	callID := s.CallInfo().CallID
	if callID == "" {
		callID = s.ID()
	}

	rec := &wavRecorder{path: filepath.Join(recordingsDir, callID+".wav")}
	recorder := audiobuffer.New(audiobuffer.Config{
		SampleRate:         8000,
		NumChannels:        2,
		AutoStartRecording: true,
	}, audiobuffer.Handlers{
		OnAudioData:        rec.write,
		OnRecordingStopped: rec.close,
	})

	var mu sync.Mutex
	var lines []transcript.Message
	transcripts := transcript.NewProcessor()
	transcripts.OnTranscriptUpdate(func(m transcript.Message) {
		mu.Lock()
		lines = append(lines, m)
		mu.Unlock()
	})

	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: mustEnv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: mustEnv("OPENAI_API_KEY"), Model: "gpt-4.1-mini", SystemPrompt: "You are a helpful phone assistant. Keep answers short."})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: mustEnv("CARTESIA_API_KEY"), VoiceID: mustEnv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})
	pair := aggregators.NewContextAggregatorPair(llm.NewContext(), aggregators.ContextAggregatorPairParams{})

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(),
		stt,
		transcripts.User(),
		pair.User(),
		llmSvc,
		tts,
		s.Output(),
		recorder,
		transcripts.Assistant(),
		pair.Assistant(),
	})
	runErr := pipeline.NewPipelineTask(pipe).Run(ctx)

	mu.Lock()
	data, err := json.MarshalIndent(lines, "", "  ")
	mu.Unlock()
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(recordingsDir, callID+".json"), data, 0o644); err != nil {
		return err
	}
	log.Printf("call %s saved", callID)
	return runErr
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("%s is required", key)
	}
	return v
}
```

After a call, `recordings/` contains the WAV file and a transcript like this:

```json
[
  {"Role": "user", "Content": "Hi, what are your opening hours?", "Timestamp": "2026-01-01T12:00:03.120Z", "UserID": ""},
  {"Role": "assistant", "Content": "We are open nine to five, Monday to Friday.", "Timestamp": "2026-01-01T12:00:04.010Z", "UserID": ""}
]
```

The field names are the Go field names of `transcript.Message`. Add JSON tags in your own struct if you want different names.

### How it works

1. `audiobuffer.New` creates the recorder. It goes **after `s.Output()`** so it sees the caller's `AudioFrame`s (which flow down the whole pipeline) and the bot's `TTSAudioFrame`s as the output plays them.
2. `AutoStartRecording` starts recording when the pipeline starts.
3. The recorder keeps the two sides aligned in time, inserting silence where one side is quiet, and hands audio to `OnAudioData` in large chunks. The writer appends each chunk to the WAV file and fixes up the header on `Close`.
4. When the call ends (an `EndFrame` or a `CancelFrame`, including a caller hang-up), the recorder flushes the rest and calls `OnRecordingStopped`, which closes the file.
5. `transcripts.User()` goes after STT and records each final transcription. `transcripts.Assistant()` goes after `s.Output()` and records what the bot actually said, sentence by sentence as TTS spoke it, flushing at the end of each response or interruption.
6. After `Run` returns, the handler writes the transcript.

## The audio buffer processor

### Configuration

| `audiobuffer.Config` field | Default | Meaning |
| --- | --- | --- |
| `SampleRate` | the task's output rate (24000 by default), or the first audio's rate | Rate of the recording. Both sides are resampled to it. Use 8000 for phone calls to avoid upsampling. |
| `NumChannels` | 1 | 1 mixes caller and bot into one track. 2 puts the caller on the left and the bot on the right. |
| `BufferSize` | 300 seconds of audio | Bytes buffered before `OnAudioData` is called. Smaller values give you audio sooner. |
| `MaxBufferSize` | none | Used as the flush size when `BufferSize` is 0. |
| `AutoStartRecording` | false | Start when the pipeline starts. Otherwise start it yourself. |
| `EnableTurnAudio` | false | Also collect audio per turn (see below). |
| `MaxTurnAudioSize` | 120 seconds of audio | Cap on one turn's audio, in bytes. |

### Callbacks

| `audiobuffer.Handlers` field | Called with |
| --- | --- |
| `OnAudioData(audio, sampleRate, numChannels)` | Mixed (mono) or interleaved (stereo) PCM, each time the buffer fills and when recording stops. |
| `OnTrackAudioData(user, bot, sampleRate, numChannels)` | The same audio as two separate mono tracks of equal length. |
| `OnUserTurnAudio(turn)`, `OnBotTurnAudio(turn)` | With `EnableTurnAudio`: one `audiobuffer.TurnAudio` per turn (`TurnNumber`, `Audio`, `SampleRate`, `NumChannels`). The caller's turn includes up to one second of audio from before they started speaking. |
| `OnRecordingStarted`, `OnRecordingStopped` | Recording state changes. |

Callbacks run on the pipeline's goroutine. Keep them quick, or hand the audio to another goroutine.

### Start and stop recording during a call

Turn recording on only after the caller agrees, for example:

```go
func startRecordingAfterConsent(task *pipeline.PipelineTask) error {
	return task.QueueFrame(frames.NewAudioBufferStartRecordingFrame())
}
```

`frames.NewAudioBufferStopRecordingFrame()` stops it and flushes what was recorded. You can also call `recorder.StartRecording()` and `recorder.StopRecording()` directly. `Recording()`, `HasAudio()`, `SampleRate()` and `NumChannels()` report state.

### Write a WAV file in one go

For short calls you can keep everything in memory and write once. `audiobuffer.WriteWAVFile(path, audio, sampleRate, numChannels)` and `audiobuffer.EncodeWAV` wrap the `wav` package:

```go
func saveWhole(path string) audiobuffer.Handlers {
	var mu sync.Mutex
	var all []byte
	var rate, channels int
	return audiobuffer.Handlers{
		OnAudioData: func(audio []byte, sampleRate, numChannels int) {
			mu.Lock()
			defer mu.Unlock()
			all = append(all, audio...)
			rate, channels = sampleRate, numChannels
		},
		OnRecordingStopped: func() {
			mu.Lock()
			defer mu.Unlock()
			if len(all) == 0 {
				return
			}
			if err := audiobuffer.WriteWAVFile(path, all, rate, channels); err != nil {
				log.Printf("save %s: %v", path, err)
			}
		},
	}
}
```

The streaming `wav.Writer` in the complete example uses constant memory, so prefer it for long calls.

> [!NOTE]
> The recorder treats audio as 16-bit PCM. Do not combine it with `TransportParams.AudioInNative` or `AudioOutNative`, which put µ-law bytes in audio frames. See [Audio: native passthrough](audio.md#native-passthrough).

## Save a transcript

`transcript.NewProcessor()` gives you a pair of processors that share one set of callbacks:

| Processor | Place it | Emits |
| --- | --- | --- |
| `User()` | after STT | One `transcript.Message` with `Role: transcript.RoleUser` per final, non-empty transcription. |
| `Assistant()` | after `s.Output()` | One message with `Role: transcript.RoleAssistant` per bot response, built from the text TTS spoke. It flushes when the bot stops speaking, on interruption, and at the end of the call, so an interrupted response contains only what was spoken before the interruption. |

Register callbacks with `OnTranscriptUpdate(func(transcript.Message))`. Each message has `Role`, `Content`, `Timestamp` (when the speech started) and `UserID`.

`transcript.NewProcessor(transcript.WithLLMText())` builds the assistant text from LLM output instead of TTS text, flushing at the end of each LLM response. Use it when your bot replies in text only. Note that it then records what the LLM wrote, not what was spoken.

You can also use the two halves alone with `transcript.NewUserProcessor()` and `transcript.NewAssistantProcessor()`.

For the full conversation including system prompt and tool calls, read the LLM context instead (`llm.Context.Messages()`); see [Conversation context](conversation-context.md).

## Next steps

- [Call recording and transcript recipe](../cookbook/call-recording-and-transcript.md)
- [Audio](audio.md)
- [Observability](observability.md): metrics and latency per call.
- [Processors reference](../reference/processors.md)
