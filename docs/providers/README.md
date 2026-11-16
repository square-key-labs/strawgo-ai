# Providers

This page lists every vendor integration in Strawgo and the test status of every framework component, so you can pick services for your bot and see how each part is built and tested. Each row links to the page that documents it in detail. This page is the single source of truth for test status; other pages link here.

> [!NOTE]
> No provider in this repository has been verified against the live vendor API. The status column says what the code is tested against: a mock server that imitates the vendor, unit tests without a server, or nothing at all. Treat every integration as something you verify in your own account before you rely on it.

A voice bot needs three kinds of service, or one service that does all three:

- **STT** (speech-to-text) turns the caller's audio into text.
- **LLM** (large language model) decides what to say next.
- **TTS** (text-to-speech) turns the reply into audio.
- **Speech-to-speech** services take audio in and give audio out, with the model in between.

Every service is a pipeline processor. A **pipeline** is the chain of processors that **frames** (small typed messages such as audio chunks or text) flow through. You put the service you construct directly into `pipeline.NewPipeline`.

## How to read the status column

| Status | Meaning |
| --- | --- |
| mock-tested | A `*_test.go` file in the package drives this constructor against a local mock server (`httptest` or a fake websocket). |
| unit-tested | The package has tests, but they check request building or validation without a server. |
| untested | No test in the package exercises this constructor. |
| not available | Absent from the repository, or present but never compiled here. |

Rows marked untested that are built on `openaicompat` still go through the shared OpenAI-compatible client, which is mock-tested in `src/services/openaicompat`. Only the vendor-specific defaults and options in those packages are untested.

To check the status yourself, run the tests of one package:

```sh
go test ./src/services/deepgram/...
```

## LLM

All LLM services return `*llmservice.Service` (or embed it), so function calling, settings updates and metrics behave the same way. See [LLM providers](llm.md).

| Provider | Package | Constructor | Connection | Notable capabilities | Status |
| --- | --- | --- | --- | --- | --- |
| Anthropic | `src/services/anthropic` | `NewLLMService` | HTTP streaming (SSE) | tools, extended thinking, prompt caching, JSON-schema output | mock-tested |
| AWS Bedrock | `src/services/aws/bedrock` | `NewLLMService`, `NewLLMServiceWithClient` | AWS SDK (`ConverseStream`) | tools, prompt caching, additional model request fields | unit-tested |
| Azure OpenAI | `src/services/azureopenai` | `NewLLMService` | HTTP streaming (SSE) | tools, deployment or `/openai/v1` endpoints | untested |
| Cerebras | `src/services/cerebras` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| DeepSeek | `src/services/deepseek` | `NewLLMService` | HTTP streaming (SSE) | tools, reasoning content (off by default) | untested |
| Fireworks | `src/services/fireworks` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| Google Gemini | `src/services/gemini` | `NewLLMService` | HTTP streaming (SSE) | tools, thinking | mock-tested |
| Google Vertex AI | `src/services/vertex` | `NewLLMService` | HTTP streaming (SSE) | tools, thinking, Google service-account auth | unit-tested |
| Groq | `src/services/groq` | `NewLLMService` | HTTP streaming (SSE) | tools | mock-tested |
| Mistral | `src/services/mistral` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| NVIDIA | `src/services/nvidia` | `NewLLMService` | HTTP streaming (SSE) | tools, inline `<think>` reasoning | mock-tested |
| Ollama | `src/services/ollama` | `NewLLMService` | HTTP streaming (SSE) | tools, local models, no API key needed | mock-tested |
| OpenAI (Chat Completions) | `src/services/openai` | `NewLLMService` | HTTP streaming (SSE) | tools, JSON-schema output | untested |
| OpenAI (Responses API) | `src/services/openai` | `NewResponsesLLMService` | HTTP streaming (SSE) | tools, reasoning summaries | mock-tested |
| OpenAI-compatible (any) | `src/services/openaicompat` | `NewService`, `New` + `Wrap` | HTTP streaming (SSE) | tools, configurable wire quirks | mock-tested |
| OpenRouter | `src/services/openrouter` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| Perplexity | `src/services/perplexity` | `NewLLMService` | HTTP streaming (SSE) | no tools (the integration does not send them) | untested |
| Qwen (DashScope) | `src/services/qwen` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| SambaNova | `src/services/sambanova` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| Sarvam | `src/services/sarvam` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| Together AI | `src/services/together` | `NewLLMService` | HTTP streaming (SSE) | tools | untested |
| xAI Grok | `src/services/xai` | `NewLLMService` | HTTP streaming (SSE) | tools, reasoning-token usage fix-up | untested |

## STT

All STT services except OpenAI Realtime STT (`openai_realtime`) embed `*stt.Base`. "Segmented" services send one HTTP request per utterance; "streaming" services keep a websocket open for the whole call. See [STT providers](stt.md).

| Provider | Package | Constructor | Connection | Notable capabilities | Status |
| --- | --- | --- | --- | --- | --- |
| AssemblyAI | `src/services/assemblyai` | `NewSTTService` | websocket | keyterms, end-of-turn callback | mock-tested |
| AWS Transcribe | `src/services/aws/transcribe` | `NewSTTService`, `NewSTTServiceWithClient` | AWS SDK (streaming) | language identification, vocabularies, speaker labels | untested |
| Azure Speech | `src/services/azure` | `NewSTTService` | websocket | region or custom endpoint | mock-tested |
| Cartesia | `src/services/cartesia` | `NewSTTService` | websocket | keyterms, finalize on end of speech | untested |
| Deepgram | `src/services/deepgram` | `NewSTTService` | websocket | interim results, smart format, mulaw/alaw input | mock-tested |
| Deepgram Flux | `src/services/deepgram` | `NewFluxSTTService` | websocket | server-side turn detection, eager end of turn | untested |
| ElevenLabs Scribe | `src/services/elevenlabs` | `NewSTTService` | HTTP, segmented | keyterms, diarization, audio-event tags | untested |
| ElevenLabs Scribe Realtime | `src/services/elevenlabs` | `NewRealtimeSTTService` | websocket | manual or VAD commit, language detection | untested |
| fal (Wizper) | `src/services/fal` | `NewSTTService` | HTTP, segmented | transcribe or translate task | untested |
| Gladia | `src/services/gladia` | `NewSTTService` | HTTP session init, then websocket | optional server-side turn detection, translation, custom vocabulary | untested |
| Google Speech-to-Text | `src/services/google` | `NewSTTService` | HTTP, segmented | API key or service-account auth, multiple languages | untested |
| Groq Whisper | `src/services/groq` | `NewSTTService` | HTTP, segmented | prompt, temperature | untested |
| OpenAI transcription | `src/services/openai` | `NewSTTService` | HTTP, segmented | prompt, keywords | untested |
| OpenAI Realtime transcription | `src/services/openai` | `NewRealtimeSTTService` | websocket | optional server-side turn detection, noise reduction | untested |
| OpenAI Realtime STT (standalone) | `src/services/openai_realtime` | `NewSTTService` | websocket | GA transcription session, server VAD option, language, noise reduction; not built on `stt.Base` | mock-tested |
| Sarvam | `src/services/sarvam` | `NewSTTService` | websocket | Indian languages, optional vendor VAD signals | untested |
| Sarvam Realtime | `src/services/sarvam` | `NewRealtimeSTTService` | websocket | low latency, server-side speech detection, interim transcripts, `saaras:v3-realtime` or `saaras:v4` | untested |
| Soniox | `src/services/soniox` | `NewSTTService` | websocket | language hints, diarization, optional server-side turn detection | untested |
| Whisper (OpenAI API) | `src/services/whisper` | `NewSTTService` | HTTP, segmented | any Whisper-compatible endpoint | mock-tested |

## TTS

All TTS services embed `*tts.Base`. See [TTS providers](tts.md).

| Provider | Package | Constructor | Connection | Notable capabilities | Status |
| --- | --- | --- | --- | --- | --- |
| Async | `src/services/asyncai` | `NewHTTPTTSService` | HTTP | mulaw/alaw output | untested |
| AWS Polly | `src/services/aws/polly` | `NewTTSService`, `NewTTSServiceWithClient` | AWS SDK | engines, lexicons, SSML prosody | untested |
| Azure Speech | `src/services/azure` | `NewTTSService` | HTTP | output formats incl. mulaw | mock-tested |
| Cartesia | `src/services/cartesia` | `NewTTSService` | websocket | word timestamps, token streaming | mock-tested |
| Cartesia (HTTP) | `src/services/cartesia` | `NewHTTPTTSService` | HTTP | generation config | untested |
| Deepgram Aura | `src/services/deepgram` | `NewTTSService` | websocket | mulaw/alaw output | mock-tested |
| Deepgram Aura (HTTP) | `src/services/deepgram` | `NewHTTPTTSService` | HTTP | speed | untested |
| ElevenLabs | `src/services/elevenlabs` | `NewTTSService` | websocket (`UseStreaming`) or HTTP | word timestamps in websocket mode, token streaming | mock-tested |
| ElevenLabs (HTTP with timestamps) | `src/services/elevenlabs` | `NewHTTPTTSService` | HTTP streaming | word timestamps | untested |
| Fish Audio | `src/services/fish` | `NewTTSService` | websocket (MessagePack) | reference voices, latency mode | untested |
| Google Cloud TTS | `src/services/google` | `NewTTSService` | HTTP | mulaw/alaw output | mock-tested |
| Groq (Orpheus) | `src/services/groq` | `NewTTSService` | HTTP | speed | untested |
| Inworld | `src/services/inworld` | `NewTTSService` | websocket | word timestamps, token streaming | untested |
| Inworld (HTTP) | `src/services/inworld` | `NewHTTPTTSService` | HTTP streaming | word timestamps | untested |
| MiniMax | `src/services/minimax` | `NewTTSService` | HTTP | group ID, extra params | untested |
| Neuphonic | `src/services/neuphonic` | `NewTTSService` | websocket | token streaming | untested |
| Neuphonic (HTTP) | `src/services/neuphonic` | `NewHTTPTTSService` | HTTP | speed, temperature | untested |
| OpenAI | `src/services/openai` | `NewTTSService` | HTTP | voice instructions, resampling from 24 kHz | untested |
| Piper (self-hosted) | `src/services/piper` | `NewTTSService` | HTTP | no API key | untested |
| Resemble | `src/services/resemble` | `NewTTSService` | websocket | word timestamps | untested |
| Rime | `src/services/rime` | `NewTTSService` | websocket | word timestamps, token streaming | untested |
| Rime (HTTP) | `src/services/rime` | `NewHTTPTTSService` | HTTP | speed, extra params | untested |
| Sarvam | `src/services/sarvam` | `NewTTSService` | websocket | Indian languages, token streaming | untested |
| Sarvam (HTTP) | `src/services/sarvam` | `NewHTTPTTSService` | HTTP | pace | untested |
| Speechify | `src/services/speechify` | `NewTTSService` | HTTP streaming | word timestamps | untested |
| XTTS (self-hosted) | `src/services/xtts` | `NewTTSService` | HTTP | no API key | untested |

## Speech-to-speech

These services replace the STT, LLM and TTS trio with one model. They are not built on `llmservice.Service`. See [Speech-to-speech providers](speech-to-speech.md).

| Provider | Package | Constructor | Connection | Notable capabilities | Status |
| --- | --- | --- | --- | --- | --- |
| OpenAI Realtime | `src/services/openai_realtime` | `NewLLMService`, `NewOpenAIRealtimeLLMService` | websocket | GA API, `gpt-realtime-2.1`, tools, server or semantic VAD, input transcription, noise reduction, truncation | mock-tested |
| Gemini Live | `src/services/gemini` | `NewLiveService`, `NewGeminiLiveService` | websocket | `gemini-3.8-live`, tools, server VAD, context window compression | mock-tested |

> [!NOTE]
> Speechmatics STT and AWS Nova Sonic have no integration in this repository.

## Support packages

These packages under `src/services` are not vendor integrations. You use them when you write your own provider or need a shared feature.

| Package | What it gives you | Status |
| --- | --- | --- |
| `src/services/llmservice` | `llmservice.Service`, the processor that wraps any `llm.Provider` | unit-tested |
| `src/services/openaicompat` | the OpenAI Chat Completions client every compatible LLM reuses | mock-tested |
| `src/services/openaicompat/httpretry` | retry and first-byte timeout policy for HTTP LLM calls | mock-tested |
| `src/services/stt` | `stt.Base` for streaming and segmented STT providers | unit-tested |
| `src/services/tts` | `tts.Base` for streaming and HTTP TTS providers | mock-tested |
| `src/services/reconnect` | the websocket session most streaming providers use | mock-tested |
| `src/services/realtime` | socket, output and function-call helpers for speech-to-speech services | unit-tested |
| `src/services/aws/awsauth` | AWS configuration loading shared by Bedrock, Polly and Transcribe | untested |
| `src/services/mcp` | MCP client that turns MCP server tools into LLM functions (see [MCP tools](../guides/mcp-tools.md)) | untested |

## Framework components

The same status values apply to the rest of the framework. "mock-tested" here means the tests run against a local fake peer: a WebSocket or HTTP test server, a local WebRTC peer, or a fake `onnx-worker` socket.

| Component | Package | Status | Guide |
| --- | --- | --- | --- |
| WebSocket server, one `Session` per connection (`NewServer`) | `src/transports` | mock-tested | [Transports overview](../guides/transports-overview.md) |
| WebSocket client (`Dial`) | `src/transports` | mock-tested | [WebSocket client](../guides/websocket-client.md) |
| WebRTC handler, G.711 (PCMU/PCMA) | `src/transports/webrtc` | mock-tested | [WebRTC](../guides/webrtc.md) |
| WebRTC Opus codec (build tag `opus`) | `src/transports/webrtc` | untested: CI compiles it with libopus, but no test exercises the codec | [WebRTC](../guides/webrtc.md) |
| Daily, LiveKit, video on any transport | none | not available | |
| Twilio serializer: media, `mark` playback ack, DTMF, clear | `src/serializers` | unit-tested | [Twilio telephony](../guides/telephony-twilio.md) |
| Twilio, Telnyx and Plivo REST hang-up | `src/serializers` | mock-tested | [Telephony providers](../guides/telephony-providers.md) |
| Telnyx, Plivo, Exotel, Vonage, Asterisk, Genesys AudioHook, Jambonz serializers | `src/serializers` | unit-tested | [Telephony providers](../guides/telephony-providers.md) |
| Protobuf and RTVI client serializers | `src/serializers` | unit-tested | [RTVI](../guides/rtvi.md) |
| Resampler, G.711, PCM, WAV, DTMF tones, input filters, output mixer | `src/audio/...` | unit-tested | [Audio](../guides/audio.md) |
| Silero VAD (through the `onnx-worker`) | `src/audio/vad` | mock-tested (fake worker socket) | [Turn-taking](../guides/turn-taking.md) |
| Smart Turn: ONNX, HTTP and fal | `src/audio/turn` | mock-tested (fake worker socket, HTTP test server) | [Turn-taking](../guides/turn-taking.md) |
| Turn start, stop and mute strategies | `src/turns/...` | unit-tested | [Turn-taking](../guides/turn-taking.md) |
| User idle detection | `src/processors`, `src/processors/aggregators` | unit-tested | [Turn-taking](../guides/turn-taking.md) |
| DTMF aggregator (`NewDTMFAggregator`) | `src/processors/aggregators` | untested | [Twilio telephony](../guides/telephony-twilio.md) |
| Context aggregators and summarization | `src/processors/aggregators` | unit-tested | [Conversation context](../guides/conversation-context.md) |
| Pipelines, `ParallelPipeline`, `ServiceSwitcher`, `PipelineTask` | `src/pipeline` | unit-tested | [Switching services](../guides/switching-services.md) |
| Recording (`audiobuffer`) | `src/processors/audiobuffer` | unit-tested | [Recording](../guides/recording.md) |
| Flows | `src/flows` | unit-tested (fake LLM provider) | [Flows](../guides/flows.md) |
| IVR navigator | `src/extensions/ivr` | untested | [IVR navigation](../guides/ivr-navigation.md) |
| Voicemail detector | `src/extensions/voicemail` | untested | [Voicemail detection](../guides/voicemail-detection.md) |
| Observers: user-to-bot latency, startup timing, turn metrics | `src/pipeline/observers` | unit-tested | [Observability](../guides/observability.md) |
| Observers: turn tracking, metrics, LLM, transcription and debug loggers | `src/pipeline/observers` | untested | [Observability](../guides/observability.md) |
| OpenTelemetry tracing | `src/tracing` | untested | [Observability](../guides/observability.md) |
| RTVI processor and observer | `src/processors/rtvi` | unit-tested | [RTVI](../guides/rtvi.md) |
| Transcript processor | `src/processors/transcript` | untested | [Recording](../guides/recording.md) |

## Next steps

- [LLM providers](llm.md): configure the model that writes the replies.
- [STT providers](stt.md) and [TTS providers](tts.md): pick the speech services.
- [Writing a custom provider](custom.md): add a vendor that is not listed here.
- [Quickstart](../getting-started/quickstart.md): run a bot before you choose.
