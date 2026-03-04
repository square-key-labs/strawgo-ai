# 🍓 StrawGo

> A high-performance, Go-based framework for building real-time conversational AI applications with voice calling support.

[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

StrawGo is a production-ready framework inspired by [Pipecat](https://github.com/pipecat-ai/pipecat), designed specifically for Go developers who need to build low-latency, scalable voice AI applications.

## ✨ Features

- 🎯 **Frame-Based Architecture** - Clean, composable pipeline system
- ⚡ **High Performance** - Native Go concurrency with goroutines and channels
- 📞 **Voice Calling** - Built-in support for Twilio, Asterisk WebSocket, and Daily WebRTC
- 🔌 **Transport/Serializer Pattern** - Extensible architecture inspired by pipecat
- 🎙️ **16 AI Services** - Deepgram STT/TTS, ElevenLabs TTS, Cartesia TTS, OpenAI LLM, Gemini LLM/Live, Groq LLM, Whisper STT, Google TTS, Azure STT/TTS, Anthropic Claude LLM, Ollama LLM, AssemblyAI STT, OpenAI Realtime STT, OpenAI Realtime S2S
- 🎛️ **Turn Strategies** - Composable turn management system (start/stop/mute strategies)
- 🔄 **Flexible Audio Processing** - Choose between mulaw passthrough or PCM pipeline
- 🚀 **Production Ready** - Comprehensive error handling and lifecycle management
- 📦 **Minimal Dependencies** - Only requires gorilla/websocket

## 🚀 Quick Start

### Installation

```bash
go get github.com/square-key-labs/strawgo-ai
```

### Basic Text Pipeline

```go
package main

import (
    "context"
    "github.com/square-key-labs/strawgo-ai/src/pipeline"
    "github.com/square-key-labs/strawgo-ai/src/processors"
)

func main() {
    // Create processors
    generator := processors.NewTextGeneratorProcessor([]string{"Hello", "World"})
    printer := processors.NewTextPrinterProcessor()

    // Build pipeline
    pipe := pipeline.NewPipeline([]processors.FrameProcessor{
        generator,
        printer,
    })

    // Run
    task := pipeline.NewPipelineTask(pipe)
    task.Run(context.Background())
}
```

### Voice Call with Twilio

```go
package main

import (
    "github.com/square-key-labs/strawgo-ai/src/pipeline"
    "github.com/square-key-labs/strawgo-ai/src/serializers"
    "github.com/square-key-labs/strawgo-ai/src/services/deepgram"
    "github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
    "github.com/square-key-labs/strawgo-ai/src/services/openai"
    "github.com/square-key-labs/strawgo-ai/src/transports"
)

func main() {
    // Create Twilio serializer (handles Twilio Media Streams protocol)
    twilioSerializer := serializers.NewTwilioFrameSerializer("", "")

    // Create WebSocket transport with Twilio serializer
    transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
        Port:       8080,
        Path:       "/media",
        Serializer: twilioSerializer,
    })

    // Create AI services (mulaw passthrough - zero conversions!)
    stt := deepgram.NewSTTService(deepgram.STTConfig{
        APIKey:   os.Getenv("DEEPGRAM_API_KEY"),
        Encoding: "mulaw", // No conversion needed!
    })

    llm := openai.NewLLMService(openai.LLMConfig{
        APIKey: os.Getenv("OPENAI_API_KEY"),
        Model:  "gpt-4-turbo",
    })

    tts := elevenlabs.NewTTSService(elevenlabs.TTSConfig{
        APIKey:       os.Getenv("ELEVENLABS_API_KEY"),
        OutputFormat: "ulaw_8000", // Direct mulaw output!
    })

    // Build pipeline
    pipe := pipeline.NewPipeline([]processors.FrameProcessor{
        transport.Input(),
        stt,
        llm,
        tts,
        transport.Output(),
    })

    // Run
    task := pipeline.NewPipelineTask(pipe)
    task.Run(context.Background())
}
```

See [examples/](examples/) for more complete examples.

## 📁 Project Structure

```
strawgo/
├── src/
│   ├── frames/              # Frame types (system/data/control)
│   ├── processors/          # Frame processors
│   │   └── aggregators/    # LLM context and sentence aggregators
│   ├── pipeline/            # Pipeline orchestration
│   ├── serializers/         # Protocol serializers (Twilio, Asterisk)
│   ├── services/            # AI service integrations
│   │   ├── deepgram/       # Deepgram STT/TTS
│   │   ├── elevenlabs/     # ElevenLabs TTS
│   │   ├── cartesia/       # Cartesia TTS
│   │   ├── openai/         # OpenAI LLM
│   │   ├── gemini/         # Google Gemini LLM/Live
│   │   ├── groq/           # Groq LLM
│   │   ├── whisper/        # Whisper STT
│   │   ├── google/         # Google Cloud TTS
│   │   └── azure/          # Azure Speech STT/TTS
│   ├── transports/          # Network transports
│   │   ├── websocket/      # WebSocket transport
│   │   └── daily/          # Daily WebRTC transport
│   ├── turns/               # Turn management strategies
│   │   ├── user_start/     # Turn start strategies
│   │   ├── user_stop/      # Turn stop strategies
│   │   └── user_mute/      # Turn mute strategies
│   └── audio/               # Audio conversion and VAD utilities
├── examples/                # Example applications
├── docs/                    # Documentation
├── .env.example             # Example environment variables
├── go.mod                   # Go module definition
├── LICENSE                  # MIT License
└── README.md                # This file
```

## 🎯 Use Cases

- **Voice Bots** - Build AI phone assistants with Twilio or Asterisk
- **Customer Support** - Automated voice response systems
- **Voice AI Apps** - Real-time conversational applications
- **IVR Systems** - Interactive Voice Response with AI
- **Call Centers** - AI-powered call routing and assistance

## 📖 Documentation

- **[Audio Strategy Guide](docs/AUDIO_STRATEGY.md)** - Choosing between mulaw passthrough vs PCM pipeline
- **[API Reference](https://pkg.go.dev/github.com/square-key-labs/strawgo-ai)** - Complete API documentation
- **[Examples](examples/)** - Working code examples
- **[Architecture](docs/BUILD_STATUS.md)** - Framework architecture and design

## 🔧 Configuration

### Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit with your API keys
DEEPGRAM_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### API Keys

- **Deepgram**: [Get API key](https://console.deepgram.com/)
- **ElevenLabs**: [Get API key](https://elevenlabs.io/)
- **Cartesia**: [Get API key](https://cartesia.ai/)
- **OpenAI**: [Get API key](https://platform.openai.com/api-keys)
- **Google Gemini**: [Get API key](https://ai.google.dev/)
- **Groq**: [Get API key](https://console.groq.com/)
- **Google Cloud**: [Get API key](https://cloud.google.com/text-to-speech)
- **Azure**: [Get API key](https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/)
- **Daily**: [Get API key](https://www.daily.co/)

## 🚦 Audio Processing Strategies

StrawGo offers **two approaches** for handling audio:

### ⚡ Mulaw Passthrough (Recommended for Telephony)
- **Zero audio conversions** - mulaw stays mulaw throughout
- **Best performance** - lowest latency and CPU usage
- **Use when**: Building pure telephony apps (Twilio, Asterisk)

### 🔧 PCM Pipeline (Maximum Flexibility)
- **Standard PCM processing** - convert mulaw ↔ PCM as needed
- **Audio processing** - add filters, VAD, mixing, resampling
- **Use when**: Need audio processing features or flexibility

See the [Audio Strategy Guide](docs/AUDIO_STRATEGY.md) for detailed comparison.

## 🔌 Transport & Serializer Architecture

StrawGo follows the **pipecat design pattern** of separating network transports from protocol serializers:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  WebSocket Transport                     │
│              (Generic, Protocol-Agnostic)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Dependency Injection
                     │
         ┌───────────┴──────────┐
         │                      │
    ┌────▼─────┐          ┌────▼─────┐
    │  Twilio  │          │ Asterisk │
    │Serializer│          │Serializer│
    └──────────┘          └──────────┘
```

### How It Works

**1. WebSocket Transport** - Generic network layer
- Handles WebSocket connections
- Manages message routing
- Protocol-agnostic

**2. Serializers** - Protocol-specific adapters
- Convert frames ↔ protocol messages
- Handle Twilio/Asterisk/etc. formats
- Injected into transport

### Example: Twilio

```go
// Create protocol serializer
twilioSerializer := serializers.NewTwilioFrameSerializer("", "")

// Inject into generic transport
transport := transports.NewWebSocketTransport(transports.WebSocketConfig{
    Port:       8080,
    Serializer: twilioSerializer, // Dependency injection
})

// Use in pipeline
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    transport.Input(),  // Deserializes Twilio → Frames
    deepgramSTT,
    openaiLLM,
    elevenLabsTTS,
    transport.Output(), // Serializes Frames → Twilio
})
```

### Benefits

- **Separation of Concerns** - Transport doesn't know about protocols
- **Extensibility** - Add new telephony providers by creating serializers
- **Reusability** - One transport works with all providers
- **Testability** - Test transport and serializers independently

### Supported Serializers

- **TwilioFrameSerializer** - Twilio Media Streams (JSON/Text)
- **AsteriskFrameSerializer** - Asterisk WebSocket (Binary/JSON)
- **Custom Serializers** - Easy to add (Telnyx, Plivo, etc.)

### Supported Transports

- **WebSocket Transport** - Generic WebSocket server with serializer injection
- **Daily WebRTC Transport** - Peer-to-peer audio via Daily.co platform

## 🎛️ Turn Strategies

StrawGo v0.0.11 introduces a composable turn management system that replaces the old interruption system. Turn strategies control when user turns start, stop, and when the bot should be muted.

### Strategy Types

**Start Strategies** - When does a user turn begin?
- `VADUserTurnStartStrategy` - Start on voice activity detection
- `TranscriptionUserTurnStartStrategy` - Start on first transcribed word
- `MinWordsUserTurnStartStrategy` - Start after N words

**Stop Strategies** - When does a user turn end?
- `SpeechTimeoutUserTurnStopStrategy` - End after silence timeout
- `TurnAnalyzerUserTurnStopStrategy` - ML-based end-of-turn detection

**Mute Strategies** - When should the bot suppress interruptions?
- `FirstSpeechUserMuteStrategy` - Mute until first bot speech
- `FunctionCallUserMuteStrategy` - Mute during function calls
- `AlwaysUserMuteStrategy` - Never allow interruptions

### Example Configuration

```go
import (
    "time"
    "github.com/square-key-labs/strawgo-ai/src/turns"
    "github.com/square-key-labs/strawgo-ai/src/turns/user_start"
    "github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
    "github.com/square-key-labs/strawgo-ai/src/turns/user_mute"
)

turnStrategies := turns.UserTurnStrategies{
    StartStrategies: []turns.UserTurnStartStrategy{
        user_start.NewVADUserTurnStartStrategy(true), // Enable interruptions
    },
    StopStrategies: []turns.UserTurnStopStrategy{
        user_stop.NewSpeechTimeoutUserTurnStopStrategy(2 * time.Second),
    },
    MuteStrategies: []turns.UserMuteStrategy{
        user_mute.NewFirstSpeechUserMuteStrategy(),
    },
}

// Use in pipeline configuration
taskConfig := pipeline.PipelineTaskConfig{
    TurnStrategies: turnStrategies,
    // ... other config
}
```

## 🎙️ Supported AI Services

### Speech-to-Text (STT)
- **Deepgram STT** - WebSocket streaming transcription
- **Whisper STT** - Batch-mode transcription via OpenAI API
- **Azure Speech STT** - Azure Cognitive Services speech recognition
- **AssemblyAI STT** - Real-time STT with built-in turn detection (`u3-rt-pro` model)

### Text-to-Speech (TTS)
- **ElevenLabs TTS** - HTTP streaming with voice cloning
- **Cartesia TTS** - WebSocket streaming with Sonic-3 model
- **Deepgram TTS** - WebSocket streaming text-to-speech
- **Google Cloud TTS** - HTTP-based text-to-speech
- **Azure Speech TTS** - Azure Cognitive Services speech synthesis

### Language Models (LLM)
- **OpenAI LLM** - Chat completions with streaming (GPT-4, GPT-3.5)
- **Gemini LLM** - Google's text-mode language model
- **Groq LLM** - Fast inference with OpenAI-compatible API
- **Anthropic Claude LLM** - Claude models with streaming and tool calling
- **Ollama LLM** - Local model hosting with OpenAI-compatible API

### Multimodal Speech-to-Speech
- **Gemini Live** - Google's multimodal live S2S (replaces STT+LLM+TTS pipeline)
- **OpenAI Realtime STT** - OpenAI Realtime API STT with local and server VAD modes
- **OpenAI Realtime LLM S2S** - OpenAI Realtime API speech-to-speech (audio in → STT+LLM+TTS frames out)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance

| Metric | Mulaw Passthrough | PCM Pipeline |
|--------|------------------|--------------|
| CPU Usage | ~5% | ~15% |
| Latency | ~50ms | ~70ms |
| Memory | Low | Medium |
| Flexibility | Limited | High |

*Benchmarks approximate, based on single call with Twilio + Deepgram + OpenAI + ElevenLabs*

## 🌟 Why StrawGo?

- **🚀 Go Performance** - Native Go concurrency, no Python GIL
- **📦 Production Ready** - Battle-tested patterns from pipecat
- **🎯 Telephony First** - Built specifically for voice calls
- **⚡ Zero Conversion Option** - Unique mulaw passthrough mode
- **🔧 Flexible** - Use what you need, when you need it
- **📚 Well Documented** - Comprehensive guides and examples

## 🙏 Acknowledgments

- Inspired by [Pipecat](https://github.com/pipecat-ai/pipecat) by Pipecat AI
- Built for the Go community with ❤️

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issues**: [GitHub Issues](https://github.com/square-key-labs/strawgo-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/square-key-labs/strawgo-ai/discussions)

---

<p align="center">
Built with 🍓 by the StrawGo community
</p>
