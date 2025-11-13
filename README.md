# 🍓 StrawGo

> A high-performance, Go-based framework for building real-time conversational AI applications with voice calling support.

[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go)](https://go.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)

StrawGo is a production-ready framework inspired by [Pipecat](https://github.com/pipecat-ai/pipecat), designed specifically for Go developers who need to build low-latency, scalable voice AI applications.

## ✨ Features

- 🎯 **Frame-Based Architecture** - Clean, composable pipeline system
- ⚡ **High Performance** - Native Go concurrency with goroutines and channels
- 📞 **Voice Calling** - Built-in support for Twilio and Asterisk WebSocket
- 🎙️ **Multiple AI Services** - Deepgram STT, ElevenLabs TTS, OpenAI & Gemini LLMs
- 🔄 **Flexible Audio Processing** - Choose between mulaw passthrough or PCM pipeline
- 🚀 **Production Ready** - Comprehensive error handling and lifecycle management
- 📦 **Zero External Dependencies** (except gorilla/websocket)

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
    "github.com/square-key-labs/strawgo-ai/src/services/deepgram"
    "github.com/square-key-labs/strawgo-ai/src/services/elevenlabs"
    "github.com/square-key-labs/strawgo-ai/src/services/openai"
    "github.com/square-key-labs/strawgo-ai/src/transports"
)

func main() {
    // Create transport
    twilio := transports.NewTwilioWebSocketTransport(transports.TwilioWebSocketConfig{
        Port: 8080,
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
        twilio.Input(),
        stt,
        llm,
        tts,
        twilio.Output(),
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
│   ├── pipeline/            # Pipeline orchestration
│   ├── services/            # AI service integrations
│   │   ├── deepgram/       # Deepgram STT
│   │   ├── elevenlabs/     # ElevenLabs TTS
│   │   ├── openai/         # OpenAI LLM
│   │   └── gemini/         # Google Gemini LLM
│   ├── transports/          # Telephony transports
│   └── audio/               # Audio conversion utilities
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
- **OpenAI**: [Get API key](https://platform.openai.com/api-keys)
- **Google Gemini**: [Get API key](https://ai.google.dev/)

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
