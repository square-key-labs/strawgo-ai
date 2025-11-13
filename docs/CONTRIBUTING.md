# Contributing to StrawGo

Thank you for your interest in contributing to StrawGo! We welcome contributions from the community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/strawgo.git
   cd strawgo
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Go 1.21 or higher
- Git
- API keys for testing (optional, see `.env.example`)

### Install Dependencies

```bash
go mod download
```

### Running Examples

```bash
# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run an example
go run examples/text_flow.go
```

## Code Guidelines

### Go Style

- Follow standard Go conventions and idioms
- Run `go fmt` on all code before committing
- Use meaningful variable and function names
- Add comments for exported types, functions, and methods

### Project Structure

```
strawgo/
├── src/                    # Core framework code
│   ├── frames/            # Frame types
│   ├── processors/        # Frame processors
│   ├── pipeline/          # Pipeline orchestration
│   ├── services/          # AI service integrations
│   │   ├── deepgram/     # Deepgram STT
│   │   ├── elevenlabs/   # ElevenLabs TTS
│   │   ├── openai/       # OpenAI LLM
│   │   └── gemini/       # Google Gemini LLM
│   ├── transports/        # Telephony transports
│   └── audio/             # Audio utilities
├── examples/              # Example applications
├── docs/                  # Documentation
└── README.md              # Main documentation
```

### Adding New Features

When adding new features:

1. **Frame Types** - Add to `src/frames/` if creating new frame types
2. **Processors** - Add to `src/processors/` for new frame processors
3. **Services** - Create new subdirectory under `src/services/` for new AI services
4. **Transports** - Add to `src/transports/` for new telephony integrations

### Code Quality

- Ensure your code builds without errors: `go build ./...`
- Run tests (when available): `go test ./...`
- Check for common issues: `go vet ./...`

## Testing

### Manual Testing

Test your changes with the examples:

```bash
# Test basic functionality
go run examples/text_flow.go

# Test with voice (requires API keys)
go run examples/voice_call_mulaw.go
```

### Automated Tests

(Coming soon - we welcome contributions for test coverage!)

## Making Changes

### Commit Messages

Write clear, concise commit messages:

```
Add feature: Brief description

Longer explanation of what changed and why, if needed.
```

Examples:
- `Add Deepgram STT service`
- `Fix audio conversion for PCM pipeline`
- `Update README with Gemini examples`
- `Refactor frame processing for better performance`

### Pull Request Process

1. **Update documentation** - Update README.md and relevant docs if needed
2. **Test your changes** - Ensure examples still work
3. **Create a pull request** with:
   - Clear title describing the change
   - Description of what changed and why
   - Any breaking changes or migration notes
   - Related issue numbers (if applicable)

4. **Address review feedback** - Respond to comments and make requested changes

### Pull Request Template

```markdown
## Summary
Brief description of changes

## Changes
- Added/Fixed/Updated feature X
- Modified Y to support Z

## Testing
How you tested these changes

## Related Issues
Closes #123
```

## Reporting Issues

### Bug Reports

Include:
- Go version (`go version`)
- Operating system
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs
- Relevant code snippets

### Feature Requests

Include:
- Use case description
- Proposed solution or API
- Examples of how it would be used
- Any alternatives considered

## Adding New AI Services

To add a new AI service integration:

1. Create a new directory under `src/services/` (e.g., `src/services/newservice/`)
2. Implement the service following this pattern:

```go
package newservice

import (
    "context"
    "github.com/square-key-labs/strawgo-ai/src/frames"
    "github.com/square-key-labs/strawgo-ai/src/processors"
    "github.com/square-key-labs/strawgo-ai/src/services"
)

type ServiceConfig struct {
    APIKey string
    // Other config fields
}

type Service struct {
    *processors.BaseProcessor
    config ServiceConfig
}

func NewService(config ServiceConfig) *Service {
    s := &Service{config: config}
    s.BaseProcessor = processors.NewBaseProcessor("NewService", s.processFrame)
    return s
}

func (s *Service) Initialize(ctx context.Context) error {
    // Setup service connection
    return nil
}

func (s *Service) processFrame(frame frames.Frame) ([]frames.Frame, error) {
    // Process frames
    return []frames.Frame{frame}, nil
}

func (s *Service) Cleanup() error {
    // Cleanup resources
    return nil
}
```

3. Add an example using the new service
4. Update README.md with the new service
5. Add configuration to `.env.example` if API keys needed

## Adding New Transports

To add a new transport (e.g., for a different telephony provider):

1. Create new file in `src/transports/`
2. Implement the transport interface
3. Handle audio codec metadata properly
4. Support both mulaw and PCM modes if applicable
5. Add example usage
6. Update documentation

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open a [GitHub Discussion](https://github.com/square-key-labs/strawgo-ai/discussions)
- File an [Issue](https://github.com/square-key-labs/strawgo-ai/issues) for bugs
- Check existing documentation in `docs/`

## License

By contributing to StrawGo, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to StrawGo!
