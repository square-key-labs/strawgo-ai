# Debug Logging in StrawGo

StrawGo includes a comprehensive logging system inspired by [Pipecat](https://github.com/pipecat-ai/pipecat)'s logging framework. This guide explains how to enable and use debug logging to troubleshoot and monitor your voice pipelines.

## Quick Start

### 1. Enable Debug Logging

The easiest way to enable debug logging is via environment variables:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run your application
go run examples/voice_call_complete.go
```

Or using a `.env` file:

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and set:
LOG_LEVEL=DEBUG
LOG_COLOR=true
```

### 2. Run the Debug Example

```bash
# With INFO level (default)
go run examples/debug_logging.go

# With DEBUG level
LOG_LEVEL=DEBUG go run examples/debug_logging.go
```

## Log Levels

StrawGo supports four log levels (in order of severity):

| Level | Description | Use Case |
|-------|-------------|----------|
| `DEBUG` | Detailed debugging information | Development, troubleshooting frame flow |
| `INFO` | General informational messages | Production logging (default) |
| `WARN` | Warning messages | Potential issues |
| `ERROR` | Error messages | Errors and failures |

### Setting Log Level

#### Via Environment Variables

```bash
export LOG_LEVEL=DEBUG    # Enable all logs
export LOG_LEVEL=INFO     # Standard logging (default)
export LOG_LEVEL=WARN     # Only warnings and errors
export LOG_LEVEL=ERROR    # Only errors
```

#### Programmatically

```go
import "github.com/square-key-labs/strawgo-ai/src/logger"

func main() {
    // Initialize logger (reads from environment)
    logger.Init()

    // Or set level programmatically
    logger.SetLevel(logger.DEBUG)

    // Check current level
    if logger.IsDebugEnabled() {
        logger.Debug("Debug mode active")
    }
}
```

## Basic Logging

### Simple Logging

```go
import "github.com/square-key-labs/strawgo-ai/src/logger"

logger.Debug("Detailed debug info: %s", debugData)
logger.Info("Application started successfully")
logger.Warn("Connection timeout, retrying...")
logger.Error("Failed to connect: %v", err)
```

### Component-Specific Loggers

Create prefixed loggers for different components:

```go
// Create component-specific loggers
sttLogger := logger.WithPrefix("STT")
llmLogger := logger.WithPrefix("LLM")
ttsLogger := logger.WithPrefix("TTS")

sttLogger.Info("Speech-to-text service initialized")
llmLogger.Debug("Processing user input: %s", text)
ttsLogger.Warn("Audio buffer full, dropping frames")
```

Output:
```
2024/01/15 10:30:45 [INFO] [STT] Speech-to-text service initialized
2024/01/15 10:30:46 [DEBUG] [LLM] Processing user input: "Hello"
2024/01/15 10:30:47 [WARN] [TTS] Audio buffer full, dropping frames
```

## Frame Logging

The `FrameLogger` processor allows you to monitor frames flowing through your pipeline - similar to Pipecat's debug observer.

### Basic Frame Logger

```go
import (
    "github.com/square-key-labs/strawgo-ai/src/logger"
    "github.com/square-key-labs/strawgo-ai/src/pipeline"
    "github.com/square-key-labs/strawgo-ai/src/processors"
)

// Create frame logger
frameLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
    Prefix:          "Pipeline",
    LogDirection:    true,  // Show → (downstream) or ← (upstream)
    LogFrameDetails: true,  // Include frame field values
})

// Add to pipeline
pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    transport.Input(),
    frameLogger,        // Logs all frames at this point
    sttService,
    llmService,
    ttsService,
    transport.Output(),
})
```

### Filtering Frames

Ignore high-frequency frames (like audio) to reduce noise:

```go
import (
    "github.com/square-key-labs/strawgo-ai/src/frames"
    "github.com/square-key-labs/strawgo-ai/src/processors"
)

frameLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
    Prefix: "Debug",
    IgnoredFrameTypes: []frames.Frame{
        &frames.AudioFrame{},      // Ignore audio frames
        &frames.VideoFrame{},      // Ignore video frames
    },
    LogDirection:    true,
    LogFrameDetails: true,
})
```

### Multiple Frame Loggers

Add multiple loggers at different pipeline stages:

```go
inputLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
    Prefix: "Input",
})

processingLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
    Prefix: "Processing",
})

outputLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
    Prefix: "Output",
})

pipe := pipeline.NewPipeline([]processors.FrameProcessor{
    transport.Input(),
    inputLogger,           // Log incoming frames
    sttService,
    processingLogger,      // Log after STT
    llmService,
    ttsService,
    outputLogger,          // Log before output
    transport.Output(),
})
```

## Conditional Debug Logging

Only perform expensive operations when debug is enabled:

```go
if logger.IsDebugEnabled() {
    // This expensive operation only runs in debug mode
    debugInfo := gatherDetailedMetrics()
    logger.Debug("Detailed metrics: %v", debugInfo)
}
```

## Configuration Options

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARN`, `ERROR` | `INFO` | Minimum log level to display |
| `LOG_COLOR` | `true`, `false` | `true` | Enable/disable colored output |

### Color Output

By default, logs are color-coded by level:
- 🔵 **DEBUG** - Cyan
- 🟢 **INFO** - Green
- 🟡 **WARN** - Yellow
- 🔴 **ERROR** - Red

Disable colors for log files or non-TTY output:

```bash
export LOG_COLOR=false
```

## Best Practices

### 1. Use Appropriate Log Levels

```go
// ❌ Don't use Debug for important messages
logger.Debug("User authentication failed")

// ✅ Use Error for failures
logger.Error("User authentication failed: %v", err)

// ✅ Use Debug for detailed information
logger.Debug("Auth token validated, user_id=%s", userID)
```

### 2. Add Context with Prefixes

```go
// Create service-specific loggers
type MyService struct {
    logger *logger.Logger
}

func NewMyService() *MyService {
    return &MyService{
        logger: logger.WithPrefix("MyService"),
    }
}

func (s *MyService) Process() {
    s.logger.Info("Processing started")
    s.logger.Debug("Step 1 complete")
}
```

### 3. Use Frame Loggers Selectively

```go
// ✅ Good: Add frame logger only when debugging
if logger.IsDebugEnabled() {
    frameLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
        Prefix: "Debug",
    })
    processorList = append(processorList, frameLogger)
}
```

### 4. Avoid Logging Sensitive Data

```go
// ❌ Don't log sensitive information
logger.Debug("API Key: %s", apiKey)

// ✅ Log safely
logger.Debug("API Key configured: %t", apiKey != "")
```

## Debugging Common Issues

### Pipeline Not Processing Frames

Enable frame-level debugging:

```bash
LOG_LEVEL=DEBUG go run your_app.go
```

Look for:
- Frame flow through processors
- Processor start/stop messages
- Frame queue activity

### High CPU Usage

Check for frame logging overhead:

```go
// Disable frame details in production
frameLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
    Prefix:          "Pipeline",
    LogDirection:    false,  // Disable direction symbols
    LogFrameDetails: false,  // Disable detailed inspection
})
```

### Missing Logs

Ensure logger is initialized:

```go
import "github.com/square-key-labs/strawgo-ai/src/logger"

func main() {
    logger.Init()  // Initialize before any logging

    // Rest of your code
}
```

## Comparison with Pipecat

StrawGo's logging is inspired by [Pipecat's logging framework](https://github.com/pipecat-ai/pipecat):

| Feature | Pipecat (Python) | StrawGo (Go) |
|---------|------------------|--------------|
| Log levels | ✅ DEBUG, INFO, WARN, ERROR | ✅ DEBUG, INFO, WARN, ERROR |
| Frame logging | ✅ `FrameLogger` | ✅ `FrameLogger` |
| Colored output | ✅ Via loguru | ✅ ANSI colors |
| Frame filtering | ✅ `ignored_frame_types` | ✅ `IgnoredFrameTypes` |
| Component prefixes | ✅ Custom prefixes | ✅ `WithPrefix()` |
| Debug observers | ✅ Observer pattern | ✅ Frame logger processors |
| Environment config | ✅ Python env | ✅ Go env vars |

## Examples

See [`examples/debug_logging.go`](../examples/debug_logging.go) for a complete working example.

## Further Reading

- [Pipecat Logging Documentation](https://docs.pipecat.ai/)
- [Go log package](https://pkg.go.dev/log)
- StrawGo API Documentation
