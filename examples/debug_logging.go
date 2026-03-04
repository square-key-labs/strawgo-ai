package main

import (
	"context"
	"fmt"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// This example demonstrates how to enable and use debug logging in StrawGo
func main() {
	fmt.Println("=== StrawGo Debug Logging Example ===\n")

	// Initialize the logger (reads LOG_LEVEL and LOG_COLOR from environment)
	logger.Init()

	// You can also set the log level programmatically
	// Uncomment to enable debug logging:
	// logger.SetLevel(logger.DEBUG)

	// Show current log level
	currentLevel := logger.GetLevel()
	levelNames := map[logger.LogLevel]string{
		logger.DEBUG: "DEBUG",
		logger.INFO:  "INFO",
		logger.WARN:  "WARN",
		logger.ERROR: "ERROR",
	}
	fmt.Printf("Current log level: %s\n", levelNames[currentLevel])
	fmt.Printf("Debug enabled: %v\n\n", logger.IsDebugEnabled())

	// Example 1: Basic logging at different levels
	fmt.Println("--- Example 1: Basic Logging ---")
	logger.Debug("This is a debug message (only visible when LOG_LEVEL=DEBUG)")
	logger.Info("This is an info message")
	logger.Warn("This is a warning message")
	logger.Error("This is an error message")
	fmt.Println()

	// Example 2: Using prefixed loggers for different components
	fmt.Println("--- Example 2: Component-Specific Loggers ---")
	sttLogger := logger.WithPrefix("STT")
	llmLogger := logger.WithPrefix("LLM")
	ttsLogger := logger.WithPrefix("TTS")

	sttLogger.Info("Speech-to-text service initialized")
	llmLogger.Info("Language model service initialized")
	ttsLogger.Info("Text-to-speech service initialized")
	sttLogger.Debug("Processing audio chunk: 1024 samples")
	fmt.Println()

	// Example 3: Using FrameLogger in a pipeline for debugging
	fmt.Println("--- Example 3: Frame Logger in Pipeline ---")
	fmt.Println("To see frame-by-frame debugging, set LOG_LEVEL=DEBUG")
	fmt.Println()

	// Create a simple processor
	simpleProc := processors.NewPassthroughProcessor("ExampleProcessor", false)

	// Create a frame logger (only logs when debug is enabled)
	frameLogger := processors.NewFrameLogger(processors.FrameLoggerConfig{
		Prefix:          "PipelineDebug",
		LogDirection:    true,  // Show upstream/downstream arrows
		LogFrameDetails: true,  // Show detailed frame information
		// IgnoredFrameTypes: []frames.Frame{
		// 	// Add frame types to ignore, e.g., high-frequency audio frames
		// },
	})

	// Build a pipeline with the frame logger
	// Note: This is just for demonstration - in a real app, you'd have actual processors
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		frameLogger,   // Logs all frames passing through
		simpleProc,
	})

	// Initialize the pipeline (normally you'd run it with task.Run(ctx))
	task := pipeline.NewPipelineTask(pipe)

	// Example 4: Conditional debug logging
	fmt.Println("--- Example 4: Conditional Debug Logging ---")
	if logger.IsDebugEnabled() {
		logger.Debug("Debug mode is active - verbose logging enabled")
		// Perform expensive debug operations only when needed
		debugInfo := gatherDebugInfo()
		logger.Debug("System info: %s", debugInfo)
	} else {
		fmt.Println("Debug logging is disabled. Set LOG_LEVEL=DEBUG to enable.")
	}
	fmt.Println()

	// Example 5: Environment variable configuration
	fmt.Println("--- Example 5: Configuration ---")
	fmt.Println("Set these environment variables to configure logging:")
	fmt.Println("  export LOG_LEVEL=DEBUG    # Enable detailed debug logs")
	fmt.Println("  export LOG_LEVEL=INFO     # Standard logging (default)")
	fmt.Println("  export LOG_LEVEL=WARN     # Only warnings and errors")
	fmt.Println("  export LOG_LEVEL=ERROR    # Only errors")
	fmt.Println("  export LOG_COLOR=false    # Disable colored output")
	fmt.Println()
	fmt.Println("Or use a .env file (copy .env.example to .env and edit)")
	fmt.Println()

	// Clean up
	ctx := context.Background()
	_ = task
	_ = ctx

	fmt.Println("=== Example Complete ===")
	fmt.Println("\nTry running with DEBUG logging:")
	fmt.Println("  LOG_LEVEL=DEBUG go run examples/debug_logging.go")
}

func gatherDebugInfo() string {
	hostname, _ := os.Hostname()
	return fmt.Sprintf("hostname=%s, pid=%d", hostname, os.Getpid())
}
