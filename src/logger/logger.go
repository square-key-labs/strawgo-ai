package logger

import (
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
)

// LogLevel represents the severity level of a log message
type LogLevel int

const (
	// DEBUG level for detailed debugging information
	DEBUG LogLevel = iota
	// INFO level for general informational messages
	INFO
	// WARN level for warning messages
	WARN
	// ERROR level for error messages
	ERROR
)

var (
	levelNames = map[LogLevel]string{
		DEBUG: "DEBUG",
		INFO:  "INFO",
		WARN:  "WARN",
		ERROR: "ERROR",
	}

	levelColors = map[LogLevel]string{
		DEBUG: "\033[36m", // Cyan
		INFO:  "\033[32m", // Green
		WARN:  "\033[33m", // Yellow
		ERROR: "\033[31m", // Red
	}
)

// Logger provides configurable logging with different log levels.
// Child loggers created via WithPrefix share the parent's level state
// through a pointer, so SetLevel on the parent propagates to children.
type Logger struct {
	parent        *Logger      // nil for root loggers, set for WithPrefix children
	mu            sync.RWMutex // protects level + enabledLevels on root loggers
	level         LogLevel
	output        io.Writer
	enableColors  bool
	prefix        string
	stdLogger     *log.Logger
	enabledLevels map[LogLevel]bool
}

var (
	defaultLogger *Logger
	once          sync.Once
)

// Init initializes the default logger with configuration from environment variables
// Environment variables:
//   - LOG_LEVEL: Set log level (DEBUG, INFO, WARN, ERROR). Default: INFO
//   - LOG_COLOR: Enable colored output (true/false). Default: true
func Init() {
	once.Do(func() {
		level := INFO
		levelStr := strings.ToUpper(os.Getenv("LOG_LEVEL"))
		switch levelStr {
		case "DEBUG":
			level = DEBUG
		case "INFO":
			level = INFO
		case "WARN", "WARNING":
			level = WARN
		case "ERROR":
			level = ERROR
		}

		enableColors := true
		if colorStr := os.Getenv("LOG_COLOR"); colorStr == "false" || colorStr == "0" {
			enableColors = false
		}

		defaultLogger = New(level, os.Stdout, enableColors, "")
	})
}

// New creates a new Logger instance
func New(level LogLevel, output io.Writer, enableColors bool, prefix string) *Logger {
	l := &Logger{
		level:         level,
		output:        output,
		enableColors:  enableColors,
		prefix:        prefix,
		stdLogger:     log.New(output, "", log.LstdFlags),
		enabledLevels: make(map[LogLevel]bool),
	}

	// Enable all levels >= configured level
	for lvl := DEBUG; lvl <= ERROR; lvl++ {
		l.enabledLevels[lvl] = lvl >= level
	}

	return l
}

// SetLevel changes the current log level.
// If this logger was created via WithPrefix, SetLevel updates the root logger
// so all sibling prefixed loggers also see the change.
func (l *Logger) SetLevel(level LogLevel) {
	root := l.root()
	root.mu.Lock()
	defer root.mu.Unlock()
	root.level = level
	for lvl := DEBUG; lvl <= ERROR; lvl++ {
		root.enabledLevels[lvl] = lvl >= level
	}
}

// GetLevel returns the current log level.
func (l *Logger) GetLevel() LogLevel {
	root := l.root()
	root.mu.RLock()
	defer root.mu.RUnlock()
	return root.level
}

// IsLevelEnabled checks if a specific log level is enabled.
func (l *Logger) IsLevelEnabled(level LogLevel) bool {
	root := l.root()
	root.mu.RLock()
	defer root.mu.RUnlock()
	return root.enabledLevels[level]
}

func (l *Logger) log(level LogLevel, format string, args ...interface{}) {
	if !l.IsLevelEnabled(level) {
		return
	}

	msg := fmt.Sprintf(format, args...)
	levelName := levelNames[level]

	var output string
	if l.enableColors {
		color := levelColors[level]
		reset := "\033[0m"
		if l.prefix != "" {
			output = fmt.Sprintf("%s[%s]%s [%s] %s", color, levelName, reset, l.prefix, msg)
		} else {
			output = fmt.Sprintf("%s[%s]%s %s", color, levelName, reset, msg)
		}
	} else {
		if l.prefix != "" {
			output = fmt.Sprintf("[%s] [%s] %s", levelName, l.prefix, msg)
		} else {
			output = fmt.Sprintf("[%s] %s", levelName, msg)
		}
	}

	l.stdLogger.Output(2, output)
}

// Debug logs a debug message
func (l *Logger) Debug(format string, args ...interface{}) {
	l.log(DEBUG, format, args...)
}

// Info logs an info message
func (l *Logger) Info(format string, args ...interface{}) {
	l.log(INFO, format, args...)
}

// Warn logs a warning message
func (l *Logger) Warn(format string, args ...interface{}) {
	l.log(WARN, format, args...)
}

// Error logs an error message
func (l *Logger) Error(format string, args ...interface{}) {
	l.log(ERROR, format, args...)
}

// WithPrefix creates a child logger that shares the parent's level state.
// Calling SetLevel on either the parent or child updates both.
func (l *Logger) WithPrefix(prefix string) *Logger {
	root := l.root()
	return &Logger{
		parent:        root,
		level:         root.level,
		output:        root.output,
		enableColors:  root.enableColors,
		prefix:        prefix,
		stdLogger:     root.stdLogger,
		enabledLevels: root.enabledLevels, // shared with root — all access goes through root.mu
	}
}

// Global convenience functions that use the default logger

// GetDefault returns the default logger instance.
// Init is idempotent (sync.Once) so it is safe to call from multiple goroutines.
func GetDefault() *Logger {
	Init()
	return defaultLogger
}

// SetLevel sets the log level for the default logger
func SetLevel(level LogLevel) {
	GetDefault().SetLevel(level)
}

// GetLevel returns the current log level of the default logger
func GetLevel() LogLevel {
	return GetDefault().GetLevel()
}

// IsDebugEnabled checks if debug logging is enabled
func IsDebugEnabled() bool {
	return GetDefault().IsLevelEnabled(DEBUG)
}

// Debug logs a debug message using the default logger
func Debug(format string, args ...interface{}) {
	GetDefault().log(DEBUG, format, args...)
}

// Info logs an info message using the default logger
func Info(format string, args ...interface{}) {
	GetDefault().log(INFO, format, args...)
}

// Warn logs a warning message using the default logger
func Warn(format string, args ...interface{}) {
	GetDefault().log(WARN, format, args...)
}

// Error logs an error message using the default logger
func Error(format string, args ...interface{}) {
	GetDefault().log(ERROR, format, args...)
}

// WithPrefix creates a new logger with a prefix from the default logger
func WithPrefix(prefix string) *Logger {
	return GetDefault().WithPrefix(prefix)
}

// root returns the root logger (self if no parent).
func (l *Logger) root() *Logger {
	if l.parent != nil {
		return l.parent
	}
	return l
}

// ParseLevel converts a string to a LogLevel.
// Accepts: "debug", "info", "warn", "warning", "error" (case-insensitive).
// Returns an error for unknown values.
func ParseLevel(s string) (LogLevel, error) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "DEBUG":
		return DEBUG, nil
	case "INFO":
		return INFO, nil
	case "WARN", "WARNING":
		return WARN, nil
	case "ERROR":
		return ERROR, nil
	default:
		return INFO, fmt.Errorf("unknown log level: %q (valid: debug, info, warn, error)", s)
	}
}

// SetLogLevel sets the default logger's level from a string.
// Accepts: "debug", "info", "warn", "warning", "error" (case-insensitive).
// Returns an error for unknown values (level unchanged).
func SetLogLevel(s string) error {
	lvl, err := ParseLevel(s)
	if err != nil {
		return err
	}
	GetDefault().SetLevel(lvl)
	return nil
}
