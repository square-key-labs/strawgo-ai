package processors

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// FrameLogger is a processor that logs frame information for debugging
// Similar to pipecat's FrameLogger, it intercepts frames and logs their details
type FrameLogger struct {
	*BaseProcessor
	logger             *logger.Logger
	prefix             string
	ignoredFrameTypes  map[reflect.Type]bool
	logDirection       bool
	logFrameDetails    bool
}

// FrameLoggerConfig configures the frame logger
type FrameLoggerConfig struct {
	// Prefix for log messages (e.g., "Pipeline", "STT", "TTS")
	Prefix string

	// IgnoredFrameTypes are frame types to skip logging (e.g., high-frequency audio frames)
	IgnoredFrameTypes []frames.Frame

	// LogDirection includes frame direction (upstream/downstream) in logs
	LogDirection bool

	// LogFrameDetails includes detailed frame information in logs
	LogFrameDetails bool

	// Logger instance to use (if nil, uses default logger)
	Logger *logger.Logger
}

// NewFrameLogger creates a new frame logger processor
func NewFrameLogger(config FrameLoggerConfig) *FrameLogger {
	if config.Prefix == "" {
		config.Prefix = "Frame"
	}

	log := config.Logger
	if log == nil {
		log = logger.GetDefault()
	}

	fl := &FrameLogger{
		logger:            log.WithPrefix(config.Prefix),
		prefix:            config.Prefix,
		ignoredFrameTypes: make(map[reflect.Type]bool),
		logDirection:      config.LogDirection,
		logFrameDetails:   config.LogFrameDetails,
	}

	// Build map of ignored frame types for fast lookup
	for _, frameType := range config.IgnoredFrameTypes {
		fl.ignoredFrameTypes[reflect.TypeOf(frameType)] = true
	}

	fl.BaseProcessor = NewBaseProcessor("FrameLogger:"+config.Prefix, fl)
	return fl
}

// HandleFrame processes and logs frame information
func (fl *FrameLogger) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Check if frame is nil (handle Go's interface nil gotcha)
	if frame == nil || reflect.ValueOf(frame).IsNil() {
		fl.logger.Warn("Received nil frame, skipping")
		return nil
	}

	// Check if this frame type should be ignored
	frameType := reflect.TypeOf(frame)
	if fl.ignoredFrameTypes[frameType] {
		return fl.PushFrame(frame, direction)
	}

	// Only log if debug is enabled
	if fl.logger.IsLevelEnabled(logger.DEBUG) {
		msg := fl.formatFrameLog(frame, direction)
		fl.logger.Debug("%s", msg)
	}

	// Pass frame through
	return fl.PushFrame(frame, direction)
}

func (fl *FrameLogger) formatFrameLog(frame frames.Frame, direction frames.FrameDirection) string {
	dirSymbol := ""
	if fl.logDirection {
		if direction == frames.Downstream {
			dirSymbol = "→ "
		} else {
			dirSymbol = "← "
		}
	}

	frameName := frame.Name()

	if !fl.logFrameDetails {
		return fmt.Sprintf("%s%s", dirSymbol, frameName)
	}

	// Add detailed information if available
	details := fl.extractFrameDetails(frame)
	if details != "" {
		return fmt.Sprintf("%s%s | %s", dirSymbol, frameName, details)
	}

	return fmt.Sprintf("%s%s", dirSymbol, frameName)
}

func (fl *FrameLogger) extractFrameDetails(frame frames.Frame) string {
	// Use reflection to get frame details
	v := reflect.ValueOf(frame)

	// Handle pointers - check for nil before dereferencing
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return ""
		}
		v = v.Elem()
	}

	// Only process structs
	if v.Kind() != reflect.Struct {
		return ""
	}

	t := v.Type()
	var details []string

	// Fields to skip (binary data that would clutter logs)
	skipFields := map[string]bool{
		"audio": true, "Audio": true,
		"data": true, "Data": true,
		"timestamp": true, "Timestamp": true,
	}

	// Extract interesting fields (skip large binary data)
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		fieldType := t.Field(i)

		// Skip unexported fields
		if !field.CanInterface() {
			continue
		}

		// Skip certain field names that would clutter logs
		fieldName := fieldType.Name
		if skipFields[fieldName] {
			continue
		}

		// Format the field value
		var valueStr string
		switch field.Kind() {
		case reflect.String:
			str := field.String()
			if len(str) > 50 {
				str = str[:50] + "..."
			}
			valueStr = fmt.Sprintf("%s: %q", fieldName, str)
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			valueStr = fmt.Sprintf("%s: %d", fieldName, field.Int())
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			valueStr = fmt.Sprintf("%s: %d", fieldName, field.Uint())
		case reflect.Float32, reflect.Float64:
			valueStr = fmt.Sprintf("%s: %.2f", fieldName, field.Float())
		case reflect.Bool:
			valueStr = fmt.Sprintf("%s: %t", fieldName, field.Bool())
		case reflect.Slice, reflect.Array:
			valueStr = fmt.Sprintf("%s: [%d items]", fieldName, field.Len())
		default:
			// For other types, just show the type
			valueStr = fmt.Sprintf("%s: (%s)", fieldName, field.Type().Name())
		}

		if valueStr != "" {
			details = append(details, valueStr)
		}
	}

	if len(details) == 0 {
		return ""
	}

	// Join with commas - use strings.Join for efficiency
	return strings.Join(details, ", ")
}
