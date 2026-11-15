package processors

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type FrameLogger struct {
	*BaseProcessor
	logger            *logger.Logger
	prefix            string
	ignoredFrameTypes map[reflect.Type]bool
	logDirection      bool
	logFrameDetails   bool
}

type FrameLoggerConfig struct {
	Prefix string

	IgnoredFrameTypes []frames.Frame

	LogDirection bool

	LogFrameDetails bool

	Logger *logger.Logger
}

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

	for _, frameType := range config.IgnoredFrameTypes {
		fl.ignoredFrameTypes[reflect.TypeOf(frameType)] = true
	}

	fl.BaseProcessor = NewBaseProcessor("FrameLogger:"+config.Prefix, fl)
	return fl
}

func (fl *FrameLogger) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if frame == nil || reflect.ValueOf(frame).IsNil() {
		fl.logger.Warn("Received nil frame, skipping")
		return nil
	}

	frameType := reflect.TypeOf(frame)
	if fl.ignoredFrameTypes[frameType] {
		return fl.PushFrame(frame, direction)
	}

	if fl.logger.IsLevelEnabled(logger.DEBUG) {
		msg := fl.formatFrameLog(frame, direction)
		fl.logger.Debug("%s", msg)
	}

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

	details := fl.extractFrameDetails(frame)
	if details != "" {
		return fmt.Sprintf("%s%s | %s", dirSymbol, frameName, details)
	}

	return fmt.Sprintf("%s%s", dirSymbol, frameName)
}

func (fl *FrameLogger) extractFrameDetails(frame frames.Frame) string {
	v := reflect.ValueOf(frame)

	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return ""
		}
		v = v.Elem()
	}

	if v.Kind() != reflect.Struct {
		return ""
	}

	t := v.Type()
	var details []string

	skipFields := map[string]bool{
		"audio": true, "Audio": true,
		"data": true, "Data": true,
		"timestamp": true, "Timestamp": true,
	}

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		fieldType := t.Field(i)

		if !field.CanInterface() {
			continue
		}

		fieldName := fieldType.Name
		if skipFields[fieldName] {
			continue
		}

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
			valueStr = fmt.Sprintf("%s: (%s)", fieldName, field.Type().Name())
		}

		if valueStr != "" {
			details = append(details, valueStr)
		}
	}

	if len(details) == 0 {
		return ""
	}

	return strings.Join(details, ", ")
}
