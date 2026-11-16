package frameinfo

import (
	"reflect"
	"time"
	"unicode"
)

type named interface {
	Name() string
}

type finalProvider interface {
	IsTranscriptionFinal() bool
}

type latencyProvider interface {
	GetTTFSP99Latency() time.Duration
}

func Name(frame any) string {
	if n, ok := frame.(named); ok {
		return n.Name()
	}
	return ""
}

func IsUserStarted(name string) bool {
	return name == "UserStartedSpeakingFrame" || name == "VADUserStartedSpeakingFrame"
}

func IsUserStopped(name string) bool {
	return name == "UserStoppedSpeakingFrame" || name == "VADUserStoppedSpeakingFrame"
}

func IsTranscription(name string) bool {
	return name == "TranscriptionFrame" || name == "InterimTranscriptionFrame"
}

func IsFinal(frame any) bool {
	if Name(frame) == "InterimTranscriptionFrame" {
		return false
	}
	if f, ok := frame.(finalProvider); ok {
		return f.IsTranscriptionFinal()
	}
	return true
}

func P99Latency(frame any) (time.Duration, bool) {
	if p, ok := frame.(latencyProvider); ok {
		return p.GetTTFSP99Latency(), true
	}
	return 0, false
}

func structValue(frame any) (reflect.Value, bool) {
	v := reflect.ValueOf(frame)
	if !v.IsValid() {
		return v, false
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return v, false
		}
		v = v.Elem()
	}
	return v, v.Kind() == reflect.Struct
}

func StringField(frame any, field string) (string, bool) {
	v, ok := structValue(frame)
	if !ok {
		return "", false
	}
	f := v.FieldByName(field)
	if !f.IsValid() || f.Kind() != reflect.String {
		return "", false
	}
	return f.String(), true
}

func Text(frame any) (string, bool) {
	return StringField(frame, "Text")
}

func ToolCallIDs(frame any) []string {
	v, ok := structValue(frame)
	if !ok {
		return nil
	}
	calls := v.FieldByName("FunctionCalls")
	if !calls.IsValid() || calls.Kind() != reflect.Slice {
		return nil
	}
	ids := make([]string, 0, calls.Len())
	for i := 0; i < calls.Len(); i++ {
		c := calls.Index(i)
		if c.Kind() == reflect.Pointer {
			if c.IsNil() {
				continue
			}
			c = c.Elem()
		}
		if c.Kind() != reflect.Struct {
			continue
		}
		if id := c.FieldByName("ToolCallID"); id.IsValid() && id.Kind() == reflect.String {
			ids = append(ids, id.String())
		}
	}
	return ids
}

func CountWords(text string) int {
	count := 0
	inWord := false
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			inWord = false
		case unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana):
			count++
			inWord = false
		default:
			if !inWord {
				count++
				inWord = true
			}
		}
	}
	return count
}

func HasTurnParams(frame any) bool {
	v, ok := structValue(frame)
	if !ok {
		return false
	}
	f := v.FieldByName("TurnParams")
	return f.IsValid() && f.Kind() == reflect.Interface && !f.IsNil()
}
