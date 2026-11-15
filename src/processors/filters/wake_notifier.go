package filters

import (
	"context"
	"reflect"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type WakeNotifierFilter struct {
	*processors.BaseProcessor
	notifier *Notifier
	types    []reflect.Type
	filter   func(frames.Frame) bool
}

func NewWakeNotifierFilter(notifier *Notifier, filter func(frames.Frame) bool, types ...frames.Frame) *WakeNotifierFilter {
	f := &WakeNotifierFilter{notifier: notifier, filter: filter}
	for _, t := range types {
		if t != nil {
			f.types = append(f.types, reflect.TypeOf(t))
		}
	}
	f.BaseProcessor = processors.NewBaseProcessor("WakeNotifierFilter", f, processors.WithDirectMode())
	return f
}

func (f *WakeNotifierFilter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if f.notifier != nil && f.matches(frame) && (f.filter == nil || f.filter(frame)) {
		f.notifier.Notify()
	}
	return f.PushFrame(frame, direction)
}

func (f *WakeNotifierFilter) matches(frame frames.Frame) bool {
	if len(f.types) == 0 {
		return true
	}
	t := reflect.TypeOf(frame)
	for _, want := range f.types {
		if t == want {
			return true
		}
	}
	return false
}
