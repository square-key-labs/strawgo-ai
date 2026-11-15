package processors

import (
	"context"
	"reflect"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func isTerminal(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.EndFrame, *frames.StopFrame:
		return true
	}
	return false
}

func alwaysPasses(frame frames.Frame) bool {
	return frames.IsSystem(frame) || isTerminal(frame)
}

type FrameFilter struct {
	*BaseProcessor
	types []reflect.Type
}

func NewFrameFilter(examples ...frames.Frame) *FrameFilter {
	f := &FrameFilter{types: make([]reflect.Type, 0, len(examples))}
	for _, ex := range examples {
		if ex != nil {
			f.types = append(f.types, reflect.TypeOf(ex))
		}
	}
	f.BaseProcessor = NewBaseProcessor("FrameFilter", f, WithDirectMode())
	return f
}

func (f *FrameFilter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if alwaysPasses(frame) || f.matches(frame) {
		return f.PushFrame(frame, direction)
	}
	return nil
}

func (f *FrameFilter) matches(frame frames.Frame) bool {
	t := reflect.TypeOf(frame)
	for _, want := range f.types {
		if t == want {
			return true
		}
	}
	return false
}

type FunctionFilter struct {
	*BaseProcessor
	predicate          func(frames.Frame) bool
	direction          frames.FrameDirection
	bothDirections     bool
	filterSystemFrames bool
}

type FunctionFilterOption func(*FunctionFilter)

func WithFunctionFilterDirection(direction frames.FrameDirection) FunctionFilterOption {
	return func(f *FunctionFilter) {
		f.direction = direction
		f.bothDirections = false
	}
}

func WithFunctionFilterBothDirections() FunctionFilterOption {
	return func(f *FunctionFilter) { f.bothDirections = true }
}

func WithFunctionFilterSystemFrames(enabled bool) FunctionFilterOption {
	return func(f *FunctionFilter) { f.filterSystemFrames = enabled }
}

func NewFunctionFilter(predicate func(frames.Frame) bool, opts ...FunctionFilterOption) *FunctionFilter {
	f := &FunctionFilter{predicate: predicate, direction: frames.Downstream}
	for _, opt := range opts {
		opt(f)
	}
	f.BaseProcessor = NewBaseProcessor("FunctionFilter", f, WithDirectMode())
	return f
}

func (f *FunctionFilter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if f.passesUnfiltered(frame, direction) || f.predicate == nil || f.predicate(frame) {
		return f.PushFrame(frame, direction)
	}
	return nil
}

func (f *FunctionFilter) passesUnfiltered(frame frames.Frame, direction frames.FrameDirection) bool {
	if !f.bothDirections && direction != f.direction {
		return true
	}
	if isLifecycleFrame(frame) {
		return true
	}
	return !f.filterSystemFrames && frames.IsSystem(frame)
}

func isLifecycleFrame(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.StopFrame, *frames.CancelFrame:
		return true
	}
	return false
}

type IdentityFilter struct {
	*BaseProcessor
}

func NewIdentityFilter() *IdentityFilter {
	f := &IdentityFilter{}
	f.BaseProcessor = NewBaseProcessor("IdentityFilter", f, WithDirectMode())
	return f
}

func (f *IdentityFilter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return f.PushFrame(frame, direction)
}

type NullFilter struct {
	*BaseProcessor
}

func NewNullFilter() *NullFilter {
	f := &NullFilter{}
	f.BaseProcessor = NewBaseProcessor("NullFilter", f, WithDirectMode())
	return f
}

func (f *NullFilter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if alwaysPasses(frame) {
		return f.PushFrame(frame, direction)
	}
	return nil
}
