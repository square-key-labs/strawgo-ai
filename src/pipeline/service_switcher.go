package pipeline

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type ServiceSwitcherStrategy interface {
	OnSwitchFrame(frame frames.Frame, services []processors.FrameProcessor, active int) int
	OnServiceError(err *frames.ErrorFrame, services []processors.FrameProcessor, active int) int
}

type ManualSwitchStrategy struct{}

func (ManualSwitchStrategy) OnSwitchFrame(frame frames.Frame, services []processors.FrameProcessor, active int) int {
	if f, ok := frame.(*frames.ManuallySwitchServiceFrame); ok {
		return indexOfService(services, f.Service)
	}
	return -1
}

func (ManualSwitchStrategy) OnServiceError(*frames.ErrorFrame, []processors.FrameProcessor, int) int {
	return -1
}

type FailoverSwitchStrategy struct {
	ManualSwitchStrategy
}

func (FailoverSwitchStrategy) OnServiceError(err *frames.ErrorFrame, services []processors.FrameProcessor, active int) int {
	if err.Fatal || len(services) < 2 {
		return -1
	}
	return (active + 1) % len(services)
}

func indexOfService(services []processors.FrameProcessor, ref frames.ProcessorRef) int {
	if ref == nil {
		return -1
	}
	for i, s := range services {
		if sameRef(s, ref) {
			return i
		}
	}
	for i, s := range services {
		if s.Name() == ref.Name() {
			return i
		}
	}
	return -1
}

func sameRef(s processors.FrameProcessor, ref frames.ProcessorRef) (same bool) {
	defer func() {
		if recover() != nil {
			same = false
		}
	}()
	return frames.ProcessorRef(s) == ref
}

type ServiceSwitcher struct {
	branchSet
	name     string
	services []processors.FrameProcessor
	strategy ServiceSwitcherStrategy
	active   atomic.Int32
	holdFlag atomic.Bool

	mu       sync.Mutex
	pending  map[uint64]int
	starting int
	flushing bool
	held     []frames.Frame
	copies   [64]uint64
	copyNext int
	switched []func(processors.FrameProcessor)
}

var switcherCounter atomic.Uint64

func NewServiceSwitcher(services []processors.FrameProcessor, strategy ServiceSwitcherStrategy) *ServiceSwitcher {
	if len(services) == 0 || len(services) > maxBranches {
		panic("pipeline: ServiceSwitcher needs between 1 and 64 services")
	}
	if strategy == nil {
		strategy = ManualSwitchStrategy{}
	}
	s := &ServiceSwitcher{
		name:     fmt.Sprintf("ServiceSwitcher#%d", switcherCounter.Add(1)),
		services: append([]processors.FrameProcessor(nil), services...),
		strategy: strategy,
		pending:  make(map[uint64]int),
	}
	specs := make([][]processors.FrameProcessor, len(services))
	for i, svc := range services {
		specs[i] = []processors.FrameProcessor{svc}
	}
	s.input = processors.NewBaseProcessor(s.name, stageHandler(s.route), processors.WithDirectMode())
	s.build(s, s.name, specs, s.emerge)
	return s
}

func (s *ServiceSwitcher) Name() string {
	return s.name
}

func (s *ServiceSwitcher) Services() []processors.FrameProcessor {
	return append([]processors.FrameProcessor(nil), s.services...)
}

func (s *ServiceSwitcher) ActiveService() processors.FrameProcessor {
	return s.services[s.active.Load()]
}

func (s *ServiceSwitcher) Strategy() ServiceSwitcherStrategy {
	return s.strategy
}

func (s *ServiceSwitcher) OnServiceSwitched(fn func(service processors.FrameProcessor)) {
	if fn == nil {
		return
	}
	s.mu.Lock()
	s.switched = append(s.switched[:len(s.switched):len(s.switched)], fn)
	s.mu.Unlock()
}

func (s *ServiceSwitcher) SwitchTo(service frames.ProcessorRef) bool {
	idx := indexOfService(s.services, service)
	if idx < 0 {
		return false
	}
	s.activate(idx)
	return true
}

func (s *ServiceSwitcher) activate(idx int) {
	s.active.Store(int32(idx))
	svc := s.services[idx]
	_ = s.branches[idx].QueueFrame(frames.NewServiceSwitcherRequestMetadataFrame(svc), frames.Downstream)
	s.mu.Lock()
	handlers := s.switched
	s.mu.Unlock()
	for _, h := range handlers {
		h(svc)
	}
}

func (s *ServiceSwitcher) route(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch frame.(type) {
	case *frames.ManuallySwitchServiceFrame, *frames.ServiceSwitcherFrame:
		if idx := s.strategy.OnSwitchFrame(frame, s.services, int(s.active.Load())); idx >= 0 && idx < len(s.services) {
			s.activate(idx)
			return nil
		}
	}

	if direction == frames.Downstream && isLifecycle(frame) {
		s.mu.Lock()
		s.pending[frame.ID()] += len(s.branches)
		if _, ok := frame.(*frames.StartFrame); ok {
			s.starting++
			s.holdFlag.Store(true)
		}
		s.mu.Unlock()
		var firstErr error
		for _, b := range s.branches {
			if err := b.QueueFrame(frame, direction); err != nil && firstErr == nil {
				firstErr = err
			}
		}
		return firstErr
	}

	active := int(s.active.Load())
	err := s.branches[active].QueueFrame(frame, direction)
	switch frame.(type) {
	case *frames.LLMUpdateSettingsFrame, *frames.TTSUpdateSettingsFrame, *frames.STTUpdateSettingsFrame:
		s.updateInactive(frame, direction, active)
	}
	return err
}

func (s *ServiceSwitcher) updateInactive(frame frames.Frame, direction frames.FrameDirection, active int) {
	var target frames.ProcessorRef
	var reach bool
	switch f := frame.(type) {
	case *frames.LLMUpdateSettingsFrame:
		target, reach = f.Service, f.ReachInactiveServices
	case *frames.TTSUpdateSettingsFrame:
		target, reach = f.Service, f.ReachInactiveServices
	case *frames.STTUpdateSettingsFrame:
		target, reach = f.Service, f.ReachInactiveServices
	}
	targetIdx := -1
	if target != nil {
		targetIdx = indexOfService(s.services, target)
		if targetIdx < 0 || targetIdx == active {
			return
		}
	} else if !reach {
		return
	}
	for i := range s.services {
		if i == active || (targetIdx >= 0 && i != targetIdx) {
			continue
		}
		cp := copySettingsFrame(frame)
		if cp == nil {
			return
		}
		s.mu.Lock()
		s.copies[s.copyNext] = cp.ID()
		s.copyNext = (s.copyNext + 1) % len(s.copies)
		s.mu.Unlock()
		_ = s.branches[i].QueueFrame(cp, direction)
	}
}

func copySettingsFrame(frame frames.Frame) frames.Frame {
	switch f := frame.(type) {
	case *frames.LLMUpdateSettingsFrame:
		cp := frames.NewLLMUpdateSettingsFrame(f.Delta)
		cp.Service, cp.ReachInactiveServices = f.Service, f.ReachInactiveServices
		return cp
	case *frames.TTSUpdateSettingsFrame:
		cp := frames.NewTTSUpdateSettingsFrame(f.Delta)
		cp.Service, cp.ReachInactiveServices = f.Service, f.ReachInactiveServices
		return cp
	case *frames.STTUpdateSettingsFrame:
		cp := frames.NewSTTUpdateSettingsFrame(f.Delta)
		cp.Service, cp.ReachInactiveServices = f.Service, f.ReachInactiveServices
		return cp
	}
	return nil
}

func (s *ServiceSwitcher) isCopy(id uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.copies {
		if c == id {
			return true
		}
	}
	return false
}

func (s *ServiceSwitcher) inactiveServiceName(name string) bool {
	active := int(s.active.Load())
	for i, svc := range s.services {
		if svc.Name() == name {
			return i != active && s.services[active].Name() != name
		}
	}
	return false
}

func (s *ServiceSwitcher) emerge(i int, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Downstream && isLifecycle(frame) {
		if done, err := s.mergeLifecycle(frame); done {
			return err
		}
	}

	switch f := frame.(type) {
	case *frames.ErrorFrame:
		if s.handleError(i, f) {
			return nil
		}
	case *frames.ServiceSwitcherRequestMetadataFrame:
		if indexOfService(s.services, f.Service) >= 0 {
			return nil
		}
	case *frames.ServiceMetadataFrame:
		if s.inactiveServiceName(f.ServiceName) {
			return nil
		}
	case *frames.LLMServiceMetadataFrame:
		if s.inactiveServiceName(f.ServiceName) {
			return nil
		}
	case *frames.TTSMetadataFrame:
		if s.inactiveServiceName(f.ServiceName) {
			return nil
		}
	case *frames.LLMUpdateSettingsFrame, *frames.TTSUpdateSettingsFrame, *frames.STTUpdateSettingsFrame:
		if s.isCopy(frame.ID()) {
			return nil
		}
	}

	if direction == frames.Downstream && s.holdFlag.Load() {
		s.mu.Lock()
		if s.starting > 0 || s.flushing {
			s.held = append(s.held, frame)
			s.mu.Unlock()
			return nil
		}
		s.mu.Unlock()
	}
	return s.PushFrame(frame, direction)
}

func (s *ServiceSwitcher) mergeLifecycle(frame frames.Frame) (bool, error) {
	s.mu.Lock()
	count, tracked := s.pending[frame.ID()]
	if !tracked {
		s.mu.Unlock()
		return false, nil
	}
	if count > 1 {
		s.pending[frame.ID()] = count - 1
		s.mu.Unlock()
		return true, nil
	}
	delete(s.pending, frame.ID())
	if _, ok := frame.(*frames.StartFrame); ok {
		s.starting--
		if s.starting > 0 {
			s.mu.Unlock()
			return true, nil
		}
		s.flushing = true
		s.mu.Unlock()
		err := s.PushFrame(frame, frames.Downstream)
		s.flushHeld()
		return true, err
	}
	if _, ok := frame.(*frames.CancelFrame); !ok && (s.starting > 0 || s.flushing) {
		s.held = append(s.held, frame)
		s.mu.Unlock()
		return true, nil
	}
	s.mu.Unlock()
	return true, s.PushFrame(frame, frames.Downstream)
}

func (s *ServiceSwitcher) flushHeld() {
	for {
		s.mu.Lock()
		if len(s.held) == 0 {
			s.flushing = false
			if s.starting == 0 {
				s.holdFlag.Store(false)
			}
			s.mu.Unlock()
			return
		}
		batch := s.held
		s.held = nil
		s.mu.Unlock()
		for _, f := range batch {
			_ = s.PushFrame(f, frames.Downstream)
		}
	}
}

func (s *ServiceSwitcher) handleError(branch int, f *frames.ErrorFrame) bool {
	if f.Fatal {
		return false
	}
	active := int(s.active.Load())
	if branch != active {
		return true
	}
	next := s.strategy.OnServiceError(f, s.services, active)
	if next < 0 || next >= len(s.services) || next == active {
		return false
	}
	logger.Warn("[%s] %s reported an error (%v), switching to %s", s.name, s.services[active].Name(), f.Error, s.services[next].Name())
	s.activate(next)
	return true
}
