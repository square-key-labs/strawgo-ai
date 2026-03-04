package pipeline

import (
	"reflect"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type Observer interface {
	OnProcessFrame(event ProcessFrameEvent)
	OnPushFrame(event PushFrameEvent)
	OnPipelineStarted()
	OnPipelineStopped()
}

type ProcessFrameEvent struct {
	ProcessorName string
	Frame         frames.Frame
	Direction     frames.FrameDirection
	Timestamp     time.Time
}

type PushFrameEvent struct {
	ProcessorName string
	Frame         frames.Frame
	Direction     frames.FrameDirection
	Timestamp     time.Time
}

type observerEventType int

const (
	processFrameObserverEvent observerEventType = iota
	pushFrameObserverEvent
	pipelineStartedObserverEvent
	pipelineStoppedObserverEvent
)

type observerEvent struct {
	typeID       observerEventType
	processEvent ProcessFrameEvent
	pushEvent    PushFrameEvent
}

type observerWorker struct {
	observer Observer
	ch       chan observerEvent
}

type TaskObserver struct {
	mu        sync.RWMutex
	observers map[Observer]*observerWorker
}

func NewTaskObserver() *TaskObserver {
	return &TaskObserver{observers: make(map[Observer]*observerWorker)}
}

func (o *TaskObserver) AddObserver(observer Observer) {
	if observer == nil {
		return
	}
	t := reflect.TypeOf(observer)
	if t != nil && !t.Comparable() {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, exists := o.observers[observer]; exists {
		return
	}

	worker := &observerWorker{
		observer: observer,
		ch:       make(chan observerEvent, 100),
	}

	o.observers[observer] = worker

	go o.runObserverWorker(worker)
}

func (o *TaskObserver) RemoveObserver(observer Observer) {
	if observer == nil {
		return
	}
	t := reflect.TypeOf(observer)
	if t != nil && !t.Comparable() {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	worker, exists := o.observers[observer]
	if !exists {
		return
	}

	delete(o.observers, observer)
	close(worker.ch)
}

func (o *TaskObserver) OnProcessFrame(processorName string, frame frames.Frame, direction frames.FrameDirection) {
	o.broadcast(observerEvent{
		typeID: processFrameObserverEvent,
		processEvent: ProcessFrameEvent{
			ProcessorName: processorName,
			Frame:         frame,
			Direction:     direction,
			Timestamp:     time.Now(),
		},
	})
}

func (o *TaskObserver) OnPushFrame(processorName string, frame frames.Frame, direction frames.FrameDirection) {
	o.broadcast(observerEvent{
		typeID: pushFrameObserverEvent,
		pushEvent: PushFrameEvent{
			ProcessorName: processorName,
			Frame:         frame,
			Direction:     direction,
			Timestamp:     time.Now(),
		},
	})
}

func (o *TaskObserver) OnPipelineStarted() {
	o.broadcast(observerEvent{typeID: pipelineStartedObserverEvent})
}

func (o *TaskObserver) OnPipelineStopped() {
	o.broadcast(observerEvent{typeID: pipelineStoppedObserverEvent})
}

func (o *TaskObserver) broadcast(event observerEvent) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	for _, worker := range o.observers {
		select {
		case worker.ch <- event:
		default:
		}
	}
}

func (o *TaskObserver) runObserverWorker(worker *observerWorker) {
	for event := range worker.ch {
		o.dispatchWithRecovery(worker.observer, event)
	}
}

func (o *TaskObserver) dispatchWithRecovery(observer Observer, event observerEvent) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("[TaskObserver] Recovered from observer panic: %v", r)
		}
	}()

	switch event.typeID {
	case processFrameObserverEvent:
		observer.OnProcessFrame(event.processEvent)
	case pushFrameObserverEvent:
		observer.OnPushFrame(event.pushEvent)
	case pipelineStartedObserverEvent:
		observer.OnPipelineStarted()
	case pipelineStoppedObserverEvent:
		observer.OnPipelineStopped()
	}
}
