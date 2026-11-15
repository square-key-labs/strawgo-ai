package pipeline

import (
	"reflect"
	"sync"
	"sync/atomic"
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

type EventKind uint8

const (
	ProcessEvent EventKind = iota + 1
	PushEvent
)

func (k EventKind) String() string {
	switch k {
	case ProcessEvent:
		return "process"
	case PushEvent:
		return "push"
	default:
		return "unknown"
	}
}

type SelectiveObserver interface {
	Observer
	Interested(frame frames.Frame, direction frames.FrameDirection, kind EventKind) bool
}

func ObserverInterested(observer Observer, frame frames.Frame, direction frames.FrameDirection, kind EventKind) bool {
	if s, ok := observer.(SelectiveObserver); ok {
		return s.Interested(frame, direction, kind)
	}
	return true
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

type EventClass uint8

const (
	MediaEvent EventClass = iota
	DataEvent
	ControlEvent
	eventClassCount
)

func (c EventClass) String() string {
	switch c {
	case MediaEvent:
		return "media"
	case DataEvent:
		return "data"
	case ControlEvent:
		return "control"
	default:
		return "unknown"
	}
}

func IsMediaFrame(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.AudioFrame, *frames.TTSAudioFrame, *frames.InputImageRawFrame, *frames.OutputImageRawFrame,
		*frames.URLImageRawFrame, *frames.UserImageRawFrame:
		return true
	}
	return false
}

func ClassifyFrame(frame frames.Frame) EventClass {
	if IsMediaFrame(frame) {
		return MediaEvent
	}
	if c, ok := frame.(frames.Categorizable); ok && c.Category() == frames.DataCategory {
		return DataEvent
	}
	return ControlEvent
}

type ObserverDropStats struct {
	Media   uint64
	Data    uint64
	Control uint64
}

func (s ObserverDropStats) Total() uint64 {
	return s.Media + s.Data + s.Control
}

const (
	pipelineStartedEvent EventKind = iota + 100
	pipelineStoppedEvent
)

const (
	MediaObserverQueueSize   = 1024
	DataObserverQueueSize    = 4096
	ControlObserverQueueSize = 65536
	initialObserverQueueSize = 64
	observerCloseTimeout     = time.Second
)

type observerEvent struct {
	frame     frames.Frame
	processor string
	timestamp time.Time
	seq       uint64
	direction frames.FrameDirection
	kind      EventKind
}

type eventRing struct {
	buf   []observerEvent
	head  int
	n     int
	limit int
}

func (r *eventRing) push(ev observerEvent) bool {
	if r.n == len(r.buf) {
		if r.n >= r.limit {
			return false
		}
		size := max(initialObserverQueueSize, 2*len(r.buf))
		grown := make([]observerEvent, min(size, r.limit))
		for i := 0; i < r.n; i++ {
			grown[i] = r.buf[(r.head+i)&(len(r.buf)-1)]
		}
		r.buf = grown
		r.head = 0
	}
	r.buf[(r.head+r.n)&(len(r.buf)-1)] = ev
	r.n++
	return true
}

func (r *eventRing) pop() observerEvent {
	ev := r.buf[r.head]
	r.buf[r.head] = observerEvent{}
	r.head = (r.head + 1) & (len(r.buf) - 1)
	r.n--
	return ev
}

type observerWorker struct {
	observer Observer
	filter   SelectiveObserver
	wake     chan struct{}

	mu      sync.Mutex
	queues  [eventClassCount]eventRing
	pending int
	seq     uint64
	closed  bool
}

func newObserverWorker(observer Observer) *observerWorker {
	w := &observerWorker{observer: observer, wake: make(chan struct{}, 1)}
	w.filter, _ = observer.(SelectiveObserver)
	w.queues[MediaEvent].limit = MediaObserverQueueSize
	w.queues[DataEvent].limit = DataObserverQueueSize
	w.queues[ControlEvent].limit = ControlObserverQueueSize
	return w
}

func (w *observerWorker) enqueue(ev observerEvent, class EventClass) bool {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return true
	}
	w.seq++
	ev.seq = w.seq
	ok := w.queues[class].push(ev)
	wake := ok && w.pending == 0
	if ok {
		w.pending++
	}
	w.mu.Unlock()
	if wake {
		w.signal()
	}
	return ok
}

func (w *observerWorker) signal() {
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

func (w *observerWorker) pop() (ev observerEvent, ok, closed bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	best := -1
	for i := range w.queues {
		q := &w.queues[i]
		if q.n == 0 {
			continue
		}
		if best < 0 || q.buf[q.head].seq < w.queues[best].buf[w.queues[best].head].seq {
			best = i
		}
	}
	if best < 0 {
		return observerEvent{}, false, w.closed
	}
	w.pending--
	return w.queues[best].pop(), true, false
}

func (w *observerWorker) close() {
	w.mu.Lock()
	w.closed = true
	w.mu.Unlock()
	w.signal()
}

type TaskObserver struct {
	mu        sync.Mutex
	observers map[Observer]*observerWorker
	snapshot  atomic.Pointer[[]*observerWorker]
	closed    bool
	dropped   [eventClassCount]atomic.Uint64
	workers   sync.WaitGroup
}

func NewTaskObserver() *TaskObserver {
	return &TaskObserver{observers: make(map[Observer]*observerWorker)}
}

func comparableObserver(observer Observer) bool {
	if observer == nil {
		return false
	}
	t := reflect.TypeOf(observer)
	return t == nil || t.Comparable()
}

func (o *TaskObserver) publishLocked() {
	if len(o.observers) == 0 {
		o.snapshot.Store(nil)
		return
	}
	list := make([]*observerWorker, 0, len(o.observers))
	for _, w := range o.observers {
		list = append(list, w)
	}
	o.snapshot.Store(&list)
}

func (o *TaskObserver) AddObserver(observer Observer) {
	if !comparableObserver(observer) {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, exists := o.observers[observer]; exists || o.closed {
		return
	}

	worker := newObserverWorker(observer)
	o.observers[observer] = worker
	o.publishLocked()

	o.workers.Add(1)
	go o.runObserverWorker(worker)
}

func (o *TaskObserver) RemoveObserver(observer Observer) {
	if !comparableObserver(observer) {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	worker, exists := o.observers[observer]
	if !exists {
		return
	}

	delete(o.observers, observer)
	o.publishLocked()
	worker.close()
}

func (o *TaskObserver) OnProcessFrame(processorName string, frame frames.Frame, direction frames.FrameDirection) {
	o.dispatch(ProcessEvent, processorName, frame, direction)
}

func (o *TaskObserver) OnPushFrame(processorName string, frame frames.Frame, direction frames.FrameDirection) {
	o.dispatch(PushEvent, processorName, frame, direction)
}

func (o *TaskObserver) dispatch(kind EventKind, processorName string, frame frames.Frame, direction frames.FrameDirection) {
	snapshot := o.snapshot.Load()
	if snapshot == nil {
		return
	}
	var ev observerEvent
	var class EventClass
	for _, worker := range *snapshot {
		if worker.filter != nil && !worker.filter.Interested(frame, direction, kind) {
			continue
		}
		if ev.kind == 0 {
			class = ClassifyFrame(frame)
			ev = observerEvent{frame: frame, processor: processorName, timestamp: time.Now(), direction: direction, kind: kind}
		}
		if !worker.enqueue(ev, class) {
			o.dropped[class].Add(1)
		}
	}
}

func (o *TaskObserver) OnPipelineStarted() {
	o.lifecycle(pipelineStartedEvent)
}

func (o *TaskObserver) OnPipelineStopped() {
	o.lifecycle(pipelineStoppedEvent)
}

func (o *TaskObserver) lifecycle(kind EventKind) {
	snapshot := o.snapshot.Load()
	if snapshot == nil {
		return
	}
	for _, worker := range *snapshot {
		if !worker.enqueue(observerEvent{kind: kind}, ControlEvent) {
			o.dropped[ControlEvent].Add(1)
		}
	}
}

func (o *TaskObserver) Dropped() uint64 {
	return o.DropStats().Total()
}

func (o *TaskObserver) DropStats() ObserverDropStats {
	return ObserverDropStats{
		Media:   o.dropped[MediaEvent].Load(),
		Data:    o.dropped[DataEvent].Load(),
		Control: o.dropped[ControlEvent].Load(),
	}
}

func (o *TaskObserver) Close() {
	o.mu.Lock()
	if o.closed {
		o.mu.Unlock()
		return
	}
	o.closed = true
	for observer, worker := range o.observers {
		delete(o.observers, observer)
		worker.close()
	}
	o.publishLocked()
	o.mu.Unlock()

	drained := make(chan struct{})
	go func() {
		o.workers.Wait()
		close(drained)
	}()
	timer := time.NewTimer(observerCloseTimeout)
	defer timer.Stop()
	select {
	case <-drained:
	case <-timer.C:
	}
	if s := o.DropStats(); s.Total() > 0 {
		logger.Warn("[TaskObserver] dropped events from slow observers: media=%d data=%d control=%d", s.Media, s.Data, s.Control)
	}
}

func (o *TaskObserver) runObserverWorker(worker *observerWorker) {
	defer o.workers.Done()
	for {
		event, ok, closed := worker.pop()
		if ok {
			o.dispatchWithRecovery(worker.observer, event)
			continue
		}
		if closed {
			return
		}
		<-worker.wake
	}
}

func (o *TaskObserver) dispatchWithRecovery(observer Observer, event observerEvent) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("[TaskObserver] Recovered from observer panic: %v", r)
		}
	}()

	switch event.kind {
	case ProcessEvent:
		observer.OnProcessFrame(ProcessFrameEvent{
			ProcessorName: event.processor,
			Frame:         event.frame,
			Direction:     event.direction,
			Timestamp:     event.timestamp,
		})
	case PushEvent:
		observer.OnPushFrame(PushFrameEvent{
			ProcessorName: event.processor,
			Frame:         event.frame,
			Direction:     event.direction,
			Timestamp:     event.timestamp,
		})
	case pipelineStartedEvent:
		observer.OnPipelineStarted()
	case pipelineStoppedEvent:
		observer.OnPipelineStopped()
	}
}
