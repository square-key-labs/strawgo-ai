package observers

import (
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
)

type Multi struct {
	observers []pipeline.Observer
}

func NewMulti(observers ...pipeline.Observer) *Multi {
	m := &Multi{observers: make([]pipeline.Observer, 0, len(observers))}
	for _, o := range observers {
		if o != nil {
			m.observers = append(m.observers, o)
		}
	}
	return m
}

func (m *Multi) Observers() []pipeline.Observer {
	return append([]pipeline.Observer(nil), m.observers...)
}

func (m *Multi) Interested(frame frames.Frame, direction frames.FrameDirection, kind pipeline.EventKind) bool {
	interested := false
	for _, o := range m.observers {
		if pipeline.ObserverInterested(o, frame, direction, kind) {
			interested = true
		}
	}
	return interested
}

func (m *Multi) OnProcessFrame(event pipeline.ProcessFrameEvent) {
	for _, o := range m.observers {
		m.process(o, event)
	}
}

func (m *Multi) OnPushFrame(event pipeline.PushFrameEvent) {
	for _, o := range m.observers {
		m.push(o, event)
	}
}

func (m *Multi) OnPipelineStarted() {
	for _, o := range m.observers {
		m.lifecycle(o, true)
	}
}

func (m *Multi) OnPipelineStopped() {
	for _, o := range m.observers {
		m.lifecycle(o, false)
	}
}

func (m *Multi) process(o pipeline.Observer, event pipeline.ProcessFrameEvent) {
	defer recoverObserver()
	o.OnProcessFrame(event)
}

func (m *Multi) push(o pipeline.Observer, event pipeline.PushFrameEvent) {
	defer recoverObserver()
	o.OnPushFrame(event)
}

func (m *Multi) lifecycle(o pipeline.Observer, started bool) {
	defer recoverObserver()
	if started {
		o.OnPipelineStarted()
	} else {
		o.OnPipelineStopped()
	}
}

func recoverObserver() {
	if r := recover(); r != nil {
		logger.Error("[observers.Multi] Recovered from observer panic: %v", r)
	}
}

type RecentFrames struct {
	ids  [256]uint64
	next int
}

func (r *RecentFrames) FirstSeen(id uint64) bool {
	for _, v := range r.ids {
		if v == id {
			return false
		}
	}
	r.ids[r.next] = id
	r.next = (r.next + 1) % len(r.ids)
	return true
}

func (r *RecentFrames) Reset() {
	*r = RecentFrames{}
}
