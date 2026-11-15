package frames

import (
	"fmt"
	"sync/atomic"
	"time"
)

var frameCounter uint64

type FrameDirection int

const (
	Downstream FrameDirection = iota
	Upstream
)

func (d FrameDirection) String() string {
	switch d {
	case Downstream:
		return "downstream"
	case Upstream:
		return "upstream"
	default:
		return "unknown"
	}
}

type Frame interface {
	ID() uint64
	Name() string
	PTS() time.Time
	Metadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	GetBroadcastSiblingID() string
	SurvivesInterruption() bool
	String() string
}

type BaseFrame struct {
	id                   uint64
	name                 string
	pts                  time.Time
	metadata             map[string]interface{}
	survivesInterruption bool
	BroadcastSiblingID   string
	TransportSource      string
	TransportDestination string
}

func NewBaseFrame(name string) *BaseFrame {
	return &BaseFrame{
		id:   atomic.AddUint64(&frameCounter, 1),
		name: name,
		pts:  time.Now(),
	}
}

func NewUninterruptibleBaseFrame(name string) *BaseFrame {
	f := NewBaseFrame(name)
	f.survivesInterruption = true
	return f
}

func (f *BaseFrame) ID() uint64 {
	return f.id
}

func (f *BaseFrame) Name() string {
	return f.name
}

func (f *BaseFrame) PTS() time.Time {
	return f.pts
}

func (f *BaseFrame) SetPTS(pts time.Time) {
	f.pts = pts
}

func (f *BaseFrame) Metadata() map[string]interface{} {
	if f.metadata == nil {
		f.metadata = make(map[string]interface{})
	}
	return f.metadata
}

func (f *BaseFrame) HasMetadata() bool {
	return len(f.metadata) > 0
}

func (f *BaseFrame) SetMetadata(key string, value interface{}) {
	if f.metadata == nil {
		f.metadata = make(map[string]interface{}, 4)
	}
	f.metadata[key] = value
}

func (f *BaseFrame) SurvivesInterruption() bool {
	return f.survivesInterruption
}

func (f *BaseFrame) SetSurvivesInterruption(v bool) {
	f.survivesInterruption = v
}

func (f *BaseFrame) String() string {
	return fmt.Sprintf("%s[id=%d, pts=%v]", f.name, f.id, f.pts.Format("15:04:05.000"))
}

func (f *BaseFrame) GetBroadcastSiblingID() string {
	return f.BroadcastSiblingID
}

func (f *BaseFrame) SetBroadcastSiblingID(id string) {
	f.BroadcastSiblingID = id
}

type FrameCategory int

const (
	SystemCategory FrameCategory = iota
	DataCategory
	ControlCategory
)

func (c FrameCategory) String() string {
	switch c {
	case SystemCategory:
		return "system"
	case DataCategory:
		return "data"
	case ControlCategory:
		return "control"
	default:
		return "unknown"
	}
}

type Categorizable interface {
	Category() FrameCategory
}

func IsSystem(f Frame) bool {
	c, ok := f.(Categorizable)
	return ok && c.Category() == SystemCategory
}
