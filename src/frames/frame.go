package frames

import (
	"fmt"
	"sync/atomic"
	"time"
)

var frameCounter uint64

// FrameDirection indicates the direction a frame is traveling
type FrameDirection int

const (
	Downstream FrameDirection = iota // Normal flow: source -> sink
	Upstream                          // Reverse flow: sink -> source
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

// Frame is the base interface for all frames in the pipeline
type Frame interface {
	ID() uint64
	Name() string
	PTS() time.Time
	Metadata() map[string]interface{}
	SetMetadata(key string, value interface{})
	String() string
}

// BaseFrame provides common frame functionality
type BaseFrame struct {
	id       uint64
	name     string
	pts      time.Time
	metadata map[string]interface{}
}

func NewBaseFrame(name string) *BaseFrame {
	return &BaseFrame{
		id:       atomic.AddUint64(&frameCounter, 1),
		name:     name,
		pts:      time.Now(),
		metadata: make(map[string]interface{}),
	}
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

func (f *BaseFrame) Metadata() map[string]interface{} {
	return f.metadata
}

func (f *BaseFrame) SetMetadata(key string, value interface{}) {
	f.metadata[key] = value
}

func (f *BaseFrame) String() string {
	return fmt.Sprintf("%s[id=%d, pts=%v]", f.name, f.id, f.pts.Format("15:04:05.000"))
}

// Frame categories for priority handling
type FrameCategory int

const (
	SystemCategory  FrameCategory = iota // Highest priority, processed immediately
	DataCategory                         // Normal priority, ordered processing
	ControlCategory                      // Ordered processing, configuration
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

// Categorizable frames can report their category
type Categorizable interface {
	Category() FrameCategory
}
