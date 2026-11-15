package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type stageHandler func(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error

func (h stageHandler) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return h(ctx, frame, direction)
}

const (
	dedupInitialCapacity = 1024
	dedupMaxCapacity     = 1 << 16
	maxBranches          = 64
)

type dedupEntry struct {
	id      uint64
	mask    uint64
	pending int
	lane    uint8
	seen    bool
	live    bool
}

func (e *dedupEntry) mark(branch int) {
	bit := uint64(1) << uint(branch)
	if e.mask&bit == 0 {
		e.mask |= bit
		e.pending--
	}
}

type frameDedup struct {
	branches int
	ring     []dedupEntry
	head     int
	n        int
	headSeq  uint64
	index    map[uint64]uint64
	cursors  [][4]uint64
}

func newFrameDedup(branches int) *frameDedup {
	return &frameDedup{
		branches: branches,
		ring:     make([]dedupEntry, dedupInitialCapacity),
		index:    make(map[uint64]uint64),
		cursors:  make([][4]uint64, branches),
	}
}

func laneOf(frame frames.Frame, direction frames.FrameDirection) uint8 {
	var lane uint8
	if direction == frames.Upstream {
		lane = 2
	}
	if frames.IsSystem(frame) {
		lane++
	}
	return lane
}

func (d *frameDedup) slot(seq uint64) *dedupEntry {
	return &d.ring[(d.head+int(seq-d.headSeq))%len(d.ring)]
}

func (d *frameDedup) track(frame frames.Frame, direction frames.FrameDirection, forwarded bool) {
	if d.branches < 2 && !forwarded {
		return
	}
	id := frame.ID()
	if seq, ok := d.index[id]; ok {
		d.slot(seq).live = false
		delete(d.index, id)
	}
	if d.n == len(d.ring) {
		d.makeRoom()
	}
	seq := d.headSeq + uint64(d.n)
	*d.slot(seq) = dedupEntry{
		id:      id,
		pending: d.branches,
		lane:    laneOf(frame, direction),
		seen:    forwarded,
		live:    true,
	}
	d.n++
	d.index[id] = seq
}

func (d *frameDedup) makeRoom() {
	head := &d.ring[d.head]
	if head.live && head.pending > 0 && len(d.ring) < dedupMaxCapacity {
		grown := make([]dedupEntry, 2*len(d.ring))
		for i := 0; i < d.n; i++ {
			grown[i] = d.ring[(d.head+i)%len(d.ring)]
		}
		d.ring = grown
		d.head = 0
		return
	}
	if head.live {
		if seq, ok := d.index[head.id]; ok && seq == d.headSeq {
			delete(d.index, head.id)
		}
	}
	*head = dedupEntry{}
	d.head = (d.head + 1) % len(d.ring)
	d.headSeq++
	d.n--
}

func (d *frameDedup) admit(frame frames.Frame, branch int, direction frames.FrameDirection) bool {
	seq, ok := d.index[frame.ID()]
	if !ok {
		return true
	}
	e := d.slot(seq)
	if lane := laneOf(frame, direction); lane == e.lane {
		cursor := &d.cursors[branch][lane]
		if *cursor < d.headSeq {
			*cursor = d.headSeq
		}
		for ; *cursor < seq; *cursor++ {
			if skipped := d.slot(*cursor); skipped.live && skipped.lane == lane {
				skipped.mark(branch)
			}
		}
		if *cursor == seq {
			*cursor = seq + 1
		}
	}
	e.mark(branch)
	first := !e.seen
	e.seen = true
	return first
}

func joinsBranches(frame frames.Frame) bool {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		return true
	}
	return isLifecycle(frame)
}

func isLifecycle(frame frames.Frame) bool {
	switch frame.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.StopFrame, *frames.CancelFrame:
		return true
	}
	return false
}

type branchSet struct {
	self     processors.FrameProcessor
	branches []*Pipeline
	tops     []*edge
	bottoms  []*edge
	input    *processors.BaseProcessor
	next     atomic.Pointer[procRef]
	prev     atomic.Pointer[procRef]

	lifeMu  sync.Mutex
	running bool
	started []processors.FrameProcessor
}

func (b *branchSet) build(self processors.FrameProcessor, name string, specs [][]processors.FrameProcessor, emerge func(i int, frame frames.Frame, direction frames.FrameDirection) error) {
	b.self = self
	b.branches = make([]*Pipeline, len(specs))
	b.tops = make([]*edge, len(specs))
	b.bottoms = make([]*edge, len(specs))
	for i, procs := range specs {
		i := i
		branch := NewPipeline(procs)
		top := newEdge(fmt.Sprintf("%s::Top%d", name, i), func(frame frames.Frame, direction frames.FrameDirection) error {
			if direction != frames.Upstream {
				return nil
			}
			return emerge(i, frame, direction)
		})
		bottom := newEdge(fmt.Sprintf("%s::Bottom%d", name, i), func(frame frames.Frame, direction frames.FrameDirection) error {
			if direction != frames.Downstream {
				return nil
			}
			return emerge(i, frame, direction)
		})
		branch.SetPrev(top)
		branch.Link(bottom)
		b.branches[i] = branch
		b.tops[i] = top
		b.bottoms[i] = bottom
	}
	b.input.SetPrev(newEdge(name+"::Errors", func(frame frames.Frame, direction frames.FrameDirection) error {
		return b.PushFrame(frame, direction)
	}))
}

func (b *branchSet) Branches() []*Pipeline {
	return b.branches
}

func (b *branchSet) Link(next processors.FrameProcessor) {
	if next == nil {
		b.next.Store(nil)
		return
	}
	b.next.Store(&procRef{p: next})
	next.SetPrev(b.self)
}

func (b *branchSet) SetPrev(prev processors.FrameProcessor) {
	if prev == nil {
		b.prev.Store(nil)
		return
	}
	b.prev.Store(&procRef{p: prev})
}

func (b *branchSet) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return b.input.QueueFrame(frame, direction)
}

func (b *branchSet) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return b.QueueFrame(frame, direction)
}

func (b *branchSet) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	var ref *procRef
	if direction == frames.Downstream {
		ref = b.next.Load()
	} else {
		ref = b.prev.Load()
	}
	if ref == nil {
		return nil
	}
	return ref.p.QueueFrame(frame, direction)
}

func (b *branchSet) SetObserver(observer processors.FrameObserver) {
	for _, branch := range b.branches {
		branch.SetObserver(observer)
	}
}

func (b *branchSet) Start(ctx context.Context) error {
	b.lifeMu.Lock()
	defer b.lifeMu.Unlock()
	if b.running {
		return fmt.Errorf("%s already started", b.input.Name())
	}

	chain := make([]processors.FrameProcessor, 0, 3*len(b.branches)+1)
	for i := range b.branches {
		chain = append(chain, b.tops[i], b.bottoms[i])
	}
	for _, branch := range b.branches {
		chain = append(chain, branch)
	}
	chain = append(chain, b.input)

	for _, proc := range chain {
		if err := proc.Start(ctx); err != nil {
			for i := len(b.started) - 1; i >= 0; i-- {
				_ = b.started[i].Stop()
			}
			b.started = nil
			return fmt.Errorf("failed to start %s: %w", proc.Name(), err)
		}
		b.started = append(b.started, proc)
	}
	b.running = true
	return nil
}

func (b *branchSet) Stop() error {
	b.lifeMu.Lock()
	defer b.lifeMu.Unlock()
	var errs []error
	for i := len(b.started) - 1; i >= 0; i-- {
		if err := b.started[i].Stop(); err != nil {
			errs = append(errs, fmt.Errorf("stop %s: %w", b.started[i].Name(), err))
		}
	}
	b.started = nil
	b.running = false
	return errors.Join(errs...)
}

type ParallelPipeline struct {
	branchSet
	name string

	mu       sync.Mutex
	dedup    *frameDedup
	pending  map[uint64]int
	starting int
	flushing bool
	held     []frames.Frame
}

var parallelCounter atomic.Uint64

func NewParallelPipeline(branches ...[]processors.FrameProcessor) *ParallelPipeline {
	if len(branches) == 0 || len(branches) > maxBranches {
		panic("pipeline: ParallelPipeline needs between 1 and 64 branches")
	}
	pp := &ParallelPipeline{
		name:    fmt.Sprintf("ParallelPipeline#%d", parallelCounter.Add(1)),
		dedup:   newFrameDedup(len(branches)),
		pending: make(map[uint64]int),
	}
	pp.input = processors.NewBaseProcessor(pp.name, stageHandler(pp.fanOut))
	pp.build(pp, pp.name, branches, pp.emerge)
	return pp
}

func (pp *ParallelPipeline) Name() string {
	return pp.name
}

func (pp *ParallelPipeline) fanOut(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	n := len(pp.branches)
	if direction == frames.Downstream && joinsBranches(frame) {
		pp.mu.Lock()
		pp.pending[frame.ID()] += n
		if _, ok := frame.(*frames.StartFrame); ok {
			pp.starting++
		}
		pp.mu.Unlock()
	} else if n > 1 {
		pp.mu.Lock()
		pp.dedup.track(frame, direction, false)
		pp.mu.Unlock()
	}

	var firstErr error
	for _, branch := range pp.branches {
		if err := branch.QueueFrame(frame, direction); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (pp *ParallelPipeline) emerge(i int, frame frames.Frame, direction frames.FrameDirection) error {
	if direction == frames.Downstream && joinsBranches(frame) {
		pp.mu.Lock()
		if count, tracked := pp.pending[frame.ID()]; tracked {
			if count > 1 {
				pp.pending[frame.ID()] = count - 1
				pp.mu.Unlock()
				return nil
			}
			delete(pp.pending, frame.ID())
			if _, ok := frame.(*frames.StartFrame); ok {
				pp.starting--
				if pp.starting > 0 {
					pp.mu.Unlock()
					return nil
				}
				pp.flushing = true
				pp.mu.Unlock()
				err := pp.PushFrame(frame, frames.Downstream)
				pp.flushHeld()
				return err
			}
			if _, ok := frame.(*frames.CancelFrame); !ok && pp.holding() {
				pp.held = append(pp.held, frame)
				pp.mu.Unlock()
				return nil
			}
			pp.mu.Unlock()
			return pp.PushFrame(frame, frames.Downstream)
		}
		pp.mu.Unlock()
	}

	pp.mu.Lock()
	if !pp.dedup.admit(frame, i, direction) {
		pp.mu.Unlock()
		return nil
	}
	if direction == frames.Downstream && pp.holding() {
		pp.held = append(pp.held, frame)
		pp.mu.Unlock()
		return nil
	}
	pp.mu.Unlock()
	return pp.PushFrame(frame, direction)
}

func (pp *ParallelPipeline) holding() bool {
	return pp.starting > 0 || pp.flushing
}

func (pp *ParallelPipeline) flushHeld() {
	for {
		pp.mu.Lock()
		if len(pp.held) == 0 {
			pp.flushing = false
			pp.mu.Unlock()
			return
		}
		batch := pp.held
		pp.held = nil
		pp.mu.Unlock()
		for _, f := range batch {
			_ = pp.PushFrame(f, frames.Downstream)
		}
	}
}
