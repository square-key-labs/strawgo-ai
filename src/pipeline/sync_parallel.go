package pipeline

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type syncFrame struct {
	*frames.ControlFrame
	gen   uint64
	drain uint64
}

func newSyncFrame(gen uint64, survivesInterruption bool) *syncFrame {
	base := frames.NewBaseFrame("SyncFrame")
	if survivesInterruption {
		base = frames.NewUninterruptibleBaseFrame("SyncFrame")
	}
	return &syncFrame{ControlFrame: &frames.ControlFrame{BaseFrame: base}, gen: gen}
}

type emergedFrame struct {
	frame     frames.Frame
	direction frames.FrameDirection
}

type SyncParallelPipeline struct {
	branchSet
	name string

	feedMu sync.Mutex
	outMu  sync.Mutex

	mu            sync.Mutex
	dedup         *frameDedup
	gen           uint64
	active        bool
	interruptible bool
	aborted       bool
	terminalID    uint64
	remaining     int
	done          []bool
	outputs       [][]emergedFrame
	wake          chan struct{}
	drainSeq      uint64
	draining      []bool
}

var syncParallelCounter atomic.Uint64

func NewSyncParallelPipeline(branches ...[]processors.FrameProcessor) *SyncParallelPipeline {
	if len(branches) == 0 || len(branches) > maxBranches {
		panic("pipeline: SyncParallelPipeline needs between 1 and 64 branches")
	}
	sp := &SyncParallelPipeline{
		name:     fmt.Sprintf("SyncParallelPipeline#%d", syncParallelCounter.Add(1)),
		dedup:    newFrameDedup(len(branches)),
		done:     make([]bool, len(branches)),
		draining: make([]bool, len(branches)),
		outputs:  make([][]emergedFrame, len(branches)),
		wake:     make(chan struct{}, 1),
	}
	sp.input = processors.NewBaseProcessor(sp.name, stageHandler(sp.handle), processors.WithLegacyConcurrency())
	sp.build(sp, sp.name, branches, sp.emerge)
	return sp
}

func (sp *SyncParallelPipeline) Name() string {
	return sp.name
}

func (sp *SyncParallelPipeline) handle(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if frames.IsSystem(frame) {
		return sp.bypass(frame, direction)
	}
	return sp.round(ctx, frame, direction)
}

func (sp *SyncParallelPipeline) bypass(frame frames.Frame, direction frames.FrameDirection) error {
	_, interruption := frame.(*frames.InterruptionFrame)

	sp.feedMu.Lock()
	defer sp.feedMu.Unlock()

	sp.outMu.Lock()
	sp.mu.Lock()
	sp.dedup.track(frame, direction, true)
	var marker *syncFrame
	if interruption {
		if sp.active && sp.interruptible {
			sp.aborted = true
			notify(sp.wake)
		}
		sp.drainSeq++
		marker = newSyncFrame(0, true)
		marker.drain = sp.drainSeq
		for i := range sp.draining {
			sp.draining[i] = true
			sp.outputs[i] = slices.DeleteFunc(sp.outputs[i], isStale)
		}
	}
	sp.mu.Unlock()

	err := sp.PushFrame(frame, direction)
	sp.outMu.Unlock()
	for _, branch := range sp.branches {
		_ = branch.QueueFrame(frame, direction)
		if marker != nil {
			_ = branch.QueueFrame(marker, direction)
		}
	}
	return err
}

func (sp *SyncParallelPipeline) round(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	terminal := false
	switch frame.(type) {
	case *frames.EndFrame, *frames.StopFrame:
		terminal = true
	}
	n := len(sp.branches)

	sp.feedMu.Lock()
	sp.mu.Lock()
	sp.gen++
	gen := sp.gen
	sp.active = true
	sp.aborted = false
	sp.interruptible = !frame.SurvivesInterruption()
	sp.remaining = n
	for i := range sp.done {
		sp.done[i] = false
	}
	sp.terminalID = 0
	if terminal {
		sp.terminalID = frame.ID()
	} else {
		sp.dedup.track(frame, direction, false)
	}
	select {
	case <-sp.wake:
	default:
	}
	sp.mu.Unlock()

	var sentinel *syncFrame
	if !terminal {
		sentinel = newSyncFrame(gen, frame.SurvivesInterruption())
	}
	for _, branch := range sp.branches {
		_ = branch.QueueFrame(frame, direction)
		if sentinel != nil {
			_ = branch.QueueFrame(sentinel, direction)
		}
	}
	sp.feedMu.Unlock()

	for {
		sp.mu.Lock()
		complete, aborted := sp.remaining == 0, sp.aborted
		sp.mu.Unlock()
		if complete {
			break
		}
		if aborted {
			sp.release(true)
			return nil
		}
		select {
		case <-sp.wake:
		case <-ctx.Done():
			sp.release(true)
			return ctx.Err()
		}
	}

	sp.release(false)
	if terminal {
		return sp.PushFrame(frame, direction)
	}
	return nil
}

func (sp *SyncParallelPipeline) release(abandoned bool) {
	sp.outMu.Lock()
	defer sp.outMu.Unlock()

	sp.mu.Lock()
	sp.active = false
	sp.terminalID = 0
	sp.mu.Unlock()

	for i := range sp.outputs {
		for _, out := range sp.outputs[i] {
			if abandoned && !out.frame.SurvivesInterruption() {
				continue
			}
			sp.mu.Lock()
			ok := sp.dedup.admit(out.frame, i, out.direction)
			sp.mu.Unlock()
			if ok {
				_ = sp.PushFrame(out.frame, out.direction)
			}
		}
	}

	sp.mu.Lock()
	for i := range sp.outputs {
		clear(sp.outputs[i])
		sp.outputs[i] = sp.outputs[i][:0]
	}
	sp.mu.Unlock()
}

func isStale(out emergedFrame) bool {
	return out.direction == frames.Downstream && !frames.IsSystem(out.frame) && !out.frame.SurvivesInterruption()
}

func (sp *SyncParallelPipeline) emerge(i int, frame frames.Frame, direction frames.FrameDirection) error {
	sp.outMu.Lock()
	defer sp.outMu.Unlock()

	sp.mu.Lock()
	if sf, ok := frame.(*syncFrame); ok {
		if sf.drain != 0 {
			if sf.drain == sp.drainSeq {
				sp.draining[i] = false
			}
			sp.mu.Unlock()
			return nil
		}
		if sp.active && sf.gen == sp.gen {
			sp.markDone(i)
		}
		sp.mu.Unlock()
		return nil
	}
	if sp.active && sp.terminalID != 0 && frame.ID() == sp.terminalID {
		sp.markDone(i)
		sp.mu.Unlock()
		return nil
	}
	if sp.draining[i] && isStale(emergedFrame{frame: frame, direction: direction}) {
		sp.mu.Unlock()
		return nil
	}
	if sp.active && !frames.IsSystem(frame) {
		sp.outputs[i] = append(sp.outputs[i], emergedFrame{frame: frame, direction: direction})
		sp.mu.Unlock()
		return nil
	}
	ok := sp.dedup.admit(frame, i, direction)
	sp.mu.Unlock()
	if !ok {
		return nil
	}
	return sp.PushFrame(frame, direction)
}

func (sp *SyncParallelPipeline) markDone(i int) {
	if sp.done[i] {
		return
	}
	sp.done[i] = true
	sp.remaining--
	if sp.remaining == 0 {
		notify(sp.wake)
	}
}

func notify(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}
