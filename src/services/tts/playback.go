package tts

import (
	"context"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

const maxDormant = 16

type audioContext struct {
	id        string
	items     []frames.Frame
	head      int
	flushed   bool
	done      bool
	gotAudio  bool
	reopened  bool
	lastItem  time.Time
	ttfbStart time.Time
}

type entry struct {
	frame frames.Frame
	ctx   *audioContext
}

type playback struct {
	b           *Base
	stopTimeout time.Duration

	pushMu sync.Mutex

	mu      sync.Mutex
	entries []entry
	head    int
	live    map[string]*audioContext
	dormant map[string]bool
	waiters []chan struct{}

	wake chan struct{}
}

func newPlayback(b *Base, stopTimeout time.Duration) *playback {
	return &playback{
		b:           b,
		stopTimeout: stopTimeout,
		live:        make(map[string]*audioContext),
		dormant:     make(map[string]bool),
		wake:        make(chan struct{}, 1),
	}
}

func (p *playback) signal() {
	select {
	case p.wake <- struct{}{}:
	default:
	}
}

func (p *playback) openLocked(id string, now time.Time) *audioContext {
	c := &audioContext{id: id, lastItem: now, ttfbStart: now}
	c.items = append(c.items, frames.NewTTSStartedFrameWithContext(id))
	p.live[id] = c
	delete(p.dormant, id)
	p.entries = append(p.entries, entry{ctx: c})
	return c
}

func (p *playback) open(id string) bool {
	p.mu.Lock()
	if _, ok := p.live[id]; ok {
		p.mu.Unlock()
		return false
	}
	p.openLocked(id, time.Now())
	p.mu.Unlock()
	p.signal()
	return true
}

func (p *playback) contextLocked(id string, now time.Time) *audioContext {
	if c := p.live[id]; c != nil {
		return c
	}
	if !p.dormant[id] {
		return nil
	}
	c := p.openLocked(id, now)
	c.reopened = true
	return c
}

func (p *playback) addAudio(id string, data []byte) bool {
	now := time.Now()
	p.mu.Lock()
	c := p.contextLocked(id, now)
	if c == nil {
		p.mu.Unlock()
		return false
	}
	first := !c.gotAudio && !c.reopened
	c.gotAudio = true
	c.lastItem = now
	c.items = append(c.items, p.b.audioFrame(id, data))
	start := c.ttfbStart
	p.mu.Unlock()
	p.signal()
	if first {
		p.b.reportTTFB(now.Sub(start))
	}
	return true
}

func (p *playback) addFrame(id string, frame frames.Frame) bool {
	p.mu.Lock()
	c := p.contextLocked(id, time.Now())
	if c == nil {
		p.mu.Unlock()
		return false
	}
	c.items = append(c.items, frame)
	p.mu.Unlock()
	p.signal()
	return true
}

func (p *playback) markFlushed(id string) {
	p.mu.Lock()
	if c := p.live[id]; c != nil && !c.flushed {
		c.flushed = true
		c.lastItem = time.Now()
	}
	p.mu.Unlock()
	p.signal()
}

func (p *playback) markDone(id string) {
	p.mu.Lock()
	if c := p.live[id]; c != nil {
		c.done = true
	}
	delete(p.dormant, id)
	p.mu.Unlock()
	p.signal()
}

func (p *playback) isLive(id string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.live[id] != nil
}

func (p *playback) pass(frame frames.Frame) error {
	p.pushMu.Lock()
	p.mu.Lock()
	if p.head == len(p.entries) {
		p.mu.Unlock()
		err := p.b.PushFrame(frame, frames.Downstream)
		p.pushMu.Unlock()
		return err
	}
	p.entries = append(p.entries, entry{frame: frame})
	p.mu.Unlock()
	p.pushMu.Unlock()
	p.signal()
	return nil
}

func (p *playback) interrupt() (cancelled []string, active bool) {
	p.pushMu.Lock()
	p.mu.Lock()
	kept := make([]entry, 0, len(p.entries)-p.head)
	for _, e := range p.entries[p.head:] {
		if e.ctx != nil {
			active = true
			if !e.ctx.done {
				cancelled = append(cancelled, e.ctx.id)
			}
			continue
		}
		if e.frame.SurvivesInterruption() {
			kept = append(kept, e)
		}
	}
	p.entries = kept
	p.head = 0
	clear(p.live)
	clear(p.dormant)
	p.releaseWaitersLocked()
	p.mu.Unlock()
	p.pushMu.Unlock()
	p.signal()
	return cancelled, active
}

func (p *playback) waitIdle(ctx context.Context, timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		p.mu.Lock()
		if p.head == len(p.entries) {
			p.mu.Unlock()
			return
		}
		ch := make(chan struct{})
		p.waiters = append(p.waiters, ch)
		p.mu.Unlock()
		select {
		case <-ch:
		case <-timer.C:
			return
		case <-ctx.Done():
			return
		}
	}
}

func (p *playback) releaseWaitersLocked() {
	for _, ch := range p.waiters {
		close(ch)
	}
	p.waiters = p.waiters[:0]
}

func (p *playback) popLocked() {
	p.entries[p.head] = entry{}
	p.head++
	if p.head == len(p.entries) {
		p.entries = p.entries[:0]
		p.head = 0
	}
}

func (p *playback) next(now time.Time) (frames.Frame, bool, time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for p.head < len(p.entries) {
		e := p.entries[p.head]
		if e.ctx == nil {
			p.popLocked()
			return e.frame, false, 0
		}
		c := e.ctx
		if c.head < len(c.items) {
			f := c.items[c.head]
			c.items[c.head] = nil
			c.head++
			if c.head == len(c.items) {
				c.items = c.items[:0]
				c.head = 0
			}
			return f, false, 0
		}
		timed := c.flushed || c.gotAudio
		idle := now.Sub(c.lastItem)
		if c.done || (timed && idle >= p.stopTimeout) {
			p.popLocked()
			if p.live[c.id] == c {
				delete(p.live, c.id)
				if !c.done {
					if len(p.dormant) >= maxDormant {
						clear(p.dormant)
					}
					p.dormant[c.id] = true
				}
			}
			stopped := frames.NewTTSStoppedFrame()
			stopped.ContextID = c.id
			return stopped, true, 0
		}
		if timed {
			return nil, false, p.stopTimeout - idle
		}
		return nil, false, -1
	}
	p.releaseWaitersLocked()
	return nil, false, -1
}

func (p *playback) run(ctx context.Context) {
	timer := time.NewTimer(time.Hour)
	timer.Stop()
	for {
		p.pushMu.Lock()
		f, stopped, wait := p.next(time.Now())
		if f != nil {
			_ = p.b.PushFrame(f, frames.Downstream)
			if stopped {
				up := frames.NewTTSStoppedFrame()
				up.ContextID = f.(*frames.TTSStoppedFrame).ContextID
				_ = p.b.PushFrame(up, frames.Upstream)
			}
			p.pushMu.Unlock()
			continue
		}
		p.pushMu.Unlock()
		if wait > 0 {
			timer.Reset(wait)
			select {
			case <-p.wake:
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return
			}
			timer.Stop()
			continue
		}
		select {
		case <-p.wake:
		case <-ctx.Done():
			return
		}
	}
}
