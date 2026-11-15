package transcribe

import "sync"

type audioRing struct {
	mu      sync.Mutex
	buf     []byte
	head    int
	n       int
	dropped int64
	notify  chan struct{}
}

func newAudioRing(size int) *audioRing {
	size &^= 1
	if size < 2 {
		size = 2
	}
	return &audioRing{buf: make([]byte, size), notify: make(chan struct{}, 1)}
}

func (r *audioRing) write(p []byte) {
	p = p[:len(p)&^1]
	if len(p) == 0 {
		return
	}
	size := len(r.buf)
	r.mu.Lock()
	if len(p) > size {
		r.dropped += int64(len(p) - size)
		p = p[len(p)-size:]
	}
	if over := r.n + len(p) - size; over > 0 {
		r.head = (r.head + over) % size
		r.n -= over
		r.dropped += int64(over)
	}
	tail := (r.head + r.n) % size
	c := copy(r.buf[tail:], p)
	copy(r.buf, p[c:])
	r.n += len(p)
	r.mu.Unlock()
	select {
	case r.notify <- struct{}{}:
	default:
	}
}

func (r *audioRing) read(dst []byte) int {
	r.mu.Lock()
	n := min(len(dst)&^1, r.n)
	c := copy(dst[:n], r.buf[r.head:])
	copy(dst[c:n], r.buf)
	r.head = (r.head + n) % len(r.buf)
	r.n -= n
	r.mu.Unlock()
	return n
}

func (r *audioRing) takeDropped() int64 {
	r.mu.Lock()
	d := r.dropped
	r.dropped = 0
	r.mu.Unlock()
	return d
}
