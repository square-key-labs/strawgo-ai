package filters

import "context"

type Notifier struct {
	ch chan struct{}
}

func NewNotifier() *Notifier {
	return &Notifier{ch: make(chan struct{}, 1)}
}

func (n *Notifier) Notify() {
	select {
	case n.ch <- struct{}{}:
	default:
	}
}

func (n *Notifier) Wait(ctx context.Context) error {
	select {
	case <-n.ch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (n *Notifier) C() <-chan struct{} {
	return n.ch
}
