package pipeline

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

type Runner struct {
	handleSignals bool
}

type RunnerOption func(*Runner)

func WithSignalHandling(enabled bool) RunnerOption {
	return func(r *Runner) { r.handleSignals = enabled }
}

func NewRunner(opts ...RunnerOption) *Runner {
	r := &Runner{handleSignals: true}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

func (r *Runner) Run(ctx context.Context, tasks ...*PipelineTask) error {
	for _, task := range tasks {
		if task == nil {
			return errors.New("pipeline: Runner.Run given a nil task")
		}
	}
	if len(tasks) == 0 {
		return nil
	}

	stop := context.CancelFunc(func() {})
	if r.handleSignals {
		ctx, stop = signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	}
	defer stop()

	runCtx := context.WithoutCancel(ctx)
	cancelAll := func() {
		for _, task := range tasks {
			task.Cancel()
		}
	}

	var (
		wg       sync.WaitGroup
		errOnce  sync.Once
		firstErr error
	)
	wg.Add(len(tasks))
	for _, task := range tasks {
		go func(task *PipelineTask) {
			defer wg.Done()
			if err := task.Run(runCtx); err != nil {
				errOnce.Do(func() {
					firstErr = err
					var fatal *FatalError
					if !errors.As(err, &fatal) {
						cancelAll()
					}
				})
			}
		}(task)
	}

	finished := make(chan struct{})
	go func() {
		wg.Wait()
		close(finished)
	}()

	select {
	case <-finished:
	case <-ctx.Done():
		stop()
		cancelAll()
		<-finished
	}
	return firstErr
}
