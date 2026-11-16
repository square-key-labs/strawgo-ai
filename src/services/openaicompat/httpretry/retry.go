package httpretry

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const (
	DefaultBackoff    = 200 * time.Millisecond
	DefaultMaxBackoff = 2 * time.Second
)

var ErrFirstByteTimeout = errors.New("no response byte before the first-byte timeout")

type Policy struct {
	MaxAttempts      int
	FirstByteTimeout time.Duration
	Backoff          time.Duration
	MaxBackoff       time.Duration
}

func (p Policy) Enabled() bool { return p.MaxAttempts > 1 }

type RequestFunc func(ctx context.Context) (*http.Request, error)

func Do(ctx context.Context, client *http.Client, policy Policy, build RequestFunc) (*http.Response, error) {
	if !policy.Enabled() {
		req, err := build(ctx)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		return resp, logger.RedactError(err)
	}
	backoff := policy.Backoff
	if backoff <= 0 {
		backoff = DefaultBackoff
	}
	maxBackoff := policy.MaxBackoff
	if maxBackoff <= 0 {
		maxBackoff = DefaultMaxBackoff
	}
	for attempt := 1; ; attempt++ {
		resp, err, retry := attemptOnce(ctx, client, policy.FirstByteTimeout, build)
		if err == nil {
			return resp, nil
		}
		if !retry || attempt >= policy.MaxAttempts {
			if attempt > 1 {
				return nil, fmt.Errorf("%w (after %d attempts)", err, attempt)
			}
			return nil, err
		}
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
		backoff = min(backoff*2, maxBackoff)
	}
}

func attemptOnce(ctx context.Context, client *http.Client, timeout time.Duration, build RequestFunc) (*http.Response, error, bool) {
	actx, cancel := context.WithCancel(ctx)
	var timedOut atomic.Bool
	var timer *time.Timer
	if timeout > 0 {
		timer = time.AfterFunc(timeout, func() {
			timedOut.Store(true)
			cancel()
		})
	}
	fail := func(err error) (*http.Response, error, bool) {
		cancel()
		if timer != nil {
			timer.Stop()
		}
		if ctx.Err() != nil {
			return nil, ctx.Err(), false
		}
		if timedOut.Load() {
			return nil, fmt.Errorf("%w (%s)", ErrFirstByteTimeout, timeout), true
		}
		return nil, err, Retryable(err)
	}

	req, err := build(actx)
	if err != nil {
		cancel()
		if timer != nil {
			timer.Stop()
		}
		return nil, err, false
	}
	resp, err := client.Do(req)
	if err != nil {
		return fail(logger.RedactError(err))
	}
	b := &body{src: resp.Body, cancel: cancel}
	n, err := io.ReadFull(resp.Body, b.first[:])
	b.pending = n == 1
	if err != nil && !errors.Is(err, io.EOF) {
		resp.Body.Close()
		return fail(err)
	}
	if timer != nil && !timer.Stop() && timedOut.Load() {
		resp.Body.Close()
		return fail(ErrFirstByteTimeout)
	}
	resp.Body = b
	return resp, nil, false
}

func Retryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	var op *net.OpError
	return errors.As(err, &op) && op.Op == "dial"
}

type body struct {
	src     io.ReadCloser
	cancel  context.CancelFunc
	first   [1]byte
	pending bool
}

func (b *body) Read(p []byte) (int, error) {
	if b.pending {
		if len(p) == 0 {
			return 0, nil
		}
		p[0] = b.first[0]
		b.pending = false
		return 1, nil
	}
	return b.src.Read(p)
}

func (b *body) Close() error {
	err := b.src.Close()
	b.cancel()
	return err
}
