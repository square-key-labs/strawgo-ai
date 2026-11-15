package realtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type Broadcaster interface {
	Name() string
	Context() context.Context
	BroadcastFrame(ctx context.Context, ctor func() frames.Frame) error
}

type Outcome struct {
	Call      llm.ToolCall
	Value     any
	RunLLM    *bool
	Err       error
	Cancelled bool
}

type registration struct {
	handler llm.FunctionHandler
	options llm.FunctionOptions
}

type runningCall struct {
	cancel               context.CancelFunc
	cancelOnInterruption bool
}

type Functions struct {
	owner Broadcaster

	mu       sync.RWMutex
	fns      map[string]registration
	catchAll *registration
	timeout  time.Duration

	callsMu sync.Mutex
	calls   map[string]*runningCall
	wg      sync.WaitGroup
}

func NewFunctions(owner Broadcaster) *Functions {
	return &Functions{owner: owner, fns: make(map[string]registration), calls: make(map[string]*runningCall)}
}

func (f *Functions) SetTimeout(d time.Duration) {
	f.mu.Lock()
	f.timeout = d
	f.mu.Unlock()
}

func (f *Functions) Register(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions) {
	reg := registration{handler: handler, options: llm.DefaultFunctionOptions()}
	if len(opts) > 0 {
		reg.options = opts[0]
	}
	f.mu.Lock()
	if name == "" {
		f.catchAll = &reg
	} else {
		f.fns[name] = reg
	}
	f.mu.Unlock()
}

func (f *Functions) Unregister(name string) {
	f.mu.Lock()
	if name == "" {
		f.catchAll = nil
	} else {
		delete(f.fns, name)
	}
	f.mu.Unlock()
}

func (f *Functions) Has(name string) bool {
	_, ok := f.lookup(name)
	return ok
}

func (f *Functions) lookup(name string) (registration, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if reg, ok := f.fns[name]; ok {
		return reg, true
	}
	if f.catchAll != nil {
		return *f.catchAll, true
	}
	return registration{}, false
}

func (f *Functions) broadcast(build func() frames.Frame) {
	if err := f.owner.BroadcastFrame(nil, build); err != nil {
		logger.Error("[%s] broadcast failed: %v", f.owner.Name(), err)
	}
}

func (f *Functions) Dispatch(c *llm.Context, calls []llm.ToolCall, done func(Outcome)) {
	if len(calls) == 0 {
		return
	}
	lifetime := f.owner.Context()
	infos := make([]frames.FunctionCallInfo, len(calls))
	for i, call := range calls {
		infos[i] = frames.FunctionCallInfo{ToolCallID: call.ID, FunctionName: call.Name}
	}
	f.broadcast(func() frames.Frame { return frames.NewFunctionCallsStartedFrame(infos) })

	f.mu.RLock()
	defaultTimeout := f.timeout
	f.mu.RUnlock()
	groupID := fmt.Sprintf("%s-%d", f.owner.Name(), time.Now().UnixNano())

	for _, call := range calls {
		reg, ok := f.lookup(call.Name)
		var args map[string]any
		var argErr error
		if len(call.Arguments) > 0 {
			argErr = json.Unmarshal(call.Arguments, &args)
		}
		if args == nil {
			args = map[string]any{}
		}
		timeout := reg.options.Timeout
		if timeout == 0 {
			timeout = defaultTimeout
		}
		var ctx context.Context
		var cancel context.CancelFunc
		if timeout > 0 {
			ctx, cancel = context.WithTimeout(lifetime, timeout)
		} else {
			ctx, cancel = context.WithCancel(lifetime)
		}
		cancelOnInterruption := reg.options.CancelOnInterruption
		f.callsMu.Lock()
		f.calls[call.ID] = &runningCall{cancel: cancel, cancelOnInterruption: cancelOnInterruption}
		f.callsMu.Unlock()

		call := call
		f.broadcast(func() frames.Frame {
			fr := frames.NewFunctionCallInProgressFrame(call.ID, call.Name, args, cancelOnInterruption)
			fr.GroupID = groupID
			return fr
		})

		f.wg.Add(1)
		go func() {
			defer f.wg.Done()
			result, err := execute(ctx, c, call, reg, ok, args, argErr)
			cancelled := err != nil && errors.Is(ctx.Err(), context.Canceled)

			f.callsMu.Lock()
			delete(f.calls, call.ID)
			f.callsMu.Unlock()
			cancel()

			if done != nil {
				done(Outcome{Call: call, Value: result.Value, RunLLM: result.RunLLM, Err: err, Cancelled: cancelled})
			}
			if cancelled {
				f.broadcast(func() frames.Frame { return frames.NewFunctionCallCancelFrame(call.ID, call.Name) })
				return
			}
			f.broadcast(func() frames.Frame {
				fr := frames.NewFunctionCallResultFrame(call.ID, call.Name, result.Value, result.RunLLM)
				fr.Arguments = args
				if err != nil {
					fr.Error = err.Error()
					if fr.Result == nil {
						fr.Result = map[string]any{"error": err.Error()}
					}
				}
				return fr
			})
		}()
	}
}

func execute(ctx context.Context, c *llm.Context, call llm.ToolCall, reg registration, registered bool, args map[string]any, argErr error) (result llm.FunctionResult, err error) {
	if err := ctx.Err(); err != nil {
		return llm.FunctionResult{}, err
	}
	if !registered {
		return llm.FunctionResult{}, fmt.Errorf("function %q is not registered", call.Name)
	}
	if argErr != nil {
		return llm.FunctionResult{}, fmt.Errorf("invalid arguments for %q: %w", call.Name, argErr)
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("function %q panicked: %v", call.Name, r)
		}
	}()
	return reg.handler(ctx, llm.FunctionCallParams{
		ToolCallID:   call.ID,
		FunctionName: call.Name,
		Arguments:    args,
		RawArguments: json.RawMessage(call.Arguments),
		Context:      c,
	})
}

func (f *Functions) Cancel(onlyInterruptible bool) {
	f.callsMu.Lock()
	var cancels []context.CancelFunc
	for _, rc := range f.calls {
		if onlyInterruptible && !rc.cancelOnInterruption {
			continue
		}
		cancels = append(cancels, rc.cancel)
	}
	f.callsMu.Unlock()
	for _, cancel := range cancels {
		cancel()
	}
}

func (f *Functions) CancelIDs(ids []string) {
	f.callsMu.Lock()
	var cancels []context.CancelFunc
	for _, id := range ids {
		if rc := f.calls[id]; rc != nil {
			cancels = append(cancels, rc.cancel)
		}
	}
	f.callsMu.Unlock()
	for _, cancel := range cancels {
		cancel()
	}
}

func (f *Functions) Wait() {
	f.wg.Wait()
}

func ResultString(value any, err error) string {
	if err != nil {
		b, _ := json.Marshal(map[string]any{"error": err.Error()})
		return string(b)
	}
	switch v := value.(type) {
	case nil:
		return "COMPLETED"
	case string:
		return v
	case json.RawMessage:
		return string(v)
	case []byte:
		return string(v)
	}
	b, merr := json.Marshal(value)
	if merr != nil {
		return fmt.Sprint(value)
	}
	return string(b)
}
