package llmwithbackend

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type backendTurn struct {
	text      strings.Builder
	thought   strings.Builder
	signature string
	calls     []llm.ToolCall
}

func (t *backendTurn) message() llm.Message {
	m := llm.Message{Role: llm.RoleAssistant}
	if t.thought.Len() > 0 || t.signature != "" {
		m.Parts = append(m.Parts, llm.ThoughtPart{Text: t.thought.String(), Signature: t.signature})
	}
	if t.text.Len() > 0 {
		m.Parts = append(m.Parts, llm.TextPart{Text: t.text.String()})
	}
	for _, c := range t.calls {
		m.Parts = append(m.Parts, llm.ToolCallPart{ID: c.ID, Name: c.Name, Arguments: json.RawMessage(c.Arguments), Signature: c.Signature})
	}
	return m
}

type callOutcome struct {
	content string
	isError bool
	runLLM  bool
}

func (w *LLMWithBackend) work(ctx context.Context, params llm.FunctionCallParams) (string, error) {
	settings := w.backend.Settings()
	settings.SystemInstruction = joinNonEmpty(settings.SystemInstruction, w.backendPairing)
	for step := 0; step < w.maxSteps; step++ {
		turn, err := w.infer(ctx, settings)
		if err != nil {
			return "", err
		}
		text := strings.TrimSpace(turn.text.String())
		if msg := turn.message(); len(msg.Parts) > 0 {
			w.backendContext.Add(msg)
		}
		if len(turn.calls) == 0 {
			return text, nil
		}
		if text != "" {
			params.ReportProgress(llm.FunctionResult{Value: BackendNotePrefix + text})
		}
		if !w.runCalls(ctx, params, turn.calls) {
			return "", nil
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}
	}
	return "", fmt.Errorf("%s did not finish within %d steps", w.backend.Name(), w.maxSteps)
}

func (w *LLMWithBackend) infer(ctx context.Context, settings llm.Settings) (*backendTurn, error) {
	cfg := w.StartConfig()
	base := frames.MetricsBase{Processor: w.backend.Name(), Model: settings.Model}
	turn := &backendTurn{}
	started := time.Now()
	var ttfb time.Duration
	var usage llm.Usage
	gotUsage := false
	var refusal *llm.Refusal
	first := func() {
		if ttfb == 0 {
			ttfb = time.Since(started)
		}
	}
	err := w.backend.Provider().Stream(ctx, w.backendContext.Snapshot(), settings, func(ev llm.Event) {
		switch e := ev.(type) {
		case llm.TextDelta:
			if e.Text != "" {
				first()
				turn.text.WriteString(e.Text)
			}
		case llm.ThoughtDelta:
			first()
			turn.thought.WriteString(e.Text)
			if e.Signature != "" {
				turn.signature = e.Signature
			}
		case llm.ToolCall:
			first()
			turn.calls = append(turn.calls, e)
		case llm.UsageEvent:
			usage, gotUsage = e.Usage, true
		case llm.Refusal:
			if refusal == nil {
				refusal = &e
			}
		}
	})
	if err == nil && refusal != nil {
		err = &llm.RefusalError{Refusal: *refusal}
	}
	var metrics []frames.MetricsData
	if cfg.EnableMetrics {
		if ttfb > 0 {
			metrics = append(metrics, frames.TTFBMetricsData{MetricsBase: base, Value: ttfb})
		}
		metrics = append(metrics, frames.ProcessingMetricsData{MetricsBase: base, Value: time.Since(started)})
	}
	if cfg.EnableUsageMetrics && gotUsage {
		metrics = append(metrics, frames.LLMUsageMetricsData{MetricsBase: base, Value: usage})
	}
	if len(metrics) > 0 {
		_ = w.PushFrame(frames.NewMetricsFrame(metrics...), frames.Downstream)
	}
	if err != nil {
		return nil, err
	}
	return turn, nil
}

func (w *LLMWithBackend) runCalls(ctx context.Context, params llm.FunctionCallParams, calls []llm.ToolCall) bool {
	outcomes := make([]callOutcome, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		params.ReportProgress(llm.FunctionResult{Value: BackendWorkingPrefix + describeCall(call)})
		wg.Add(1)
		go func() {
			defer wg.Done()
			outcomes[i] = w.runCall(ctx, params, call)
		}()
	}
	wg.Wait()
	rerun := false
	for i, call := range calls {
		o := outcomes[i]
		w.backendContext.Add(llm.Message{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart{
			ToolCallID: call.ID, Name: call.Name, Content: o.content, IsError: o.isError,
		}}})
		rerun = rerun || o.runLLM
	}
	return rerun
}

func (w *LLMWithBackend) runCall(ctx context.Context, params llm.FunctionCallParams, call llm.ToolCall) callOutcome {
	handler, opts, ok := w.backend.LookupFunction(call.Name)
	if !ok {
		return failed(fmt.Errorf("function %q is not registered", call.Name))
	}
	args := map[string]any{}
	if len(call.Arguments) > 0 {
		if err := json.Unmarshal(call.Arguments, &args); err != nil {
			return failed(fmt.Errorf("invalid arguments for %q: %w", call.Name, err))
		}
		if args == nil {
			args = map[string]any{}
		}
	}
	callCtx, cancel := ctx, context.CancelFunc(func() {})
	if opts.Timeout > 0 {
		callCtx, cancel = context.WithTimeout(ctx, opts.Timeout)
	}
	defer cancel()

	type result struct {
		value llm.FunctionResult
		err   error
	}
	done := make(chan result, 1)
	go func() {
		var r result
		defer func() {
			if p := recover(); p != nil {
				r.err = fmt.Errorf("function %q panicked: %v", call.Name, p)
			}
			done <- r
		}()
		r.value, r.err = handler(callCtx, llm.FunctionCallParams{
			ToolCallID:   call.ID,
			FunctionName: call.Name,
			Arguments:    args,
			RawArguments: json.RawMessage(call.Arguments),
			Context:      w.backendContext,
			ReportProgress: func(progress llm.FunctionResult) bool {
				if callCtx.Err() != nil {
					return false
				}
				progress.Value = map[string]any{"backend_function": call.Name, "progress": progress.Value}
				return params.ReportProgress(progress)
			},
		})
	}()
	var r result
	select {
	case r = <-done:
	case <-callCtx.Done():
		select {
		case r = <-done:
		default:
			r.err = callCtx.Err()
		}
	}
	if ctx.Err() != nil && r.err != nil {
		return callOutcome{content: llm.ToolResultInterrupted, isError: true}
	}
	if r.err != nil {
		o := failed(r.err)
		if r.value.Value != nil {
			o.content = encodeResult(r.value.Value)
		}
		return o
	}
	return callOutcome{content: encodeResult(r.value.Value), runLLM: r.value.RunLLM == nil || *r.value.RunLLM}
}

func failed(err error) callOutcome {
	content, _ := json.Marshal(map[string]any{"error": err.Error()})
	return callOutcome{content: string(content), isError: true, runLLM: true}
}

func encodeResult(value any) string {
	switch v := value.(type) {
	case nil:
		return "COMPLETED"
	case string:
		encoded, _ := json.Marshal(v)
		return string(encoded)
	case json.RawMessage:
		return string(v)
	case []byte:
		return string(v)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(encoded)
}

func describeCall(call llm.ToolCall) string {
	var args map[string]any
	_ = json.Unmarshal(call.Arguments, &args)
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		v, _ := json.Marshal(args[k])
		parts[i] = k + "=" + string(v)
	}
	return call.Name + "(" + strings.Join(parts, ", ") + ")"
}

func (w *LLMWithBackend) reportError(err error) {
	category := frames.ClassifyError(err)
	msg := fmt.Sprintf("%s generation failed", w.backend.Name())
	if _, refused := llm.AsRefusal(err); refused {
		category = frames.ErrorCategoryContentFiltered
		msg = fmt.Sprintf("%s response refused", w.backend.Name())
	}
	f := frames.NewCategorizedErrorFrame(fmt.Errorf("%s: %w", msg, err), category)
	f.Processor = w.backend.Name()
	_ = w.PushErrorFrame(f)
}
