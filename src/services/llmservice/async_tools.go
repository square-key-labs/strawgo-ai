package llmservice

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const cancelSettleWait = time.Second

func cancellableByLLM(opts llm.FunctionOptions) bool {
	return opts.CancellableByLLM && !opts.CancelOnInterruption
}

func resultDelivery(result llm.FunctionResult, opts llm.FunctionOptions) (llm.ResultDelivery, *bool) {
	delivery := result.Delivery
	if delivery == llm.ResultDeliveryDefault {
		delivery = opts.Delivery
	}
	if delivery == llm.ResultDeliverySilent {
		no := false
		return delivery, &no
	}
	return delivery, result.RunLLM
}

func (s *Service) resolve(name string) (registration, bool) {
	s.mu.RLock()
	reg, ok := s.functions[name]
	s.mu.RUnlock()
	if ok {
		return reg, true
	}
	if target, ok := s.cancelTarget(name); ok {
		return registration{
			handler: s.cancelToolHandler(target),
			options: llm.FunctionOptions{CancelOnInterruption: true},
		}, true
	}
	return s.lookup(name)
}

func (s *Service) cancelTarget(name string) (string, bool) {
	target, ok := strings.CutPrefix(name, llm.CancelToolPrefix)
	if !ok || target == "" {
		return "", false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if _, taken := s.functions[name]; taken {
		return "", false
	}
	reg, ok := s.functions[target]
	return target, ok && cancellableByLLM(reg.options)
}

func (s *Service) withCancelTools(snapshot llm.Snapshot) llm.Snapshot {
	s.callsMu.Lock()
	var running []string
	for _, rc := range s.calls {
		if rc.cancellableByLLM && !slices.Contains(running, rc.name) {
			running = append(running, rc.name)
		}
	}
	s.callsMu.Unlock()
	if len(running) == 0 {
		return snapshot
	}
	slices.Sort(running)

	functions := slices.Clone(snapshot.Tools.Functions)
	for _, name := range running {
		cancelName := llm.CancelToolName(name)
		if _, ok := s.cancelTarget(cancelName); !ok {
			continue
		}
		if slices.ContainsFunc(functions, func(f llm.FunctionSchema) bool { return f.Name == cancelName }) {
			logger.Warn("[%s] %s is cancellable but %s is already a tool of its own; the model cannot stop it", s.Name(), name, cancelName)
			continue
		}
		functions = append(functions, llm.CancelToolSchema(name))
	}
	snapshot.Tools.Functions = functions
	return snapshot
}

func (s *Service) checkOptions(name string, opts llm.FunctionOptions) {
	if opts.CancellableByLLM && opts.CancelOnInterruption {
		logger.Warn("[%s] %q sets CancellableByLLM with CancelOnInterruption; the model cannot cancel a call it is waiting on, so no cancel tool is offered", s.Name(), name)
	}
}

func refusal(reason string, extra map[string]any) llm.FunctionResult {
	value := map[string]any{"cancelled": nil, "reason": reason}
	for k, v := range extra {
		value[k] = v
	}
	return llm.FunctionResult{Value: value}
}

func (s *Service) cancelToolHandler(target string) llm.FunctionHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) (llm.FunctionResult, error) {
		id, _ := params.Arguments["tool_call_id"].(string)

		s.callsMu.Lock()
		var candidates []*runningCall
		for _, rc := range s.calls {
			if rc.name == target && rc.cancellableByLLM && !rc.cancelledByLLM {
				candidates = append(candidates, rc)
			}
		}
		slices.SortFunc(candidates, func(a, b *runningCall) int { return int(a.seq) - int(b.seq) })
		var match *runningCall
		switch {
		case len(candidates) == 0:
		case id != "":
			for _, rc := range candidates {
				if rc.id == id {
					match = rc
				}
			}
		case len(candidates) == 1:
			match = candidates[0]
		}
		if match != nil {
			match.cancelledByLLM = true
		}
		s.callsMu.Unlock()

		switch {
		case len(candidates) == 0:
			return refusal(fmt.Sprintf("no %s call is running", target), nil), nil
		case match == nil && id != "":
			return refusal(fmt.Sprintf("no running %s call has that tool_call_id", target), nil), nil
		case match == nil:
			choices := make([]string, len(candidates))
			running := make([]map[string]any, len(candidates))
			for i, rc := range candidates {
				args, _ := json.Marshal(rc.args)
				choices[i] = fmt.Sprintf("%s (called with %s)", rc.id, args)
				running[i] = map[string]any{"tool_call_id": rc.id, "arguments": rc.args}
			}
			reason := fmt.Sprintf("%d %s calls are running: %s. Call %s again with tool_call_id set to the one to stop.",
				len(candidates), target, strings.Join(choices, "; "), params.FunctionName)
			return refusal(reason, map[string]any{"running": running}), nil
		}

		logger.Debug("[%s] model cancelled %s (%s)", s.Name(), target, match.id)
		match.cancel()
		timer := time.NewTimer(cancelSettleWait)
		defer timer.Stop()
		select {
		case <-match.done:
		case <-ctx.Done():
		case <-timer.C:
		}
		return llm.FunctionResult{Value: map[string]any{"cancelled": match.id, "function_name": target}}, nil
	}
}

func (s *Service) reportProgress(call llm.ToolCall, args map[string]any, opts llm.FunctionOptions, progress llm.FunctionResult) bool {
	s.callsMu.Lock()
	rc, running := s.calls[call.ID]
	running = running && !rc.cancelledByLLM
	s.callsMu.Unlock()
	if !running {
		return false
	}
	wantsRun := progress.RunLLM != nil && *progress.RunLLM
	delivery := progress.Delivery
	if wantsRun && delivery == llm.ResultDeliveryDefault {
		delivery = opts.Delivery
	}
	run := wantsRun && delivery != llm.ResultDeliverySilent
	if !run {
		delivery = llm.ResultDeliverySilent
	}
	s.broadcast(func() frames.Frame {
		f := frames.NewFunctionCallResultFrame(call.ID, call.Name, progress.Value, &run)
		f.Arguments = args
		f.Delivery = delivery
		f.Properties = &frames.FunctionCallResultProperties{IsFinal: false, RunLLM: &run}
		return f
	})
	return true
}
