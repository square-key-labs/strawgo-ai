package llmservice

import (
	"slices"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type builtins struct {
	tools        []llm.FunctionSchema
	instructions []string
}

func (s *Service) AddBuiltinTool(schema llm.FunctionSchema) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.builtins.tools = slices.DeleteFunc(slices.Clone(s.builtins.tools), func(f llm.FunctionSchema) bool { return f.Name == schema.Name })
	s.builtins.tools = append(s.builtins.tools, schema)
}

func (s *Service) BuiltinTools() []llm.FunctionSchema {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.builtins.tools)
}

func (s *Service) AppendSystemInstruction(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	s.mu.Lock()
	s.builtins.instructions = append(slices.Clone(s.builtins.instructions), text)
	s.mu.Unlock()
}

func (s *Service) LookupFunction(name string) (llm.FunctionHandler, llm.FunctionOptions, bool) {
	reg, ok := s.lookup(name)
	return reg.handler, reg.options, ok
}

func (s *Service) withBuiltins(snapshot llm.Snapshot, settings llm.Settings) (llm.Snapshot, llm.Settings) {
	s.mu.RLock()
	b := s.builtins
	s.mu.RUnlock()
	if len(b.instructions) > 0 {
		settings.SystemInstruction = joinInstructions(append([]string{settings.SystemInstruction}, b.instructions...)...)
	}
	if len(b.tools) == 0 {
		return snapshot, settings
	}
	functions := slices.Clone(snapshot.Tools.Functions)
	for _, tool := range b.tools {
		if !slices.ContainsFunc(functions, func(f llm.FunctionSchema) bool { return f.Name == tool.Name }) {
			functions = append(functions, tool)
		}
	}
	snapshot.Tools.Functions = functions
	return snapshot, settings
}

func joinInstructions(parts ...string) string {
	var kept []string
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, "\n\n")
}
