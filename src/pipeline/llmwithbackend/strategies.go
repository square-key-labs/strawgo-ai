package llmwithbackend

import (
	"strings"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type RequestStrategy interface {
	Tool() llm.FunctionSchema
	FrontendInstruction() string
	ComposeRequest(params llm.FunctionCallParams) (string, bool)
}

type TranscriptStrategy struct {
	Instruction        string
	Description        string
	RequestInstruction string

	mu      sync.Mutex
	through int
}

func NewTranscriptStrategy() *TranscriptStrategy {
	return &TranscriptStrategy{
		Instruction:        TranscriptFrontendInstruction,
		Description:        TranscriptToolDescription,
		RequestInstruction: TranscriptRequestInstruction,
	}
}

func (s *TranscriptStrategy) Tool() llm.FunctionSchema {
	return llm.FunctionSchema{Name: DelegateToolName, Description: s.Description}
}

func (s *TranscriptStrategy) FrontendInstruction() string {
	return s.Instruction
}

func (s *TranscriptStrategy) ComposeRequest(params llm.FunctionCallParams) (string, bool) {
	if params.Context == nil {
		return "", false
	}
	messages := params.Context.Messages()
	s.mu.Lock()
	if s.through > len(messages) {
		s.through = 0
	}
	from, first := s.through, s.through == 0
	s.through = len(messages)
	s.mu.Unlock()
	return renderTranscript(messages[from:], s.RequestInstruction, first)
}

func renderTranscript(messages []llm.Message, instruction string, first bool) (string, bool) {
	var lines []string
	for _, m := range messages {
		if m.Role != llm.RoleUser && m.Role != llm.RoleAssistant {
			continue
		}
		if text := strings.TrimSpace(m.Text()); text != "" {
			lines = append(lines, strings.ToUpper(string(m.Role))+": "+text)
		}
	}
	if len(lines) == 0 {
		return "", false
	}
	heading := "Voice conversation since the previous delegation:"
	if first {
		heading = "Voice conversation so far:"
	}
	return heading + "\n" + strings.Join(lines, "\n") + "\n\n" + instruction, true
}

type ExplicitStrategy struct {
	Instruction        string
	Description        string
	RequestInstruction string
}

func NewExplicitStrategy() *ExplicitStrategy {
	return &ExplicitStrategy{
		Instruction:        ExplicitFrontendInstruction,
		Description:        ExplicitToolDescription,
		RequestInstruction: ExplicitRequestInstruction,
	}
}

func (s *ExplicitStrategy) Tool() llm.FunctionSchema {
	return llm.FunctionSchema{
		Name:        DelegateToolName,
		Description: s.Description,
		Properties: map[string]any{
			"request": map[string]any{"type": "string", "description": ExplicitRequestDescription},
		},
		Required: []string{"request"},
	}
}

func (s *ExplicitStrategy) FrontendInstruction() string {
	return s.Instruction
}

func (s *ExplicitStrategy) ComposeRequest(params llm.FunctionCallParams) (string, bool) {
	request, _ := params.Arguments["request"].(string)
	request = strings.TrimSpace(request)
	if request == "" {
		return "", false
	}
	return request + "\n\n" + s.RequestInstruction, true
}
