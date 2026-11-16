package llmwithbackend

import (
	"crypto/sha256"
	"errors"
	"strings"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

var (
	ErrNothingNew   = errors.New("nothing new to delegate: the backend already has everything said since the previous delegation")
	ErrEmptyRequest = errors.New("the delegate call needs a non-empty request argument")
)

type RequestStrategy interface {
	Tool() llm.FunctionSchema
	FrontendInstruction() string
	ComposeRequest(params llm.FunctionCallParams) (string, error)
}

type TranscriptStrategy struct {
	Instruction        string
	Description        string
	RequestInstruction string

	mu       sync.Mutex
	consumed int
	digest   [sha256.Size]byte
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

func (s *TranscriptStrategy) ComposeRequest(params llm.FunctionCallParams) (string, error) {
	if params.Context == nil {
		return "", ErrNothingNew
	}
	lines := dialogueLines(params.Context.Messages())
	s.mu.Lock()
	if s.consumed > len(lines) || digest(lines[:s.consumed]) != s.digest {
		s.consumed = 0
	}
	from := s.consumed
	s.consumed, s.digest = len(lines), digest(lines)
	s.mu.Unlock()
	if from == len(lines) {
		return "", ErrNothingNew
	}
	heading := "Voice conversation since the previous delegation:"
	if from == 0 {
		heading = "Voice conversation so far:"
	}
	return heading + "\n" + strings.Join(lines[from:], "\n") + "\n\n" + s.RequestInstruction, nil
}

func dialogueLines(messages []llm.Message) []string {
	var lines []string
	for _, m := range messages {
		if m.Role != llm.RoleUser && m.Role != llm.RoleAssistant {
			continue
		}
		if text := strings.TrimSpace(m.Text()); text != "" {
			lines = append(lines, strings.ToUpper(string(m.Role))+": "+text)
		}
	}
	return lines
}

func digest(lines []string) [sha256.Size]byte {
	h := sha256.New()
	for _, l := range lines {
		h.Write([]byte(l))
		h.Write([]byte{0})
	}
	var out [sha256.Size]byte
	copy(out[:], h.Sum(nil))
	return out
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

func (s *ExplicitStrategy) ComposeRequest(params llm.FunctionCallParams) (string, error) {
	request, _ := params.Arguments["request"].(string)
	request = strings.TrimSpace(request)
	if request == "" {
		return "", ErrEmptyRequest
	}
	return request + "\n\n" + s.RequestInstruction, nil
}
