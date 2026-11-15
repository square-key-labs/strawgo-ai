package llm

import (
	"encoding/json"
	"strings"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Part interface {
	isPart()
}

type TextPart struct {
	Text string
}

type ImagePart struct {
	URL    string
	Data   []byte
	MIME   string
	Detail string
}

type AudioPart struct {
	Data       []byte
	MIME       string
	SampleRate int
}

type FilePart struct {
	URL  string
	Data []byte
	MIME string
	Name string
}

type ToolCallPart struct {
	ID        string
	Name      string
	Arguments json.RawMessage
	Signature string
}

type ToolResultPart struct {
	ToolCallID string
	Name       string
	Content    string
	IsError    bool
}

type ThoughtPart struct {
	Text      string
	Signature string
	Redacted  bool
}

type ProviderPart struct {
	Provider string
	Raw      json.RawMessage
}

func (TextPart) isPart()       {}
func (ImagePart) isPart()      {}
func (AudioPart) isPart()      {}
func (FilePart) isPart()       {}
func (ToolCallPart) isPart()   {}
func (ToolResultPart) isPart() {}
func (ThoughtPart) isPart()    {}
func (ProviderPart) isPart()   {}

type Message struct {
	Role  Role
	Name  string
	Parts []Part
}

func Text(role Role, text string) Message {
	return Message{Role: role, Parts: []Part{TextPart{Text: text}}}
}

func System(text string) Message    { return Text(RoleSystem, text) }
func Developer(text string) Message { return Text(RoleDeveloper, text) }
func User(text string) Message      { return Text(RoleUser, text) }
func Assistant(text string) Message { return Text(RoleAssistant, text) }

func ToolCalls(calls ...ToolCallPart) Message {
	parts := make([]Part, len(calls))
	for i, c := range calls {
		parts[i] = c
	}
	return Message{Role: RoleAssistant, Parts: parts}
}

func ToolResult(toolCallID, name, content string) Message {
	return Message{Role: RoleTool, Parts: []Part{ToolResultPart{ToolCallID: toolCallID, Name: name, Content: content}}}
}

func (m Message) Text() string {
	var b strings.Builder
	for _, p := range m.Parts {
		if t, ok := p.(TextPart); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func (m Message) ToolCalls() []ToolCallPart {
	var out []ToolCallPart
	for _, p := range m.Parts {
		if c, ok := p.(ToolCallPart); ok {
			out = append(out, c)
		}
	}
	return out
}

func (m Message) ToolResult() (ToolResultPart, bool) {
	for _, p := range m.Parts {
		if r, ok := p.(ToolResultPart); ok {
			return r, true
		}
	}
	return ToolResultPart{}, false
}

func (m Message) clone() Message {
	out := m
	out.Parts = append([]Part(nil), m.Parts...)
	return out
}
