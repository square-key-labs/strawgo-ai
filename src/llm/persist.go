package llm

import (
	"encoding/json"
	"errors"
	"fmt"
)

const ContextFormatVersion = 1

var (
	ErrUnsupportedContextVersion = errors.New("llm: unsupported context format version")
	ErrUnknownPartType           = errors.New("llm: unknown message part type")
)

type MarshalOption func(*marshalConfig)

type marshalConfig struct {
	omitBinary bool
}

func OmitBinary() MarshalOption {
	return func(c *marshalConfig) { c.omitBinary = true }
}

type wireContext struct {
	Version    int             `json:"version"`
	Messages   []wireMessage   `json:"messages"`
	Tools      *wireTools      `json:"tools,omitempty"`
	ToolChoice *wireToolChoice `json:"tool_choice,omitempty"`
}

type wireMessage struct {
	Role  Role       `json:"role"`
	Name  string     `json:"name,omitempty"`
	Parts []wirePart `json:"parts"`
}

type wirePart struct {
	Type       string `json:"type"`
	Text       string `json:"text,omitempty"`
	URL        string `json:"url,omitempty"`
	Data       []byte `json:"data,omitempty"`
	MIME       string `json:"mime,omitempty"`
	Detail     string `json:"detail,omitempty"`
	SampleRate int    `json:"sample_rate,omitempty"`
	Name       string `json:"name,omitempty"`
	ID         string `json:"id,omitempty"`
	Arguments  string `json:"arguments,omitempty"`
	Signature  string `json:"signature,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	Content    string `json:"content,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`
	Redacted   bool   `json:"redacted,omitempty"`
	Provider   string `json:"provider,omitempty"`
	Raw        string `json:"raw,omitempty"`
}

type wireToolChoice struct {
	Mode ToolChoiceMode `json:"mode"`
	Name string         `json:"name,omitempty"`
}

type wireTools struct {
	Functions []wireFunction               `json:"functions,omitempty"`
	Custom    map[string][]json.RawMessage `json:"custom,omitempty"`
}

type wireFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Properties  map[string]any `json:"properties,omitempty"`
	Required    []string       `json:"required,omitempty"`
	Strict      bool           `json:"strict,omitempty"`
}

func MarshalContext(c *Context, opts ...MarshalOption) ([]byte, error) {
	return MarshalSnapshot(c.Snapshot(), opts...)
}

func MarshalSnapshot(s Snapshot, opts ...MarshalOption) ([]byte, error) {
	var cfg marshalConfig
	for _, o := range opts {
		o(&cfg)
	}
	w := wireContext{Version: ContextFormatVersion, Messages: make([]wireMessage, len(s.Messages))}
	for i, m := range s.Messages {
		wm := wireMessage{Role: m.Role, Name: m.Name, Parts: make([]wirePart, len(m.Parts))}
		for j, p := range m.Parts {
			wp, err := encodePart(p, cfg)
			if err != nil {
				return nil, fmt.Errorf("message %d part %d: %w", i, j, err)
			}
			wm.Parts[j] = wp
		}
		w.Messages[i] = wm
	}
	if !s.Tools.Empty() {
		wt := &wireTools{Custom: s.Tools.Custom}
		for _, f := range s.Tools.Functions {
			wt.Functions = append(wt.Functions, wireFunction(f))
		}
		w.Tools = wt
	}
	if s.ToolChoice != (ToolChoice{}) {
		w.ToolChoice = &wireToolChoice{Mode: s.ToolChoice.Mode, Name: s.ToolChoice.Name}
	}
	return json.Marshal(w)
}

func UnmarshalContext(data []byte) (*Context, error) {
	var w wireContext
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, fmt.Errorf("llm: decode context: %w", err)
	}
	if w.Version < 1 || w.Version > ContextFormatVersion {
		return nil, fmt.Errorf("%w: got %d, this build reads up to %d", ErrUnsupportedContextVersion, w.Version, ContextFormatVersion)
	}
	msgs := make([]Message, len(w.Messages))
	for i, wm := range w.Messages {
		m := Message{Role: wm.Role, Name: wm.Name}
		if len(wm.Parts) > 0 {
			m.Parts = make([]Part, len(wm.Parts))
		}
		for j, wp := range wm.Parts {
			p, err := decodePart(wp)
			if err != nil {
				return nil, fmt.Errorf("message %d part %d: %w", i, j, err)
			}
			m.Parts[j] = p
		}
		msgs[i] = m
	}
	c := NewContext(msgs...)
	if w.Tools != nil {
		c.tools.Custom = w.Tools.Custom
		for _, f := range w.Tools.Functions {
			c.tools.Functions = append(c.tools.Functions, FunctionSchema(f))
		}
	}
	if w.ToolChoice != nil {
		c.toolChoice = ToolChoice(*w.ToolChoice)
	}
	return c, nil
}

func encodePart(p Part, cfg marshalConfig) (wirePart, error) {
	switch v := p.(type) {
	case TextPart:
		return wirePart{Type: "text", Text: v.Text}, nil
	case ImagePart:
		if cfg.omitBinary && len(v.Data) > 0 {
			if v.URL == "" {
				return omittedPart("image", v.MIME, len(v.Data)), nil
			}
			v.Data = nil
		}
		return wirePart{Type: "image", URL: v.URL, Data: v.Data, MIME: v.MIME, Detail: v.Detail}, nil
	case AudioPart:
		if cfg.omitBinary && len(v.Data) > 0 {
			return omittedPart("audio", v.MIME, len(v.Data)), nil
		}
		return wirePart{Type: "audio", Data: v.Data, MIME: v.MIME, SampleRate: v.SampleRate}, nil
	case FilePart:
		if cfg.omitBinary && len(v.Data) > 0 {
			if v.URL == "" {
				return omittedPart("file", v.MIME, len(v.Data)), nil
			}
			v.Data = nil
		}
		return wirePart{Type: "file", URL: v.URL, Data: v.Data, MIME: v.MIME, Name: v.Name}, nil
	case ToolCallPart:
		return wirePart{Type: "tool_call", ID: v.ID, Name: v.Name, Arguments: string(v.Arguments), Signature: v.Signature}, nil
	case ToolResultPart:
		return wirePart{Type: "tool_result", ToolCallID: v.ToolCallID, Name: v.Name, Content: v.Content, IsError: v.IsError}, nil
	case ThoughtPart:
		return wirePart{Type: "thought", Text: v.Text, Signature: v.Signature, Redacted: v.Redacted}, nil
	case ProviderPart:
		return wirePart{Type: "provider", Provider: v.Provider, Raw: string(v.Raw)}, nil
	default:
		return wirePart{}, fmt.Errorf("%w: %T", ErrUnknownPartType, p)
	}
}

func omittedPart(kind, mime string, size int) wirePart {
	label := kind
	if mime != "" {
		label = mime + " " + kind
	}
	return wirePart{Type: "text", Text: fmt.Sprintf("[%s omitted, %d bytes]", label, size)}
}

func decodePart(w wirePart) (Part, error) {
	switch w.Type {
	case "text":
		return TextPart{Text: w.Text}, nil
	case "image":
		return ImagePart{URL: w.URL, Data: w.Data, MIME: w.MIME, Detail: w.Detail}, nil
	case "audio":
		return AudioPart{Data: w.Data, MIME: w.MIME, SampleRate: w.SampleRate}, nil
	case "file":
		return FilePart{URL: w.URL, Data: w.Data, MIME: w.MIME, Name: w.Name}, nil
	case "tool_call":
		return ToolCallPart{ID: w.ID, Name: w.Name, Arguments: rawOrNil(w.Arguments), Signature: w.Signature}, nil
	case "tool_result":
		return ToolResultPart{ToolCallID: w.ToolCallID, Name: w.Name, Content: w.Content, IsError: w.IsError}, nil
	case "thought":
		return ThoughtPart{Text: w.Text, Signature: w.Signature, Redacted: w.Redacted}, nil
	case "provider":
		return ProviderPart{Provider: w.Provider, Raw: rawOrNil(w.Raw)}, nil
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnknownPartType, w.Type)
	}
}

func rawOrNil(s string) json.RawMessage {
	if s == "" {
		return nil
	}
	return json.RawMessage(s)
}
