package openaicompat

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

const AdapterID = "openai"

type Message struct {
	Role             string          `json:"role"`
	Content          any             `json:"content,omitempty"`
	Name             string          `json:"name,omitempty"`
	ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ReasoningContent *string         `json:"reasoning_content,omitempty"`
	Prefix           bool            `json:"prefix,omitempty"`
	Raw              json.RawMessage `json:"-"`
}

func (m Message) MarshalJSON() ([]byte, error) {
	if len(m.Raw) > 0 {
		return m.Raw, nil
	}
	type plain Message
	return json.Marshal(plain(m))
}

type ContentPart struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ImageURL   *ImageURL   `json:"image_url,omitempty"`
	InputAudio *InputAudio `json:"input_audio,omitempty"`
	File       *FileData   `json:"file,omitempty"`
}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type InputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

type FileData struct {
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolSpec struct {
	Type     string       `json:"type"`
	Function functionSpec `json:"function"`
}

type functionSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict,omitempty"`
}

func (p *Provider) convertMessages(snapshot llm.Snapshot, settings llm.Settings) ([]Message, error) {
	out := make([]Message, 0, len(snapshot.Messages)+1)
	if settings.SystemInstruction != "" {
		out = append(out, Message{Role: string(llm.RoleSystem), Content: settings.SystemInstruction})
	}
	for _, m := range snapshot.Messages {
		var err error
		out, err = p.appendMessage(out, m)
		if err != nil {
			return nil, err
		}
	}
	if p.opts.transformMessages != nil {
		out = p.opts.transformMessages(out, settings)
	}
	return out, nil
}

func (p *Provider) appendMessage(out []Message, m llm.Message) ([]Message, error) {
	var parts []ContentPart
	var calls []ToolCall
	var thought strings.Builder
	hasThought := false
	for _, part := range m.Parts {
		switch v := part.(type) {
		case llm.ProviderPart:
			if v.Provider == AdapterID && len(v.Raw) > 0 {
				out = append(out, Message{Raw: v.Raw})
			}
		case llm.ToolResultPart:
			out = append(out, Message{Role: string(llm.RoleTool), ToolCallID: v.ToolCallID, Content: v.Content})
		case llm.ToolCallPart:
			args := string(v.Arguments)
			if args == "" {
				args = "{}"
			}
			calls = append(calls, ToolCall{ID: v.ID, Type: "function", Function: FunctionCall{Name: v.Name, Arguments: args}})
		case llm.ThoughtPart:
			if p.opts.reasoningContent && !v.Redacted {
				thought.WriteString(v.Text)
				hasThought = true
			}
		case llm.TextPart:
			parts = append(parts, ContentPart{Type: "text", Text: v.Text})
		case llm.ImagePart:
			cp, err := imagePart(v)
			if err != nil {
				return nil, err
			}
			parts = append(parts, cp)
		case llm.AudioPart:
			cp, err := audioPart(v)
			if err != nil {
				return nil, err
			}
			parts = append(parts, cp)
		case llm.FilePart:
			cp, err := filePart(v)
			if err != nil {
				return nil, err
			}
			parts = append(parts, cp)
		}
	}
	if len(parts) == 0 && len(calls) == 0 && !hasThought {
		return out, nil
	}

	role := m.Role
	if role == llm.RoleDeveloper && p.opts.noDeveloperRole {
		role = llm.RoleUser
	}
	if role == llm.RoleTool {
		role = llm.RoleUser
	}
	msg := Message{Role: string(role), Name: m.Name}

	if role == llm.RoleAssistant {
		var text strings.Builder
		for _, cp := range parts {
			if cp.Type == "text" {
				text.WriteString(cp.Text)
			}
		}
		if text.Len() > 0 || len(calls) == 0 {
			msg.Content = text.String()
		}
		msg.ToolCalls = calls
		if p.opts.reasoningContent {
			r := thought.String()
			msg.ReasoningContent = &r
		}
		return append(out, msg), nil
	}

	if len(parts) == 0 {
		return out, nil
	}
	if len(parts) == 1 && parts[0].Type == "text" {
		msg.Content = parts[0].Text
	} else {
		msg.Content = parts
	}
	return append(out, msg), nil
}

func DataURL(mime string, data []byte) string {
	var b strings.Builder
	b.Grow(len(mime) + 13 + base64.StdEncoding.EncodedLen(len(data)))
	b.WriteString("data:")
	b.WriteString(mime)
	b.WriteString(";base64,")
	b.WriteString(base64.StdEncoding.EncodeToString(data))
	return b.String()
}

func imagePart(v llm.ImagePart) (ContentPart, error) {
	url := v.URL
	if len(v.Data) > 0 {
		mime := v.MIME
		if mime == "" {
			mime = "image/jpeg"
		}
		url = DataURL(mime, v.Data)
	}
	if url == "" {
		return ContentPart{}, fmt.Errorf("openaicompat: image part has neither URL nor data")
	}
	return ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: url, Detail: v.Detail}}, nil
}

func audioPart(v llm.AudioPart) (ContentPart, error) {
	if len(v.Data) == 0 {
		return ContentPart{}, fmt.Errorf("openaicompat: audio part has no data")
	}
	mime := strings.ToLower(v.MIME)
	data := v.Data
	var format string
	switch {
	case strings.Contains(mime, "wav"):
		format = "wav"
	case strings.Contains(mime, "mpeg"), strings.Contains(mime, "mp3"):
		format = "mp3"
	case mime == "", strings.Contains(mime, "pcm"), strings.Contains(mime, "l16"):
		if v.SampleRate <= 0 {
			return ContentPart{}, fmt.Errorf("openaicompat: raw PCM audio part needs a sample rate")
		}
		data = wavFromPCM16(v.Data, v.SampleRate)
		format = "wav"
	default:
		return ContentPart{}, fmt.Errorf("openaicompat: unsupported audio MIME type %q", v.MIME)
	}
	return ContentPart{Type: "input_audio", InputAudio: &InputAudio{Data: base64.StdEncoding.EncodeToString(data), Format: format}}, nil
}

func wavFromPCM16(pcm []byte, sampleRate int) []byte {
	out := make([]byte, 44+len(pcm))
	copy(out[0:], "RIFF")
	binary.LittleEndian.PutUint32(out[4:], uint32(36+len(pcm)))
	copy(out[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(out[16:], 16)
	binary.LittleEndian.PutUint16(out[20:], 1)
	binary.LittleEndian.PutUint16(out[22:], 1)
	binary.LittleEndian.PutUint32(out[24:], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[28:], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(out[32:], 2)
	binary.LittleEndian.PutUint16(out[34:], 16)
	copy(out[36:], "data")
	binary.LittleEndian.PutUint32(out[40:], uint32(len(pcm)))
	copy(out[44:], pcm)
	return out
}

func filePart(v llm.FilePart) (ContentPart, error) {
	if strings.HasPrefix(v.MIME, "image/") {
		return imagePart(llm.ImagePart{URL: v.URL, Data: v.Data, MIME: v.MIME})
	}
	if len(v.Data) == 0 {
		return ContentPart{}, fmt.Errorf("openaicompat: file %q must be inlined; URLs are only supported for images", v.Name)
	}
	if v.MIME != "application/pdf" {
		return ContentPart{}, fmt.Errorf("openaicompat: unsupported file MIME type %q", v.MIME)
	}
	return ContentPart{Type: "file", File: &FileData{Filename: v.Name, FileData: DataURL(v.MIME, v.Data)}}, nil
}

func convertTools(tools llm.Tools) []any {
	custom := tools.Custom[AdapterID]
	if len(tools.Functions) == 0 && len(custom) == 0 {
		return nil
	}
	out := make([]any, 0, len(tools.Functions)+len(custom))
	for _, f := range tools.Functions {
		out = append(out, toolSpec{Type: "function", Function: functionSpec{
			Name:        f.Name,
			Description: f.Description,
			Parameters:  f.JSONSchema(),
			Strict:      f.Strict,
		}})
	}
	for _, raw := range custom {
		out = append(out, raw)
	}
	return out
}

func convertToolChoice(choice llm.ToolChoice) any {
	switch choice.Mode {
	case llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
		return string(choice.Mode)
	case llm.ToolChoiceFunction:
		return map[string]any{"type": "function", "function": map[string]string{"name": choice.Name}}
	}
	return nil
}
