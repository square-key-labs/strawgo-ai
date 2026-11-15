package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat/httpretry"
)

const ResponsesAdapterID = "openai.responses"

type ResponsesProvider struct {
	name     string
	endpoint string
	apiKey   string
	headers  http.Header
	client   *http.Client
	retry    openaicompat.RetryPolicy
}

func NewResponsesProvider(cfg LLMConfig) *ResponsesProvider {
	pcfg, _ := cfg.Resolve(defaults)
	p := &ResponsesProvider{
		name:     "OpenAIResponses",
		endpoint: strings.TrimRight(pcfg.BaseURL, "/") + "/responses",
		apiKey:   pcfg.APIKey,
		headers:  make(http.Header, len(pcfg.Headers)),
		client:   pcfg.HTTPClient,
		retry:    pcfg.Retry,
	}
	if p.client == nil {
		p.client = openaicompat.DefaultHTTPClient()
	}
	for k, v := range pcfg.Headers {
		p.headers.Set(k, v)
	}
	return p
}

func NewResponsesLLMService(cfg LLMConfig) *llmservice.Service {
	_, settings := cfg.Resolve(defaults)
	return openaicompat.Wrap(NewResponsesProvider(cfg), settings, cfg.ServiceOptions...)
}

func (p *ResponsesProvider) Name() string { return p.name }

type respContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
}

type respItem struct {
	Type      string          `json:"type,omitempty"`
	Role      string          `json:"role,omitempty"`
	Content   any             `json:"content,omitempty"`
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Output    *string         `json:"output,omitempty"`
	Raw       json.RawMessage `json:"-"`
}

func (it respItem) MarshalJSON() ([]byte, error) {
	if len(it.Raw) > 0 {
		return it.Raw, nil
	}
	type plain respItem
	return json.Marshal(plain(it))
}

func responsesInput(snapshot llm.Snapshot) ([]respItem, error) {
	items := make([]respItem, 0, len(snapshot.Messages))
	for _, m := range snapshot.Messages {
		var content []respContent
		var calls []respItem
		for _, part := range m.Parts {
			switch v := part.(type) {
			case llm.ProviderPart:
				if v.Provider == ResponsesAdapterID && len(v.Raw) > 0 {
					items = append(items, respItem{Raw: v.Raw})
				}
			case llm.ToolResultPart:
				out := v.Content
				items = append(items, respItem{Type: "function_call_output", CallID: v.ToolCallID, Output: &out})
			case llm.ToolCallPart:
				args := string(v.Arguments)
				if args == "" {
					args = "{}"
				}
				calls = append(calls, respItem{Type: "function_call", CallID: v.ID, Name: v.Name, Arguments: args})
			case llm.TextPart:
				typ := "input_text"
				if m.Role == llm.RoleAssistant {
					typ = "output_text"
				}
				content = append(content, respContent{Type: typ, Text: v.Text})
			case llm.ImagePart:
				url := v.URL
				if len(v.Data) > 0 {
					mime := v.MIME
					if mime == "" {
						mime = "image/jpeg"
					}
					url = openaicompat.DataURL(mime, v.Data)
				}
				content = append(content, respContent{Type: "input_image", ImageURL: url, Detail: orDefault(v.Detail, "auto")})
			case llm.FilePart:
				if strings.HasPrefix(v.MIME, "image/") {
					url := v.URL
					if len(v.Data) > 0 {
						url = openaicompat.DataURL(v.MIME, v.Data)
					}
					content = append(content, respContent{Type: "input_image", ImageURL: url, Detail: "auto"})
					continue
				}
				if len(v.Data) == 0 {
					return nil, fmt.Errorf("openai responses: file %q must be inlined", v.Name)
				}
				content = append(content, respContent{Type: "input_file", Filename: v.Name, FileData: openaicompat.DataURL(v.MIME, v.Data)})
			case llm.AudioPart:
				return nil, errors.New("openai responses: audio input is not supported")
			}
		}
		if len(content) > 0 {
			role := m.Role
			if role == llm.RoleTool {
				role = llm.RoleUser
			}
			items = append(items, respItem{Type: "message", Role: string(role), Content: content})
		}
		items = append(items, calls...)
	}
	return items, nil
}

func orDefault(s, d string) string {
	if s == "" {
		return d
	}
	return s
}

func responsesTools(tools llm.Tools) []any {
	custom := tools.Custom[ResponsesAdapterID]
	if len(tools.Functions) == 0 && len(custom) == 0 {
		return nil
	}
	out := make([]any, 0, len(tools.Functions)+len(custom))
	for _, f := range tools.Functions {
		t := map[string]any{"type": "function", "name": f.Name, "parameters": f.JSONSchema(), "strict": f.Strict}
		if f.Description != "" {
			t["description"] = f.Description
		}
		out = append(out, t)
	}
	for _, raw := range custom {
		out = append(out, raw)
	}
	return out
}

func (p *ResponsesProvider) BuildRequestBody(snapshot llm.Snapshot, settings llm.Settings) (map[string]any, error) {
	input, err := responsesInput(snapshot)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"model":  settings.Model,
		"input":  input,
		"stream": true,
		"store":  false,
	}
	if settings.SystemInstruction != "" {
		body["instructions"] = settings.SystemInstruction
	}
	if settings.Temperature != nil {
		body["temperature"] = *settings.Temperature
	}
	if settings.TopP != nil {
		body["top_p"] = *settings.TopP
	}
	if settings.MaxTokens != nil {
		body["max_output_tokens"] = *settings.MaxTokens
	}
	if format := responsesFormat(settings.ResponseFormat); format != nil {
		body["text"] = map[string]any{"format": format}
	}
	if tools := responsesTools(snapshot.Tools); tools != nil {
		body["tools"] = tools
		switch snapshot.ToolChoice.Mode {
		case llm.ToolChoiceAuto, llm.ToolChoiceNone, llm.ToolChoiceRequired:
			body["tool_choice"] = string(snapshot.ToolChoice.Mode)
		case llm.ToolChoiceFunction:
			body["tool_choice"] = map[string]any{"type": "function", "name": snapshot.ToolChoice.Name}
		}
	}
	for k, v := range settings.Extra {
		body[k] = v
	}
	return body, nil
}

func responsesFormat(f *llm.ResponseFormat) map[string]any {
	if !f.Structured() {
		return nil
	}
	if f.Type == llm.ResponseFormatJSONObject {
		return map[string]any{"type": llm.ResponseFormatJSONObject}
	}
	format := map[string]any{"type": llm.ResponseFormatJSONSchema, "name": f.SchemaName(), "schema": f.Schema, "strict": f.Strict}
	if f.Description != "" {
		format["description"] = f.Description
	}
	return format
}

type respUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

type respEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
	Item  *struct {
		Type      string `json:"type"`
		CallID    string `json:"call_id"`
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"item"`
	Response *struct {
		ID                string          `json:"id"`
		Model             string          `json:"model"`
		Status            string          `json:"status"`
		Usage             *respUsage      `json:"usage"`
		Error             json.RawMessage `json:"error"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
	} `json:"response"`
	Message string `json:"message"`
	Code    string `json:"code"`
}

func (p *ResponsesProvider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	body, err := p.BuildRequestBody(snapshot, settings)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	resp, err := httpretry.Do(ctx, p.client, p.retry, func(ctx context.Context) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		for k, v := range p.headers {
			req.Header[k] = v
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		if p.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+p.apiKey)
		}
		return req, nil
	})
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("%s: request failed: %w", p.name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return openaicompat.ReadAPIError(p.name, resp)
	}
	err = p.consume(resp.Body, openaicompat.RequestID(resp.Header), emit)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

func (p *ResponsesProvider) streamError(data []byte, requestID string) error {
	err := openaicompat.ParseStreamError(p.name, data)
	var api *openaicompat.APIError
	if errors.As(err, &api) {
		api.RequestID = requestID
	}
	return err
}

func (p *ResponsesProvider) consume(r io.Reader, requestID string, emit func(llm.Event)) error {
	sse := openaicompat.NewSSEReader(r)
	info := llm.ResponseInfo{RequestID: requestID}
	infoSent := false
	calls := 0
	var refusal strings.Builder
	for {
		_, data, err := sse.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return fmt.Errorf("%s: stream ended before response.completed", p.name)
			}
			return fmt.Errorf("%s: read stream: %w", p.name, err)
		}
		var ev respEvent
		if err := json.Unmarshal(data, &ev); err != nil {
			continue
		}
		if !infoSent && ev.Response != nil && ev.Response.ID != "" {
			info.ResponseID, info.Model = ev.Response.ID, ev.Response.Model
			infoSent = true
			emit(info)
		}
		switch ev.Type {
		case "response.refusal.delta":
			refusal.WriteString(ev.Delta)
		case "response.output_text.delta":
			emit(llm.TextDelta{Text: ev.Delta})
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			emit(llm.ThoughtDelta{Text: ev.Delta})
		case "response.output_item.done":
			if ev.Item != nil && ev.Item.Type == "function_call" {
				args := ev.Item.Arguments
				if strings.TrimSpace(args) == "" {
					args = "{}"
				}
				emit(llm.ToolCall{ID: ev.Item.CallID, Name: ev.Item.Name, Arguments: []byte(args)})
				calls++
			}
		case "response.completed", "response.incomplete":
			if !infoSent && info != (llm.ResponseInfo{}) {
				emit(info)
			}
			raw := ""
			if ev.Response != nil {
				if u := ev.Response.Usage; u != nil {
					usage := llm.Usage{PromptTokens: u.InputTokens, CompletionTokens: u.OutputTokens, TotalTokens: u.TotalTokens}
					if u.InputTokensDetails != nil {
						usage.CacheReadTokens = u.InputTokensDetails.CachedTokens
					}
					if u.OutputTokensDetails != nil {
						usage.ReasoningTokens = u.OutputTokensDetails.ReasoningTokens
					}
					emit(llm.UsageEvent{Usage: usage})
				}
				raw = ev.Response.Status
				if d := ev.Response.IncompleteDetails; d != nil && d.Reason != "" {
					raw = d.Reason
				}
			}
			if raw == openaicompat.CodeContentFilter || refusal.Len() > 0 {
				r := llm.Refusal{Category: llm.RefusalContentFilter, Provider: p.name, Reason: openaicompat.CodeContentFilter}
				stop := llm.StopReasonContentFilter
				if raw != openaicompat.CodeContentFilter {
					r = llm.Refusal{Category: llm.RefusalModel, Provider: p.name, Reason: "refusal", Message: refusal.String()}
					stop = llm.StopReasonRefusal
				}
				emit(r)
				emit(llm.Done{StopReason: stop, RawStopReason: raw})
				return &llm.RefusalError{Refusal: r}
			}
			stop := llm.StopReasonStop
			switch {
			case ev.Type == "response.incomplete":
				stop = openaicompat.NormalizeStopReason(raw)
			case calls > 0:
				stop = llm.StopReasonToolCalls
			}
			emit(llm.Done{StopReason: stop, RawStopReason: raw})
			return nil
		case "response.failed":
			if ev.Response != nil && len(ev.Response.Error) > 0 {
				return p.streamError(ev.Response.Error, requestID)
			}
			return p.streamError(data, requestID)
		case "error":
			return p.streamError(data, requestID)
		}
	}
}
