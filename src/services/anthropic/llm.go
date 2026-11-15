package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat/httpretry"
)

type RetryPolicy = httpretry.Policy

const (
	DefaultBaseURL   = "https://api.anthropic.com/v1"
	DefaultModel     = "claude-sonnet-4-6"
	DefaultMaxTokens = 4096
	APIVersion       = "2023-06-01"

	ExtraPromptCaching = "enable_prompt_caching"
)

type Thinking struct {
	Type         string
	BudgetTokens int
	Display      string
}

type LLMConfig struct {
	APIKey              string
	Model               string
	SystemPrompt        string
	Temperature         float64
	MaxTokens           int
	BaseURL             string
	EnablePromptCaching bool
	PromptCacheTTL      string
	Thinking            *Thinking
	Effort              string
	Betas               []string
	HTTPClient          *http.Client
	Retry               RetryPolicy
}

type Provider struct {
	apiKey   string
	baseURL  string
	model    string
	caching  bool
	cacheTTL string
	thinking *Thinking
	effort   string
	betas    string
	client   *http.Client
	retry    RetryPolicy
}

func NewProvider(config LLMConfig) *Provider {
	baseURL := strings.TrimRight(config.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	client := config.HTTPClient
	if client == nil {
		client = newHTTPClient()
	}
	return &Provider{
		apiKey:   config.APIKey,
		baseURL:  baseURL,
		model:    model,
		caching:  config.EnablePromptCaching,
		cacheTTL: config.PromptCacheTTL,
		thinking: config.Thinking,
		effort:   config.Effort,
		betas:    strings.Join(config.Betas, ","),
		client:   client,
		retry:    config.Retry,
	}
}

func NewLLMService(config LLMConfig, opts ...llmservice.Option) *llmservice.Service {
	p := NewProvider(config)
	settings := llm.Settings{Model: p.model, SystemInstruction: config.SystemPrompt}
	if config.Temperature > 0 {
		t := config.Temperature
		settings.Temperature = &t
	}
	maxTokens := config.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	settings.MaxTokens = &maxTokens
	return llmservice.New(p, append([]llmservice.Option{llmservice.WithName("Anthropic"), llmservice.WithSettings(settings)}, opts...)...)
}

func newHTTPClient() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	t.TLSHandshakeTimeout = 10 * time.Second
	t.ResponseHeaderTimeout = 60 * time.Second
	t.MaxIdleConnsPerHost = 16
	return &http.Client{Transport: t}
}

func (p *Provider) Name() string { return "Anthropic" }

type APIError struct {
	Status    int
	Type      string
	Message   string
	RequestID string
}

func (e *APIError) Error() string {
	suffix := ""
	if e.RequestID != "" {
		suffix = " (request id " + e.RequestID + ")"
	}
	if e.Status > 0 {
		return fmt.Sprintf("anthropic: %d %s: %s%s", e.Status, e.Type, e.Message, suffix)
	}
	return fmt.Sprintf("anthropic: %s: %s%s", e.Type, e.Message, suffix)
}

func normalizeStopReason(raw string) string {
	switch raw {
	case "":
		return ""
	case "end_turn", "stop_sequence":
		return llm.StopReasonStop
	case "max_tokens", "model_context_window_exceeded":
		return llm.StopReasonLength
	case "tool_use":
		return llm.StopReasonToolCalls
	case "refusal":
		return llm.StopReasonRefusal
	}
	return llm.StopReasonOther
}

func (p *Provider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	body, err := p.buildRequest(snapshot, settings)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	resp, err := httpretry.Do(ctx, p.client, p.retry, func(ctx context.Context) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/messages", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("x-api-key", p.apiKey)
		req.Header.Set("anthropic-version", APIVersion)
		req.Header.Set("content-type", "application/json")
		req.Header.Set("accept", "text/event-stream")
		if p.betas != "" {
			req.Header.Set("anthropic-beta", p.betas)
		}
		return req, nil
	})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return readAPIError(resp)
	}
	return parseStream(ctx, resp.Body, resp.Header.Get("request-id"), emit)
}

func readAPIError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var env struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	requestID := resp.Header.Get("request-id")
	if json.Unmarshal(raw, &env) == nil && env.Error.Message != "" {
		return &APIError{Status: resp.StatusCode, Type: env.Error.Type, Message: env.Error.Message, RequestID: requestID}
	}
	return &APIError{Status: resp.StatusCode, Type: http.StatusText(resp.StatusCode), Message: strings.TrimSpace(string(raw)), RequestID: requestID}
}

type streamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message *struct {
		ID    string `json:"id"`
		Model string `json:"model"`
		Usage usage  `json:"usage"`
	} `json:"message"`
	ContentBlock *struct {
		Type      string `json:"type"`
		ID        string `json:"id"`
		Name      string `json:"name"`
		Text      string `json:"text"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
	} `json:"content_block"`
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		StopReason  string `json:"stop_reason"`
		StopDetails *struct {
			Category string `json:"category"`
		} `json:"stop_details"`
	} `json:"delta"`
	Usage *usage `json:"usage"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type usage struct {
	InputTokens              *int `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
}

func (u usage) apply(out *llm.Usage) {
	if u.InputTokens != nil {
		out.PromptTokens = *u.InputTokens
	}
	if u.OutputTokens != nil {
		out.CompletionTokens = *u.OutputTokens
	}
	if u.CacheCreationInputTokens != nil {
		out.CacheCreationTokens = *u.CacheCreationInputTokens
	}
	if u.CacheReadInputTokens != nil {
		out.CacheReadTokens = *u.CacheReadInputTokens
	}
}

type toolBlock struct {
	id   string
	name string
	args []byte
}

func parseStream(ctx context.Context, body io.Reader, requestID string, emit func(llm.Event)) error {
	r := newSSEReader(body)
	tools := map[int]*toolBlock{}
	var u llm.Usage
	gotUsage := false
	stopReason := ""
	stopCategory := ""
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		data, err := r.next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return err
		}
		if len(data) == 0 {
			continue
		}
		var ev streamEvent
		if err := json.Unmarshal(data, &ev); err != nil {
			return fmt.Errorf("anthropic: invalid stream event: %w", err)
		}
		switch ev.Type {
		case "message_start":
			if ev.Message != nil {
				ev.Message.Usage.apply(&u)
				gotUsage = true
				if ev.Message.ID != "" || requestID != "" {
					emit(llm.ResponseInfo{ResponseID: ev.Message.ID, RequestID: requestID, Model: ev.Message.Model})
				}
			}
		case "content_block_start":
			if ev.ContentBlock == nil {
				continue
			}
			switch ev.ContentBlock.Type {
			case "tool_use":
				tools[ev.Index] = &toolBlock{id: ev.ContentBlock.ID, name: ev.ContentBlock.Name}
			case "text":
				if ev.ContentBlock.Text != "" {
					emit(llm.TextDelta{Text: ev.ContentBlock.Text})
				}
			case "thinking":
				if ev.ContentBlock.Thinking != "" || ev.ContentBlock.Signature != "" {
					emit(llm.ThoughtDelta{Text: ev.ContentBlock.Thinking, Signature: ev.ContentBlock.Signature})
				}
			}
		case "content_block_delta":
			if ev.Delta == nil {
				continue
			}
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text != "" {
					emit(llm.TextDelta{Text: ev.Delta.Text})
				}
			case "thinking_delta":
				if ev.Delta.Thinking != "" {
					emit(llm.ThoughtDelta{Text: ev.Delta.Thinking})
				}
			case "signature_delta":
				emit(llm.ThoughtDelta{Signature: ev.Delta.Signature})
			case "input_json_delta":
				if tb := tools[ev.Index]; tb != nil {
					tb.args = append(tb.args, ev.Delta.PartialJSON...)
				}
			}
		case "content_block_stop":
			if tb := tools[ev.Index]; tb != nil {
				delete(tools, ev.Index)
				args := tb.args
				if len(bytes.TrimSpace(args)) == 0 {
					args = []byte("{}")
				}
				emit(llm.ToolCall{ID: tb.id, Name: tb.name, Arguments: args})
			}
		case "message_delta":
			if ev.Delta != nil && ev.Delta.StopReason != "" {
				stopReason = ev.Delta.StopReason
				if ev.Delta.StopDetails != nil {
					stopCategory = ev.Delta.StopDetails.Category
				}
			}
			if ev.Usage != nil {
				ev.Usage.apply(&u)
				gotUsage = true
			}
		case "message_stop":
			if gotUsage {
				u.TotalTokens = u.PromptTokens + u.CompletionTokens + u.CacheReadTokens + u.CacheCreationTokens
				emit(llm.UsageEvent{Usage: u})
			}
			if stopReason == "refusal" {
				r := llm.Refusal{Category: llm.RefusalModel, Provider: "Anthropic", Reason: stopReason, Detail: stopCategory}
				emit(r)
				emit(llm.Done{StopReason: llm.StopReasonRefusal, RawStopReason: stopReason})
				return &llm.RefusalError{Refusal: r, Err: &APIError{Type: "refusal", Message: "the model declined to respond", RequestID: requestID}}
			}
			emit(llm.Done{StopReason: normalizeStopReason(stopReason), RawStopReason: stopReason})
			return nil
		case "error":
			if ev.Error != nil {
				return &APIError{Type: ev.Error.Type, Message: ev.Error.Message, RequestID: requestID}
			}
			return &APIError{Type: "error", Message: string(data), RequestID: requestID}
		}
	}
	return errors.New("anthropic: stream ended before message_stop")
}

type sseReader struct {
	r    *bufio.Reader
	data []byte
	line []byte
}

func newSSEReader(r io.Reader) *sseReader {
	return &sseReader{r: bufio.NewReaderSize(r, 32<<10)}
}

func (s *sseReader) readLine() ([]byte, error) {
	s.line = s.line[:0]
	for {
		chunk, err := s.r.ReadSlice('\n')
		s.line = append(s.line, chunk...)
		if err == nil {
			break
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		if errors.Is(err, io.EOF) && len(s.line) > 0 {
			break
		}
		return nil, err
	}
	return bytes.TrimRight(s.line, "\r\n"), nil
}

func (s *sseReader) next() ([]byte, error) {
	s.data = s.data[:0]
	has := false
	for {
		line, err := s.readLine()
		if err != nil {
			if errors.Is(err, io.EOF) && has {
				return s.data, nil
			}
			return nil, err
		}
		if len(line) == 0 {
			if has {
				return s.data, nil
			}
			continue
		}
		if line[0] == ':' {
			continue
		}
		field, value, _ := bytes.Cut(line, []byte(":"))
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
		if string(field) == "data" {
			if has {
				s.data = append(s.data, '\n')
			}
			s.data = append(s.data, value...)
			has = true
		}
	}
}

func (p *Provider) buildRequest(snapshot llm.Snapshot, settings llm.Settings) (map[string]any, error) {
	model := settings.Model
	if model == "" {
		model = p.model
	}
	maxTokens := DefaultMaxTokens
	if settings.MaxTokens != nil && *settings.MaxTokens > 0 {
		maxTokens = *settings.MaxTokens
	}
	caching := p.caching
	if v, ok := settings.Extra[ExtraPromptCaching].(bool); ok {
		caching = v
	}

	system, messages := convertMessages(snapshot.Messages, settings.SystemInstruction)
	body := map[string]any{
		"model":      model,
		"max_tokens": maxTokens,
		"stream":     true,
		"messages":   messages,
	}
	if len(system) > 0 {
		if caching {
			cc := map[string]any{"type": "ephemeral"}
			if p.cacheTTL != "" {
				cc["ttl"] = p.cacheTTL
			}
			system[len(system)-1]["cache_control"] = cc
		}
		body["system"] = system
	}
	if caching {
		addCacheMarkers(messages)
	}
	if settings.Temperature != nil {
		body["temperature"] = *settings.Temperature
	}
	if settings.TopP != nil {
		body["top_p"] = *settings.TopP
	}
	if p.thinking != nil && p.thinking.Type != "" {
		th := map[string]any{"type": p.thinking.Type}
		if p.thinking.BudgetTokens > 0 {
			th["budget_tokens"] = p.thinking.BudgetTokens
		}
		if p.thinking.Display != "" {
			th["display"] = p.thinking.Display
		}
		body["thinking"] = th
	}
	output := map[string]any{}
	if p.effort != "" {
		output["effort"] = p.effort
	}
	if f := settings.ResponseFormat; f.Structured() {
		if f.Type != llm.ResponseFormatJSONSchema || len(f.Schema) == 0 {
			return nil, fmt.Errorf("anthropic: %w without a JSON schema", llm.ErrResponseFormatUnsupported)
		}
		output["format"] = map[string]any{"type": "json_schema", "schema": f.Schema}
	}
	if len(output) > 0 {
		body["output_config"] = output
	}
	if tools := convertTools(snapshot.Tools); len(tools) > 0 {
		if caching {
			tools[len(tools)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
		}
		body["tools"] = tools
		if tc := convertToolChoice(snapshot.ToolChoice); tc != nil {
			body["tool_choice"] = tc
		}
	}
	for k, v := range settings.Extra {
		if k == ExtraPromptCaching {
			continue
		}
		body[k] = v
	}
	return body, nil
}

func convertTools(tools llm.Tools) []map[string]any {
	out := make([]map[string]any, 0, len(tools.Functions))
	for _, f := range tools.Functions {
		t := map[string]any{
			"name":         f.Name,
			"description":  f.Description,
			"input_schema": f.JSONSchema(),
		}
		if f.Strict {
			t["strict"] = true
		}
		out = append(out, t)
	}
	for _, raw := range tools.Custom["anthropic"] {
		var t map[string]any
		if json.Unmarshal(raw, &t) == nil {
			out = append(out, t)
		}
	}
	return out
}

func convertToolChoice(tc llm.ToolChoice) map[string]any {
	switch tc.Mode {
	case llm.ToolChoiceAuto:
		return map[string]any{"type": "auto"}
	case llm.ToolChoiceNone:
		return map[string]any{"type": "none"}
	case llm.ToolChoiceRequired:
		return map[string]any{"type": "any"}
	case llm.ToolChoiceFunction:
		return map[string]any{"type": "tool", "name": tc.Name}
	}
	return nil
}

type message struct {
	Role    string           `json:"role"`
	Content []map[string]any `json:"content"`
}

func convertMessages(msgs []llm.Message, instruction string) ([]map[string]any, []*message) {
	var system []map[string]any
	if strings.TrimSpace(instruction) != "" {
		system = append(system, map[string]any{"type": "text", "text": instruction})
	}
	i := 0
	for ; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role != llm.RoleSystem && m.Role != llm.RoleDeveloper {
			break
		}
		if text := m.Text(); strings.TrimSpace(text) != "" {
			system = append(system, map[string]any{"type": "text", "text": text})
		}
	}

	var out []*message
	for _, m := range msgs[i:] {
		role := "user"
		if m.Role == llm.RoleAssistant {
			role = "assistant"
		}
		blocks := convertParts(m)
		if len(blocks) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			continue
		}
		out = append(out, &message{Role: role, Content: blocks})
	}
	for _, m := range out {
		if m.Role == "user" {
			hoistToolResults(m.Content)
		}
	}
	return system, out
}

func hoistToolResults(blocks []map[string]any) {
	var results, rest []map[string]any
	for _, b := range blocks {
		if b["type"] == "tool_result" {
			results = append(results, b)
		} else {
			rest = append(rest, b)
		}
	}
	if len(results) == 0 || len(rest) == 0 {
		return
	}
	n := copy(blocks, results)
	copy(blocks[n:], rest)
}

func convertParts(m llm.Message) []map[string]any {
	blocks := make([]map[string]any, 0, len(m.Parts))
	for _, part := range m.Parts {
		switch p := part.(type) {
		case llm.TextPart:
			if p.Text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": p.Text})
			}
		case llm.ImagePart:
			if src := mediaSource(p.URL, p.Data, p.MIME); src != nil {
				blocks = append(blocks, map[string]any{"type": "image", "source": src})
			}
		case llm.FilePart:
			mime := p.MIME
			kind := "document"
			if strings.HasPrefix(mime, "image/") {
				kind = "image"
			} else if mime != "" && mime != "application/pdf" && !strings.HasPrefix(mime, "text/") {
				continue
			}
			if strings.HasPrefix(mime, "text/") && len(p.Data) > 0 {
				blocks = append(blocks, map[string]any{"type": "document", "source": map[string]any{"type": "text", "media_type": "text/plain", "data": string(p.Data)}})
				continue
			}
			if src := mediaSource(p.URL, p.Data, mime); src != nil {
				blocks = append(blocks, map[string]any{"type": kind, "source": src})
			}
		case llm.ToolCallPart:
			var input any = map[string]any{}
			if len(bytes.TrimSpace(p.Arguments)) > 0 {
				var v any
				if json.Unmarshal(p.Arguments, &v) == nil && v != nil {
					input = v
				}
			}
			blocks = append(blocks, map[string]any{"type": "tool_use", "id": p.ID, "name": p.Name, "input": input})
		case llm.ToolResultPart:
			content := p.Content
			if content == "" {
				content = "(empty)"
			}
			b := map[string]any{"type": "tool_result", "tool_use_id": p.ToolCallID, "content": content}
			if p.IsError {
				b["is_error"] = true
			}
			blocks = append(blocks, b)
		case llm.ThoughtPart:
			if m.Role != llm.RoleAssistant {
				continue
			}
			if p.Redacted {
				data := p.Signature
				if data == "" {
					data = p.Text
				}
				if data != "" {
					blocks = append(blocks, map[string]any{"type": "redacted_thinking", "data": data})
				}
				continue
			}
			if p.Signature != "" {
				blocks = append(blocks, map[string]any{"type": "thinking", "thinking": p.Text, "signature": p.Signature})
			}
		case llm.ProviderPart:
			if p.Provider != "anthropic" {
				continue
			}
			var b map[string]any
			if json.Unmarshal(p.Raw, &b) == nil && b != nil {
				blocks = append(blocks, b)
			}
		}
	}
	return blocks
}

func mediaSource(url string, data []byte, mime string) map[string]any {
	if len(data) > 0 {
		if mime == "" {
			mime = http.DetectContentType(data)
		}
		return map[string]any{"type": "base64", "media_type": mime, "data": base64.StdEncoding.EncodeToString(data)}
	}
	if strings.HasPrefix(url, "data:") {
		meta, payload, ok := strings.Cut(strings.TrimPrefix(url, "data:"), ",")
		if !ok {
			return nil
		}
		return map[string]any{"type": "base64", "media_type": strings.TrimSuffix(meta, ";base64"), "data": payload}
	}
	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		return map[string]any{"type": "url", "url": url}
	}
	return nil
}

func addCacheMarkers(messages []*message) {
	marked := 0
	for i := len(messages) - 1; i >= 0 && marked < 2; i-- {
		m := messages[i]
		if m.Role != "user" {
			continue
		}
		for j := len(m.Content) - 1; j >= 0; j-- {
			t, _ := m.Content[j]["type"].(string)
			if t == "thinking" || t == "redacted_thinking" {
				continue
			}
			m.Content[j]["cache_control"] = map[string]any{"type": "ephemeral"}
			break
		}
		marked++
	}
}
