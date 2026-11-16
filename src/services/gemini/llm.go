package gemini

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
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat/httpretry"
)

type RetryPolicy = httpretry.Policy

const (
	DefaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"
	DefaultModel   = "gemini-3.6-flash"

	generatedIDPrefix = "gemini-call-"
	signatureCacheCap = 1024
)

var placeholderSignature = base64.StdEncoding.EncodeToString([]byte("skip_thought_signature_validator"))

type Thinking struct {
	Budget          *int
	Level           string
	IncludeThoughts bool
}

type LLMConfig struct {
	APIKey         string
	Model          string
	SystemPrompt   string
	Temperature    float64
	MaxTokens      int
	BaseURL        string
	Thinking       *Thinking
	SafetySettings []map[string]any
	HTTPClient     *http.Client
	Retry          RetryPolicy
}

type ProviderConfig struct {
	Name           string
	Model          string
	URL            func(model string) string
	Authorize      func(ctx context.Context, h http.Header) error
	Thinking       *Thinking
	SafetySettings []map[string]any
	HTTPClient     *http.Client
	Retry          RetryPolicy
}

type Provider struct {
	cfg ProviderConfig

	sigMu    sync.Mutex
	sigs     map[string]string
	sigOrder []string
}

func NewProvider(cfg ProviderConfig) *Provider {
	if cfg.Name == "" {
		cfg.Name = "Gemini"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = NewHTTPClient()
	}
	return &Provider{cfg: cfg, sigs: make(map[string]string)}
}

func NewLLMService(config LLMConfig, opts ...llmservice.Option) *llmservice.Service {
	base := strings.TrimRight(config.BaseURL, "/")
	if base == "" {
		base = DefaultBaseURL
	}
	key := config.APIKey
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	p := NewProvider(ProviderConfig{
		Name:  "Gemini",
		Model: model,
		URL: func(model string) string {
			return base + "/" + modelPath(model) + ":streamGenerateContent?alt=sse"
		},
		Authorize: func(_ context.Context, h http.Header) error {
			h.Set("x-goog-api-key", key)
			return nil
		},
		Thinking:       config.Thinking,
		SafetySettings: config.SafetySettings,
		HTTPClient:     config.HTTPClient,
		Retry:          config.Retry,
	})
	return llmservice.New(p, append([]llmservice.Option{llmservice.WithName("Gemini"), llmservice.WithSettings(SettingsFrom(model, config.SystemPrompt, config.Temperature, config.MaxTokens))}, opts...)...)
}

func SettingsFrom(model, system string, temperature float64, maxTokens int) llm.Settings {
	s := llm.Settings{Model: model, SystemInstruction: system}
	if temperature > 0 {
		s.Temperature = &temperature
	}
	if maxTokens > 0 {
		s.MaxTokens = &maxTokens
	}
	return s
}

func NewHTTPClient() *http.Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	t.TLSHandshakeTimeout = 10 * time.Second
	t.ResponseHeaderTimeout = 60 * time.Second
	t.MaxIdleConnsPerHost = 16
	return &http.Client{Transport: t}
}

func (p *Provider) Name() string { return p.cfg.Name }

type APIError struct {
	Status  int
	Code    string
	Message string
}

func (e *APIError) Error() string {
	if e.Status > 0 {
		return fmt.Sprintf("gemini: %d %s: %s", e.Status, e.Code, e.Message)
	}
	return fmt.Sprintf("gemini: %s: %s", e.Code, e.Message)
}

func (p *Provider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	model := settings.Model
	if model == "" {
		model = p.cfg.Model
	}
	if model == "" {
		return errors.New("gemini: model is required")
	}
	payload, err := json.Marshal(p.BuildRequest(snapshot, settings))
	if err != nil {
		return err
	}
	target := p.cfg.URL(model)
	resp, err := httpretry.Do(ctx, p.cfg.HTTPClient, p.cfg.Retry, func(ctx context.Context) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		if p.cfg.Authorize != nil {
			if err := p.cfg.Authorize(ctx, req.Header); err != nil {
				return nil, err
			}
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
	return p.parseStream(ctx, resp.Body, emit)
}

type apiErrorBody struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

func readAPIError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var env struct {
		Error apiErrorBody `json:"error"`
	}
	if json.Unmarshal(raw, &env) == nil && env.Error.Message != "" {
		return &APIError{Status: resp.StatusCode, Code: env.Error.Status, Message: env.Error.Message}
	}
	var arr []struct {
		Error apiErrorBody `json:"error"`
	}
	if json.Unmarshal(raw, &arr) == nil && len(arr) > 0 && arr[0].Error.Message != "" {
		return &APIError{Status: resp.StatusCode, Code: arr[0].Error.Status, Message: arr[0].Error.Message}
	}
	return &APIError{Status: resp.StatusCode, Code: http.StatusText(resp.StatusCode), Message: strings.TrimSpace(string(raw))}
}

type wirePart struct {
	Text             string            `json:"text,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
	InlineData       *wireBlob         `json:"inlineData,omitempty"`
	FileData         *wireFile         `json:"fileData,omitempty"`
	FunctionCall     *wireFunctionCall `json:"functionCall,omitempty"`
	FunctionResponse *wireFunctionResp `json:"functionResponse,omitempty"`
}

type wireBlob struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type wireFile struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type wireFunctionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type wireFunctionResp struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type Content struct {
	Role  string     `json:"role,omitempty"`
	Parts []wirePart `json:"parts"`
}

type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	ResponseTokenCount      int `json:"responseTokenCount"`
	TotalTokenCount         int `json:"totalTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
}

func (u *usageMetadata) toUsage() llm.Usage {
	completion := u.CandidatesTokenCount
	if completion == 0 {
		completion = u.ResponseTokenCount
	}
	return llm.Usage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: completion,
		TotalTokens:      u.TotalTokenCount,
		CacheReadTokens:  u.CachedContentTokenCount,
		ReasoningTokens:  u.ThoughtsTokenCount,
	}
}

type safetyRating struct {
	Category string `json:"category"`
	Blocked  bool   `json:"blocked"`
}

func blockedCategories(ratings []safetyRating) string {
	var names []string
	for _, r := range ratings {
		if r.Blocked && r.Category != "" {
			names = append(names, r.Category)
		}
	}
	return strings.Join(names, ",")
}

type streamChunk struct {
	ResponseID   string `json:"responseId"`
	ModelVersion string `json:"modelVersion"`
	Candidates   []struct {
		Content       Content        `json:"content"`
		FinishReason  string         `json:"finishReason"`
		FinishMessage string         `json:"finishMessage"`
		SafetyRatings []safetyRating `json:"safetyRatings"`
	} `json:"candidates"`
	PromptFeedback *struct {
		BlockReason        string `json:"blockReason"`
		BlockReasonMessage string `json:"blockReasonMessage"`
	} `json:"promptFeedback"`
	UsageMetadata *usageMetadata `json:"usageMetadata"`
	Error         *apiErrorBody  `json:"error"`
}

var refusalFinish = map[string]llm.RefusalCategory{
	"SAFETY":             llm.RefusalContentFilter,
	"BLOCKLIST":          llm.RefusalContentFilter,
	"PROHIBITED_CONTENT": llm.RefusalContentFilter,
	"SPII":               llm.RefusalContentFilter,
	"IMAGE_SAFETY":       llm.RefusalContentFilter,
	"RECITATION":         llm.RefusalRecitation,
}

var blockedFinish = map[string]bool{
	"SAFETY": true, "RECITATION": true, "BLOCKLIST": true, "PROHIBITED_CONTENT": true,
	"SPII": true, "MALFORMED_FUNCTION_CALL": true, "IMAGE_SAFETY": true, "LANGUAGE": true,
	"UNEXPECTED_TOOL_CALL": true,
}

func (p *Provider) parseStream(ctx context.Context, body io.Reader, emit func(llm.Event)) error {
	r := newSSEReader(body)
	var usage *usageMetadata
	finish := ""
	finishMessage := ""
	blocked := ""
	calls := 0
	infoSent := false
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
		var chunk streamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return fmt.Errorf("gemini: invalid stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return &APIError{Status: chunk.Error.Code, Code: chunk.Error.Status, Message: chunk.Error.Message}
		}
		if !infoSent && chunk.ResponseID != "" {
			infoSent = true
			emit(llm.ResponseInfo{ResponseID: chunk.ResponseID, Model: chunk.ModelVersion})
		}
		if chunk.PromptFeedback != nil && chunk.PromptFeedback.BlockReason != "" {
			reason := chunk.PromptFeedback.BlockReason
			msg := chunk.PromptFeedback.BlockReasonMessage
			r := llm.Refusal{Category: llm.RefusalPromptBlocked, Provider: p.cfg.Name, Reason: reason, Message: msg}
			if msg == "" {
				msg = "prompt blocked"
			}
			if chunk.UsageMetadata != nil {
				emit(llm.UsageEvent{Usage: chunk.UsageMetadata.toUsage()})
			}
			emit(r)
			emit(llm.Done{StopReason: llm.StopReasonContentFilter, RawStopReason: reason})
			return &llm.RefusalError{Refusal: r, Err: &APIError{Code: "BLOCKED_" + reason, Message: msg}}
		}
		if chunk.UsageMetadata != nil {
			usage = chunk.UsageMetadata
		}
		if len(chunk.Candidates) == 0 {
			continue
		}
		cand := chunk.Candidates[0]
		for _, part := range cand.Content.Parts {
			switch {
			case part.FunctionCall != nil:
				id := part.FunctionCall.ID
				if id == "" {
					id = generatedIDPrefix + uuid.NewString()
				}
				if part.ThoughtSignature != "" {
					p.rememberSignature(id, part.ThoughtSignature)
				}
				args := []byte(part.FunctionCall.Args)
				if len(bytes.TrimSpace(args)) == 0 || string(args) == "null" {
					args = []byte("{}")
				}
				calls++
				emit(llm.ToolCall{ID: id, Name: part.FunctionCall.Name, Arguments: args, Signature: part.ThoughtSignature})
			case part.Thought:
				if part.Text != "" || part.ThoughtSignature != "" {
					emit(llm.ThoughtDelta{Text: part.Text, Signature: part.ThoughtSignature})
				}
			case part.Text != "":
				emit(llm.TextDelta{Text: part.Text})
			}
		}
		if cand.FinishReason != "" {
			finish = cand.FinishReason
			finishMessage = cand.FinishMessage
			blocked = blockedCategories(cand.SafetyRatings)
		}
	}
	if usage != nil {
		emit(llm.UsageEvent{Usage: usage.toUsage()})
	}
	if blockedFinish[finish] {
		apiErr := &APIError{Code: finish, Message: "response blocked (finishReason " + finish + ")"}
		category, refused := refusalFinish[finish]
		if !refused {
			return apiErr
		}
		r := llm.Refusal{Category: category, Provider: p.cfg.Name, Reason: finish, Detail: blocked, Message: finishMessage}
		emit(r)
		emit(llm.Done{StopReason: llm.StopReasonContentFilter, RawStopReason: finish})
		return &llm.RefusalError{Refusal: r, Err: apiErr}
	}
	emit(llm.Done{StopReason: normalizeFinish(finish, calls), RawStopReason: finish})
	return nil
}

func normalizeFinish(finish string, calls int) string {
	switch finish {
	case "":
		return ""
	case "STOP":
		if calls > 0 {
			return llm.StopReasonToolCalls
		}
		return llm.StopReasonStop
	case "MAX_TOKENS":
		return llm.StopReasonLength
	}
	return llm.StopReasonOther
}

func (p *Provider) rememberSignature(id, sig string) {
	p.sigMu.Lock()
	defer p.sigMu.Unlock()
	if _, ok := p.sigs[id]; !ok {
		p.sigOrder = append(p.sigOrder, id)
		if len(p.sigOrder) > signatureCacheCap {
			delete(p.sigs, p.sigOrder[0])
			p.sigOrder = p.sigOrder[1:]
		}
	}
	p.sigs[id] = sig
}

func (p *Provider) signature(id string) string {
	p.sigMu.Lock()
	defer p.sigMu.Unlock()
	return p.sigs[id]
}

func (p *Provider) BuildRequest(snapshot llm.Snapshot, settings llm.Settings) map[string]any {
	system, contents := p.ConvertMessages(snapshot.Messages, settings.SystemInstruction)
	body := map[string]any{"contents": contents}
	if system != "" {
		body["systemInstruction"] = map[string]any{"parts": []map[string]any{{"text": system}}}
	}
	gen := map[string]any{}
	if settings.Temperature != nil {
		gen["temperature"] = *settings.Temperature
	}
	if settings.TopP != nil {
		gen["topP"] = *settings.TopP
	}
	if settings.MaxTokens != nil && *settings.MaxTokens > 0 {
		gen["maxOutputTokens"] = *settings.MaxTokens
	}
	if settings.Seed != nil {
		gen["seed"] = *settings.Seed
	}
	if th := p.cfg.Thinking; th != nil {
		tc := map[string]any{}
		if th.Budget != nil {
			tc["thinkingBudget"] = *th.Budget
		}
		if th.Level != "" {
			tc["thinkingLevel"] = th.Level
		}
		if th.IncludeThoughts {
			tc["includeThoughts"] = true
		}
		gen["thinkingConfig"] = tc
	}
	if f := settings.ResponseFormat; f.Structured() {
		gen["responseMimeType"] = "application/json"
		if f.Type == llm.ResponseFormatJSONSchema && len(f.Schema) > 0 {
			gen["responseJsonSchema"] = f.Schema
		}
	}
	if len(gen) > 0 {
		body["generationConfig"] = gen
	}
	if len(p.cfg.SafetySettings) > 0 {
		body["safetySettings"] = p.cfg.SafetySettings
	}
	if tools := ConvertTools(snapshot.Tools); len(tools) > 0 {
		body["tools"] = tools
		if tc := convertToolChoice(snapshot.ToolChoice); tc != nil {
			body["toolConfig"] = tc
		}
	}
	for k, v := range settings.Extra {
		body[k] = v
	}
	return body
}

func ConvertTools(tools llm.Tools) []map[string]any {
	var out []map[string]any
	if len(tools.Functions) > 0 {
		decls := make([]map[string]any, 0, len(tools.Functions))
		for _, f := range tools.Functions {
			d := map[string]any{"name": f.Name, "description": f.Description}
			if len(f.Properties) > 0 {
				d["parameters"] = f.JSONSchema()
			}
			decls = append(decls, d)
		}
		out = append(out, map[string]any{"functionDeclarations": decls})
	}
	for _, raw := range tools.Custom["gemini"] {
		var t map[string]any
		if json.Unmarshal(raw, &t) == nil {
			out = append(out, t)
		}
	}
	return out
}

func convertToolChoice(tc llm.ToolChoice) map[string]any {
	var cfg map[string]any
	switch tc.Mode {
	case llm.ToolChoiceAuto:
		cfg = map[string]any{"mode": "AUTO"}
	case llm.ToolChoiceNone:
		cfg = map[string]any{"mode": "NONE"}
	case llm.ToolChoiceRequired:
		cfg = map[string]any{"mode": "ANY"}
	case llm.ToolChoiceFunction:
		cfg = map[string]any{"mode": "ANY", "allowedFunctionNames": []string{tc.Name}}
	default:
		return nil
	}
	return map[string]any{"functionCallingConfig": cfg}
}

func (p *Provider) ConvertMessages(msgs []llm.Message, instruction string) (string, []Content) {
	var sys []string
	if s := strings.TrimSpace(instruction); s != "" {
		sys = append(sys, instruction)
	}
	i := 0
	for ; i < len(msgs); i++ {
		if msgs[i].Role != llm.RoleSystem && msgs[i].Role != llm.RoleDeveloper {
			break
		}
		if t := msgs[i].Text(); strings.TrimSpace(t) != "" {
			sys = append(sys, t)
		}
	}
	system := strings.Join(sys, "\n\n")

	names := map[string]string{}
	for _, m := range msgs {
		for _, c := range m.ToolCalls() {
			names[c.ID] = c.Name
		}
	}

	var contents []Content
	for _, m := range msgs[i:] {
		role := "user"
		if m.Role == llm.RoleAssistant {
			role = "model"
		}
		parts := p.convertParts(m, role, names)
		if len(parts) == 0 {
			continue
		}
		if n := len(contents); n > 0 && contents[n-1].Role == role {
			contents[n-1].Parts = append(contents[n-1].Parts, parts...)
			continue
		}
		contents = append(contents, Content{Role: role, Parts: parts})
	}
	for ci := range contents {
		c := &contents[ci]
		if c.Role != "model" {
			continue
		}
		signed := false
		for _, part := range c.Parts {
			if part.ThoughtSignature != "" {
				signed = true
				break
			}
		}
		if signed {
			continue
		}
		for pi := range c.Parts {
			if c.Parts[pi].FunctionCall != nil {
				c.Parts[pi].ThoughtSignature = placeholderSignature
			}
		}
	}
	if len(contents) == 0 && system != "" {
		contents = append(contents, Content{Role: "user", Parts: []wirePart{{Text: system}}})
	}
	return system, contents
}

func (p *Provider) convertParts(m llm.Message, role string, names map[string]string) []wirePart {
	parts := make([]wirePart, 0, len(m.Parts))
	pendingSig := ""
	for _, part := range m.Parts {
		var wp wirePart
		switch v := part.(type) {
		case llm.TextPart:
			if v.Text == "" {
				continue
			}
			wp.Text = v.Text
		case llm.ImagePart:
			b, f := media(v.URL, v.Data, v.MIME)
			if b == nil && f == nil {
				continue
			}
			wp.InlineData, wp.FileData = b, f
		case llm.AudioPart:
			if len(v.Data) == 0 {
				continue
			}
			mime := v.MIME
			if mime == "" {
				mime = fmt.Sprintf("audio/pcm;rate=%d", v.SampleRate)
			}
			wp.InlineData = &wireBlob{MimeType: mime, Data: base64.StdEncoding.EncodeToString(v.Data)}
		case llm.FilePart:
			b, f := media(v.URL, v.Data, v.MIME)
			if b == nil && f == nil {
				continue
			}
			wp.InlineData, wp.FileData = b, f
		case llm.ThoughtPart:
			if role == "model" && v.Signature != "" && !v.Redacted {
				pendingSig = v.Signature
			}
			continue
		case llm.ToolCallPart:
			id := v.ID
			if strings.HasPrefix(id, generatedIDPrefix) {
				id = ""
			}
			args := json.RawMessage(v.Arguments)
			if len(bytes.TrimSpace(args)) == 0 {
				args = json.RawMessage("{}")
			}
			wp.FunctionCall = &wireFunctionCall{ID: id, Name: v.Name, Args: args}
			wp.ThoughtSignature = v.Signature
			if wp.ThoughtSignature == "" {
				wp.ThoughtSignature = p.signature(v.ID)
			}
		case llm.ToolResultPart:
			name := v.Name
			if name == "" {
				name = names[v.ToolCallID]
			}
			id := v.ToolCallID
			if strings.HasPrefix(id, generatedIDPrefix) {
				id = ""
			}
			wp.FunctionResponse = &wireFunctionResp{ID: id, Name: name, Response: FunctionResponse(v.Content, v.IsError)}
		case llm.ProviderPart:
			if v.Provider != "gemini" {
				continue
			}
			if json.Unmarshal(v.Raw, &wp) != nil {
				continue
			}
		default:
			continue
		}
		if pendingSig != "" && wp.ThoughtSignature == "" {
			wp.ThoughtSignature = pendingSig
			pendingSig = ""
		}
		parts = append(parts, wp)
	}
	if pendingSig != "" && len(parts) > 0 && parts[len(parts)-1].ThoughtSignature == "" {
		parts[len(parts)-1].ThoughtSignature = pendingSig
	}
	return parts
}

func FunctionResponse(content string, isError bool) map[string]any {
	var decoded any
	if err := json.Unmarshal([]byte(content), &decoded); err != nil {
		decoded = content
	}
	if isError {
		return map[string]any{"error": decoded}
	}
	if m, ok := decoded.(map[string]any); ok {
		return m
	}
	return map[string]any{"value": decoded}
}

func media(rawURL string, data []byte, mime string) (*wireBlob, *wireFile) {
	if len(data) > 0 {
		if mime == "" {
			mime = http.DetectContentType(data)
		}
		return &wireBlob{MimeType: mime, Data: base64.StdEncoding.EncodeToString(data)}, nil
	}
	if strings.HasPrefix(rawURL, "data:") {
		meta, payload, ok := strings.Cut(strings.TrimPrefix(rawURL, "data:"), ",")
		if !ok {
			return nil, nil
		}
		return &wireBlob{MimeType: strings.TrimSuffix(meta, ";base64"), Data: payload}, nil
	}
	if rawURL == "" {
		return nil, nil
	}
	if _, err := url.Parse(rawURL); err != nil {
		return nil, nil
	}
	return nil, &wireFile{MimeType: mime, FileURI: rawURL}
}

func modelPath(model string) string {
	if strings.HasPrefix(model, "models/") {
		return model
	}
	return "models/" + model
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
