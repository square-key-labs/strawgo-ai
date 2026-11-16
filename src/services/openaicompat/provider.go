package openaicompat

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/openaicompat/httpretry"
)

type RetryPolicy = httpretry.Policy

var sharedClient = &http.Client{Transport: &http.Transport{
	Proxy:                 http.ProxyFromEnvironment,
	DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
	ForceAttemptHTTP2:     true,
	MaxIdleConns:          256,
	MaxIdleConnsPerHost:   32,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   10 * time.Second,
	ResponseHeaderTimeout: 90 * time.Second,
	ExpectContinueTimeout: time.Second,
}}

func DefaultHTTPClient() *http.Client { return sharedClient }

type Config struct {
	Name       string
	BaseURL    string
	APIKey     string
	Headers    map[string]string
	HTTPClient *http.Client
	Retry      RetryPolicy
}

type Option func(*options)

type options struct {
	path              string
	query             url.Values
	urlFunc           func(llm.Settings) string
	authHeader        string
	authPrefix        string
	noStreamOptions   bool
	noDeveloperRole   bool
	noTools           bool
	reasoningContent  bool
	skipAnsweredCalls bool
	maxTokensField    string
	seedField         string
	transformMessages func([]Message, llm.Settings) []Message
	prepareBody       func(map[string]any, llm.Settings)
	adjustUsage       func(*llm.Usage)
	retry             *RetryPolicy
	thoughtOpen       string
	thoughtClose      string
}

func WithPath(path string) Option { return func(o *options) { o.path = path } }

func WithQuery(key, value string) Option {
	return func(o *options) {
		if o.query == nil {
			o.query = url.Values{}
		}
		o.query.Set(key, value)
	}
}

func WithURLFunc(fn func(llm.Settings) string) Option {
	return func(o *options) { o.urlFunc = fn }
}

func WithAuthHeader(header, prefix string) Option {
	return func(o *options) { o.authHeader, o.authPrefix = header, prefix }
}

func WithoutStreamOptions() Option { return func(o *options) { o.noStreamOptions = true } }

func WithoutDeveloperRole() Option { return func(o *options) { o.noDeveloperRole = true } }

func WithoutTools() Option { return func(o *options) { o.noTools = true } }

func WithReasoningContent() Option { return func(o *options) { o.reasoningContent = true } }

func WithSkipAnsweredToolCalls() Option { return func(o *options) { o.skipAnsweredCalls = true } }

func WithMaxTokensField(field string) Option { return func(o *options) { o.maxTokensField = field } }

func WithSeedField(field string) Option { return func(o *options) { o.seedField = field } }

func WithMessageTransform(fn func([]Message, llm.Settings) []Message) Option {
	return func(o *options) { o.transformMessages = fn }
}

func WithBodyHook(fn func(body map[string]any, settings llm.Settings)) Option {
	return func(o *options) { o.prepareBody = fn }
}

func WithUsageHook(fn func(*llm.Usage)) Option {
	return func(o *options) { o.adjustUsage = fn }
}

func WithRetry(policy RetryPolicy) Option {
	return func(o *options) { o.retry = &policy }
}

func WithInlineThoughts() Option { return WithThoughtTags("<think>", "</think>") }

func WithThoughtTags(open, close string) Option {
	return func(o *options) { o.thoughtOpen, o.thoughtClose = open, close }
}

type Provider struct {
	retry    RetryPolicy
	name     string
	endpoint string
	apiKey   string
	headers  http.Header
	client   *http.Client
	opts     options
}

func New(cfg Config, opts ...Option) *Provider {
	p := &Provider{
		retry:   cfg.Retry,
		name:    cfg.Name,
		apiKey:  cfg.APIKey,
		client:  cfg.HTTPClient,
		headers: make(http.Header, len(cfg.Headers)),
		opts: options{
			path:           "/chat/completions",
			authHeader:     "Authorization",
			authPrefix:     "Bearer ",
			maxTokensField: "max_tokens",
			seedField:      "seed",
		},
	}
	for _, opt := range opts {
		opt(&p.opts)
	}
	if p.opts.retry != nil {
		p.retry = *p.opts.retry
	}
	if p.name == "" {
		p.name = "OpenAICompatible"
	}
	if p.client == nil {
		p.client = sharedClient
	}
	for k, v := range cfg.Headers {
		p.headers.Set(k, v)
	}
	p.endpoint = strings.TrimRight(cfg.BaseURL, "/") + p.opts.path
	if len(p.opts.query) > 0 {
		p.endpoint += "?" + p.opts.query.Encode()
	}
	return p
}

func (p *Provider) Name() string { return p.name }

type APIError struct {
	Provider   string
	StatusCode int
	Type       string
	Code       string
	Param      string
	Message    string
	Body       string
	RequestID  string
}

func (e *APIError) ErrorCategory() frames.ErrorCategory {
	switch {
	case e.Code == "content_filter":
		return frames.ErrorCategoryContentFiltered
	case e.Code == "invalid_api_key":
		return frames.ErrorCategoryAuthentication
	case e.Code == "insufficient_quota" || e.Type == "insufficient_quota":
		return frames.ErrorCategoryQuota
	case e.Code == "model_not_found" || e.Code == "DeploymentNotFound":
		return frames.ErrorCategoryConfiguration
	}
	return frames.ClassifyHTTPStatus(e.StatusCode)
}

func (e *APIError) Error() string {
	var b strings.Builder
	b.WriteString(e.Provider)
	b.WriteString(" API error")
	if e.StatusCode != 0 {
		fmt.Fprintf(&b, " (HTTP %d)", e.StatusCode)
	}
	if e.Type != "" || e.Code != "" {
		fmt.Fprintf(&b, " [%s]", strings.Trim(e.Type+" "+e.Code, " "))
	}
	b.WriteString(": ")
	if e.Message != "" {
		b.WriteString(e.Message)
	} else {
		b.WriteString(e.Body)
	}
	if e.RequestID != "" {
		fmt.Fprintf(&b, " (request id %s)", e.RequestID)
	}
	return b.String()
}

const CodeContentFilter = "content_filter"

type filterResults map[string]json.RawMessage

func (r filterResults) filtered() string {
	var names []string
	for name, raw := range r {
		var v struct {
			Filtered bool `json:"filtered"`
		}
		if json.Unmarshal(raw, &v) == nil && v.Filtered {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}

type wireError struct {
	Message    string          `json:"message"`
	Type       string          `json:"type"`
	Code       json.RawMessage `json:"code"`
	Param      string          `json:"param"`
	InnerError *struct {
		Code                string        `json:"code"`
		ContentFilterResult filterResults `json:"content_filter_result"`
	} `json:"innererror"`
}

func RequestID(h http.Header) string {
	return h.Get("x-request-id")
}

func ReadAPIError(provider string, resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var envelope struct {
		Error *wireError `json:"error"`
	}
	_ = json.Unmarshal(raw, &envelope)
	e := newAPIError(provider, resp.StatusCode, envelope.Error, strings.TrimSpace(string(raw)))
	e.RequestID = RequestID(resp.Header)
	return classify(e, envelope.Error, llm.RefusalPromptBlocked)
}

func classify(e *APIError, w *wireError, category llm.RefusalCategory) error {
	if e.Code != CodeContentFilter {
		return e
	}
	r := llm.Refusal{Category: category, Provider: e.Provider, Reason: CodeContentFilter, Message: e.Message}
	if w != nil && w.InnerError != nil {
		r.Detail = w.InnerError.ContentFilterResult.filtered()
	}
	return &llm.RefusalError{Refusal: r, Err: e}
}

func ParseStreamError(provider string, data []byte) error {
	var w wireError
	if err := json.Unmarshal(data, &w); err != nil || w.Message == "" {
		return newAPIError(provider, 0, nil, string(data))
	}
	return classify(newAPIError(provider, 0, &w, string(data)), &w, llm.RefusalContentFilter)
}

func (p *Provider) apiError(w *wireError, body, requestID string) error {
	e := newAPIError(p.name, 0, w, body)
	e.RequestID = requestID
	return classify(e, w, llm.RefusalContentFilter)
}

func ResponseFormatBody(f *llm.ResponseFormat) map[string]any {
	if !f.Structured() {
		return nil
	}
	if f.Type == llm.ResponseFormatJSONObject {
		return map[string]any{"type": llm.ResponseFormatJSONObject}
	}
	schema := map[string]any{"name": f.SchemaName(), "schema": f.Schema, "strict": f.Strict}
	if f.Description != "" {
		schema["description"] = f.Description
	}
	return map[string]any{"type": llm.ResponseFormatJSONSchema, "json_schema": schema}
}

func NormalizeStopReason(raw string) string {
	switch raw {
	case "":
		return ""
	case "stop", "eos", "end_turn":
		return llm.StopReasonStop
	case "length", "max_tokens", "max_output_tokens":
		return llm.StopReasonLength
	case "tool_calls", "function_call":
		return llm.StopReasonToolCalls
	case CodeContentFilter:
		return llm.StopReasonContentFilter
	}
	return llm.StopReasonOther
}

func newAPIError(provider string, status int, w *wireError, body string) *APIError {
	e := &APIError{Provider: provider, StatusCode: status, Body: body}
	if w != nil {
		e.Message, e.Type, e.Param = w.Message, w.Type, w.Param
		e.Code = strings.Trim(string(w.Code), `"`)
		if e.Code == "null" {
			e.Code = ""
		}
	}
	return e
}

func (p *Provider) BuildRequestBody(snapshot llm.Snapshot, settings llm.Settings) (map[string]any, error) {
	messages, err := p.convertMessages(snapshot, settings)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"model":    settings.Model,
		"messages": messages,
		"stream":   true,
	}
	if !p.opts.noStreamOptions {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if settings.Temperature != nil {
		body["temperature"] = *settings.Temperature
	}
	if settings.TopP != nil {
		body["top_p"] = *settings.TopP
	}
	if settings.MaxTokens != nil && p.opts.maxTokensField != "" {
		body[p.opts.maxTokensField] = *settings.MaxTokens
	}
	if settings.Seed != nil && p.opts.seedField != "" {
		body[p.opts.seedField] = *settings.Seed
	}
	if format := ResponseFormatBody(settings.ResponseFormat); format != nil {
		body["response_format"] = format
	}
	if !p.opts.noTools {
		if tools := convertTools(snapshot.Tools); tools != nil {
			body["tools"] = tools
			if choice := convertToolChoice(snapshot.ToolChoice); choice != nil {
				body["tool_choice"] = choice
			}
		}
	}
	if p.opts.prepareBody != nil {
		p.opts.prepareBody(body, settings)
	}
	for k, v := range settings.Extra {
		if k == "extra_body" {
			if m, ok := v.(map[string]any); ok {
				for ek, ev := range m {
					body[ek] = ev
				}
				continue
			}
		}
		body[k] = v
	}
	return body, nil
}

func (p *Provider) url(settings llm.Settings) string {
	if p.opts.urlFunc != nil {
		return p.opts.urlFunc(settings)
	}
	return p.endpoint
}

func (p *Provider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	body, err := p.BuildRequestBody(snapshot, settings)
	if err != nil {
		return err
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("%s: encode request: %w", p.name, err)
	}
	target := p.url(settings)
	resp, err := httpretry.Do(ctx, p.client, p.retry, func(ctx context.Context) (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		for k, v := range p.headers {
			req.Header[k] = v
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		if p.apiKey != "" && p.opts.authHeader != "" {
			req.Header.Set(p.opts.authHeader, p.opts.authPrefix+p.apiKey)
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
		return ReadAPIError(p.name, resp)
	}

	err = p.consume(ctx, resp.Body, RequestID(resp.Header), snapshot, emit)
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

type pendingCall struct {
	id   string
	name string
	args []byte
}

type streamState struct {
	calls   []*pendingCall
	byIndex map[int]*pendingCall
	usage   *llm.Usage
}

func (st *streamState) toolDelta(i int, d wireToolDelta) {
	var call *pendingCall
	switch {
	case d.Index != nil:
		call = st.byIndex[*d.Index]
		if call != nil && d.ID != "" && call.id != "" && d.ID != call.id {
			call = nil
		}
		if call == nil {
			call = &pendingCall{}
			st.calls = append(st.calls, call)
			if st.byIndex == nil {
				st.byIndex = make(map[int]*pendingCall)
			}
			st.byIndex[*d.Index] = call
		}
	case d.ID != "" || len(st.calls) == 0 || i > 0:
		call = &pendingCall{}
		st.calls = append(st.calls, call)
	default:
		call = st.calls[len(st.calls)-1]
	}
	if d.ID != "" {
		call.id = d.ID
	}
	if name := d.Function.Name; name != "" && name != call.name {
		call.name += name
	}
	call.args = append(call.args, d.Function.Arguments...)
}

type wireToolDelta struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

type wireChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		ContentFilterResults filterResults `json:"content_filter_results"`
		Delta                struct {
			Content          string          `json:"content"`
			Refusal          string          `json:"refusal"`
			ReasoningContent string          `json:"reasoning_content"`
			Reasoning        string          `json:"reasoning"`
			ToolCalls        []wireToolDelta `json:"tool_calls"`
			Audio            *struct {
				Transcript string `json:"transcript"`
			} `json:"audio"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *wireUsage `json:"usage"`
	Error *wireError `json:"error"`
}

func (p *Provider) consume(ctx context.Context, r io.Reader, requestID string, snapshot llm.Snapshot, emit func(llm.Event)) error {
	sse := NewSSEReader(r)
	var st streamState
	stop := ""
	info := llm.ResponseInfo{RequestID: requestID}
	infoSent := false
	filtered := ""
	var refusal strings.Builder
	var thoughts *thoughtFilter
	if p.opts.thoughtOpen != "" && p.opts.thoughtClose != "" {
		thoughts = newThoughtFilter(p.opts.thoughtOpen, p.opts.thoughtClose)
	}
	for {
		event, data, err := sse.Next()
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			return fmt.Errorf("%s: read stream: %w", p.name, err)
		}
		if bytes.Equal(data, []byte("[DONE]")) {
			break
		}
		var chunk wireChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			if string(event) == "error" {
				return p.apiError(nil, string(data), requestID)
			}
			continue
		}
		if chunk.Error != nil {
			return p.apiError(chunk.Error, string(data), requestID)
		}
		if !infoSent && chunk.ID != "" {
			info.ResponseID, info.Model = chunk.ID, chunk.Model
			infoSent = true
			emit(info)
		}
		if chunk.Usage != nil {
			st.usage = p.convertUsage(chunk.Usage)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		c := &chunk.Choices[0]
		if c.FinishReason != "" {
			stop = c.FinishReason
			if stop == CodeContentFilter {
				filtered = c.ContentFilterResults.filtered()
			}
		}
		d := &c.Delta
		if d.Refusal != "" {
			refusal.WriteString(d.Refusal)
		}
		if d.ReasoningContent != "" {
			emit(llm.ThoughtDelta{Text: d.ReasoningContent})
		} else if d.Reasoning != "" {
			emit(llm.ThoughtDelta{Text: d.Reasoning})
		}
		for i, tc := range d.ToolCalls {
			st.toolDelta(i, tc)
		}
		if d.Content != "" {
			if thoughts != nil {
				thoughts.write(d.Content, emit)
			} else {
				emit(llm.TextDelta{Text: d.Content})
			}
		} else if d.Audio != nil && d.Audio.Transcript != "" {
			emit(llm.TextDelta{Text: d.Audio.Transcript})
		}
	}

	if thoughts != nil {
		thoughts.flush(emit)
	}
	if !infoSent && info != (llm.ResponseInfo{}) {
		emit(info)
	}
	if st.usage != nil {
		emit(llm.UsageEvent{Usage: *st.usage})
	}
	if stop == CodeContentFilter || refusal.Len() > 0 {
		r := llm.Refusal{Category: llm.RefusalContentFilter, Provider: p.name, Reason: CodeContentFilter, Detail: filtered}
		normalized := llm.StopReasonContentFilter
		if stop != CodeContentFilter {
			r = llm.Refusal{Category: llm.RefusalModel, Provider: p.name, Reason: "refusal", Message: refusal.String()}
			normalized = llm.StopReasonRefusal
		}
		emit(r)
		emit(llm.Done{StopReason: normalized, RawStopReason: stop})
		return &llm.RefusalError{Refusal: r}
	}
	if len(st.calls) > 0 {
		var answered map[string]bool
		if p.opts.skipAnsweredCalls {
			answered = answeredCalls(snapshot)
		}
		for _, c := range st.calls {
			if c.name == "" {
				continue
			}
			if c.id == "" {
				c.id = newCallID()
			}
			if answered[c.id] {
				continue
			}
			args := c.args
			if len(bytes.TrimSpace(args)) == 0 {
				args = []byte("{}")
			}
			emit(llm.ToolCall{ID: c.id, Name: c.name, Arguments: args})
		}
	}
	emit(llm.Done{StopReason: NormalizeStopReason(stop), RawStopReason: stop})
	return nil
}

func (p *Provider) convertUsage(w *wireUsage) *llm.Usage {
	u := llm.Usage{
		PromptTokens:     w.PromptTokens,
		CompletionTokens: w.CompletionTokens,
		TotalTokens:      w.TotalTokens,
	}
	if w.PromptTokensDetails != nil {
		u.CacheReadTokens = w.PromptTokensDetails.CachedTokens
		u.CacheCreationTokens = w.PromptTokensDetails.CacheWriteTokens
	}
	if w.CompletionTokensDetails != nil {
		u.ReasoningTokens = w.CompletionTokensDetails.ReasoningTokens
	}
	if p.opts.adjustUsage != nil {
		p.opts.adjustUsage(&u)
	}
	return &u
}

func answeredCalls(snapshot llm.Snapshot) map[string]bool {
	out := make(map[string]bool)
	for _, m := range snapshot.Messages {
		for _, part := range m.Parts {
			if r, ok := part.(llm.ToolResultPart); ok {
				out[r.ToolCallID] = true
			}
		}
	}
	return out
}

func newCallID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "call_" + hex.EncodeToString(b[:])
}
