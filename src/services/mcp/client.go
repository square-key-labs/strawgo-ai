package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

const (
	DefaultConnectTimeout = 30 * time.Second
	DefaultCallTimeout    = 60 * time.Second
	maxErrorDetail        = 200
	fallbackResult        = "Sorry, could not call the mcp tool"
)

var (
	ErrNotConnected = errors.New("mcp client is not connected: call Start, Tools or RegisterTools first")
	ErrClosed       = errors.New("mcp client closed")
)

type ServerParams interface {
	transport() (sdk.Transport, error)
}

type StdioParams struct {
	Command           string
	Args              []string
	Env               []string
	Dir               string
	TerminateDuration time.Duration
}

func (p StdioParams) transport() (sdk.Transport, error) {
	if p.Command == "" {
		return nil, errors.New("mcp stdio: command is required")
	}
	cmd := exec.Command(p.Command, p.Args...)
	cmd.Env = p.Env
	cmd.Dir = p.Dir
	return &sdk.CommandTransport{Command: cmd, TerminateDuration: p.TerminateDuration}, nil
}

type StreamableHTTPParams struct {
	URL        string
	Headers    map[string]string
	HTTPClient *http.Client
	MaxRetries int
}

func (p StreamableHTTPParams) transport() (sdk.Transport, error) {
	if p.URL == "" {
		return nil, errors.New("mcp streamable http: url is required")
	}
	return &sdk.StreamableClientTransport{Endpoint: p.URL, HTTPClient: withHeaders(p.HTTPClient, p.Headers), MaxRetries: p.MaxRetries}, nil
}

type SSEParams struct {
	URL        string
	Headers    map[string]string
	HTTPClient *http.Client
}

func (p SSEParams) transport() (sdk.Transport, error) {
	if p.URL == "" {
		return nil, errors.New("mcp sse: url is required")
	}
	return &sdk.SSEClientTransport{Endpoint: p.URL, HTTPClient: withHeaders(p.HTTPClient, p.Headers)}, nil
}

type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	return t.base.RoundTrip(req)
}

func withHeaders(client *http.Client, headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return client
	}
	base := http.DefaultTransport
	out := &http.Client{}
	if client != nil {
		*out = *client
		if client.Transport != nil {
			base = client.Transport
		}
	}
	copied := make(map[string]string, len(headers))
	for k, v := range headers {
		copied[k] = v
	}
	out.Transport = &headerTransport{base: base, headers: copied}
	return out
}

type OutputFilter func(result any) (any, error)

type FunctionRegistrar interface {
	RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions)
}

type Option func(*Client)

func WithToolsFilter(names ...string) Option {
	return func(c *Client) {
		c.allow = make(map[string]struct{}, len(names))
		for _, n := range names {
			c.allow[n] = struct{}{}
		}
	}
}

func WithOutputFilter(tool string, filter OutputFilter) Option {
	return func(c *Client) { c.outputFilters[tool] = filter }
}

func WithToolArguments(tool string, fixed map[string]any) Option {
	return func(c *Client) { c.fixedArgs[tool] = fixed }
}

func WithConnectTimeout(d time.Duration) Option {
	return func(c *Client) { c.connectTimeout = d }
}

func WithCallTimeout(d time.Duration) Option {
	return func(c *Client) { c.callTimeout = d }
}

func WithFunctionOptions(opts llm.FunctionOptions) Option {
	return func(c *Client) { c.fnOpts = &opts }
}

func WithClientInfo(name, version string) Option {
	return func(c *Client) { c.impl = &sdk.Implementation{Name: name, Version: version} }
}

type session struct {
	cs     *sdk.ClientSession
	cancel context.CancelFunc
}

type Client struct {
	params         ServerParams
	impl           *sdk.Implementation
	allow          map[string]struct{}
	outputFilters  map[string]OutputFilter
	fixedArgs      map[string]map[string]any
	connectTimeout time.Duration
	callTimeout    time.Duration
	fnOpts         *llm.FunctionOptions

	sdkClient *sdk.Client
	life      context.Context
	stop      context.CancelFunc

	mu        sync.Mutex
	sess      *session
	requested bool
	dialing   chan struct{}
	gen       uint64
	watchers  sync.WaitGroup
}

func New(params ServerParams, opts ...Option) (*Client, error) {
	switch params.(type) {
	case StdioParams, *StdioParams, StreamableHTTPParams, *StreamableHTTPParams, SSEParams, *SSEParams:
	default:
		return nil, fmt.Errorf("mcp: unsupported server params %T", params)
	}
	c := &Client{
		params:         params,
		impl:           &sdk.Implementation{Name: "strawgo", Version: "1.0.0"},
		outputFilters:  map[string]OutputFilter{},
		fixedArgs:      map[string]map[string]any{},
		connectTimeout: DefaultConnectTimeout,
		callTimeout:    DefaultCallTimeout,
	}
	for _, opt := range opts {
		opt(c)
	}
	c.sdkClient = sdk.NewClient(c.impl, nil)
	c.life, c.stop = context.WithCancel(context.Background())
	return c, nil
}

func (c *Client) Start(ctx context.Context) error {
	_, err := c.session(ctx, true)
	return err
}

func (c *Client) Close() error {
	c.mu.Lock()
	c.requested = false
	c.gen++
	s := c.sess
	c.sess = nil
	c.mu.Unlock()
	var err error
	if s != nil {
		err = s.close()
	}
	c.watchers.Wait()
	return err
}

func (s *session) close() error {
	err := s.cs.Close()
	s.cancel()
	return err
}

func (c *Client) session(ctx context.Context, owner bool) (*sdk.ClientSession, error) {
	for {
		c.mu.Lock()
		if owner {
			c.requested = true
		}
		if c.sess != nil {
			cs := c.sess.cs
			c.mu.Unlock()
			return cs, nil
		}
		if !c.requested {
			c.mu.Unlock()
			return nil, ErrNotConnected
		}
		if c.dialing != nil {
			ch := c.dialing
			c.mu.Unlock()
			select {
			case <-ch:
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		ch := make(chan struct{})
		c.dialing = ch
		gen := c.gen
		c.mu.Unlock()

		s, err := c.connect(ctx)

		c.mu.Lock()
		c.dialing = nil
		close(ch)
		if err != nil {
			c.mu.Unlock()
			return nil, err
		}
		if !c.requested || gen != c.gen {
			c.mu.Unlock()
			_ = s.close()
			return nil, ErrClosed
		}
		c.sess = s
		c.watchers.Add(1)
		c.mu.Unlock()
		go c.watch(s)
		return s.cs, nil
	}
}

func (c *Client) connect(ctx context.Context) (*session, error) {
	t, err := c.params.transport()
	if err != nil {
		return nil, err
	}
	dialCtx, cancel := context.WithCancel(c.life)
	stopCaller := context.AfterFunc(ctx, cancel)
	var timer *time.Timer
	if c.connectTimeout > 0 {
		timer = time.AfterFunc(c.connectTimeout, cancel)
	}
	cs, err := c.sdkClient.Connect(dialCtx, t, nil)
	callerOK := stopCaller()
	timerOK := timer == nil || timer.Stop()
	if err == nil && (!callerOK || !timerOK) {
		_ = cs.Close()
		err = context.DeadlineExceeded
		if ctx.Err() != nil {
			err = ctx.Err()
		}
	}
	if err != nil {
		cancel()
		return nil, fmt.Errorf("mcp connect: %w", err)
	}
	return &session{cs: cs, cancel: cancel}, nil
}

func (c *Client) watch(s *session) {
	defer c.watchers.Done()
	_ = s.cs.Wait()
	if c.drop(s) {
		logger.Warn("[MCP] Server connection lost; the next tool call reconnects")
	}
}

func (c *Client) drop(s *session) bool {
	c.mu.Lock()
	if c.sess != s {
		c.mu.Unlock()
		return false
	}
	c.sess = nil
	c.mu.Unlock()
	_ = s.close()
	return true
}

func (c *Client) dropSession(cs *sdk.ClientSession) {
	c.mu.Lock()
	s := c.sess
	c.mu.Unlock()
	if s != nil && s.cs == cs {
		c.drop(s)
	}
}

func (c *Client) Tools(ctx context.Context) (llm.Tools, error) {
	cs, err := c.session(ctx, true)
	if err != nil {
		return llm.Tools{}, err
	}
	var out []llm.FunctionSchema
	seen := map[string]bool{}
	for tool, err := range cs.Tools(ctx, nil) {
		if err != nil {
			if isConnectionLost(err) {
				c.dropSession(cs)
			}
			return llm.Tools{}, fmt.Errorf("mcp list tools: %w", err)
		}
		seen[tool.Name] = true
		if c.allow != nil {
			if _, ok := c.allow[tool.Name]; !ok {
				continue
			}
		}
		schema, err := c.convert(tool)
		if err != nil {
			logger.Warn("[MCP] Skipping tool %q: %v", tool.Name, err)
			continue
		}
		out = append(out, schema)
	}
	for name := range c.fixedArgs {
		if !seen[name] {
			logger.Warn("[MCP] Fixed arguments configured for tool %q, which the server does not advertise", name)
		}
	}
	return llm.Tools{Functions: out}, nil
}

func (c *Client) RegisterTools(ctx context.Context, registrar FunctionRegistrar) (llm.Tools, error) {
	tools, err := c.Tools(ctx)
	if err != nil {
		return llm.Tools{}, err
	}
	for _, fn := range tools.Functions {
		if c.fnOpts != nil {
			registrar.RegisterFunction(fn.Name, c.Handler(), *c.fnOpts)
		} else {
			registrar.RegisterFunction(fn.Name, c.Handler())
		}
	}
	return tools, nil
}

func (c *Client) Handler() llm.FunctionHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) (llm.FunctionResult, error) {
		value, err := c.CallTool(ctx, params.FunctionName, params.Arguments)
		if err != nil && (ctx.Err() != nil || errors.Is(err, ErrNotConnected) || errors.Is(err, ErrClosed)) {
			return llm.FunctionResult{}, err
		}
		if err != nil {
			logger.Warn("[MCP] %v", err)
			if isEmpty(value) {
				value = err.Error()
			}
		}
		return llm.FunctionResult{Value: value}, nil
	}
}

func (c *Client) convert(tool *sdk.Tool) (llm.FunctionSchema, error) {
	schema, err := schemaMap(tool.InputSchema)
	if err != nil {
		return llm.FunctionSchema{}, err
	}
	properties, _ := schema["properties"].(map[string]any)
	var required []string
	switch r := schema["required"].(type) {
	case []any:
		for _, v := range r {
			if s, ok := v.(string); ok {
				required = append(required, s)
			}
		}
	case []string:
		required = append(required, r...)
	}
	if fixed := c.fixedArgs[tool.Name]; len(fixed) > 0 {
		trimmed := make(map[string]any, len(properties))
		for k, v := range properties {
			if _, hidden := fixed[k]; !hidden {
				trimmed[k] = v
			}
		}
		properties = trimmed
		kept := required[:0]
		for _, r := range required {
			if _, hidden := fixed[r]; !hidden {
				kept = append(kept, r)
			}
		}
		required = kept
	}
	if properties == nil {
		properties = map[string]any{}
	}
	return llm.FunctionSchema{
		Name:        tool.Name,
		Description: tool.Description,
		Properties:  properties,
		Required:    required,
	}, nil
}

func schemaMap(v any) (map[string]any, error) {
	switch s := v.(type) {
	case nil:
		return map[string]any{}, nil
	case map[string]any:
		return s, nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("input schema: %w", err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("input schema: %w", err)
	}
	return out, nil
}

func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]any) (any, error) {
	if fixed := c.fixedArgs[name]; len(fixed) > 0 {
		merged := make(map[string]any, len(arguments)+len(fixed))
		for k, v := range arguments {
			merged[k] = v
		}
		for k, v := range fixed {
			merged[k] = v
		}
		arguments = merged
	}

	cs, err := c.session(ctx, false)
	if err != nil {
		return nil, err
	}
	callCtx := ctx
	if c.callTimeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, c.callTimeout)
		defer cancel()
	}
	res, err := cs.CallTool(callCtx, &sdk.CallToolParams{Name: name, Arguments: arguments})
	if err != nil {
		if isConnectionLost(err) {
			logger.Warn("[MCP] Dropping the session tool %q ran on", name)
			c.dropSession(cs)
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		msg := fmt.Sprintf("Error calling mcp tool %s: %s", name, errorDetail(err))
		return msg, errors.New(msg)
	}

	var value any
	text := resultText(res)
	switch {
	case res.StructuredContent != nil:
		value = res.StructuredContent
	case text != "":
		value = text
	}
	if filter := c.outputFilters[name]; filter != nil {
		filtered, ferr := filter(value)
		if ferr != nil {
			logger.Error("[MCP] Output filter for %q failed: %v", name, ferr)
			value = nil
		} else {
			value = filtered
		}
	}
	if isEmpty(value) {
		value = fallbackResult
	}
	if res.IsError {
		detail := text
		if detail == "" {
			detail = "tool reported an error"
		}
		return value, fmt.Errorf("mcp tool %s: %s", name, truncate(detail))
	}
	return value, nil
}

func resultText(res *sdk.CallToolResult) string {
	var b strings.Builder
	for _, content := range res.Content {
		if t, ok := content.(*sdk.TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

func isEmpty(v any) bool {
	switch s := v.(type) {
	case nil:
		return true
	case string:
		return s == ""
	}
	return false
}

func isConnectionLost(err error) bool {
	return errors.Is(err, sdk.ErrConnectionClosed) ||
		errors.Is(err, sdk.ErrSessionMissing) ||
		errors.Is(err, io.EOF) ||
		errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, io.ErrClosedPipe) ||
		errors.Is(err, net.ErrClosed)
}

func errorDetail(err error) string {
	return truncate(err.Error())
}

func truncate(s string) string {
	if len(s) > maxErrorDetail {
		return s[:maxErrorDetail] + "..."
	}
	return s
}
