// Package openai_responses provides an LLM service backed by OpenAI's
// Responses API (POST /v1/responses) — a stateful chat endpoint that
// stores conversation history server-side. Each response carries an
// `id` that the next request references via `previous_response_id`, so
// the client only ships the new user turn instead of the full history.
//
// Why a separate package from `openai/`: the Responses wire format is
// different from /v1/chat/completions (input items vs messages,
// instructions vs system role, output_item events vs choices.delta).
// Mirrors pipecat's OpenAIResponsesLLMService introduced in v0.0.106.
package openai_responses

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// LLMService talks to OpenAI's stateful Responses API and streams output
// text deltas as LLMTextFrames. Conversation continuity is handled via
// `previous_response_id`; on cache miss / invalid id the next request
// falls back to a full-context replay.
type LLMService struct {
	*processors.BaseProcessor
	apiKey      string
	baseURL     string
	model       string
	temperature float64
	context     *services.LLMContext
	log         *logger.Logger
	ctx         context.Context
	cancel      context.CancelFunc

	toolWarnOnce sync.Once

	// previousResponseID is the server-side response handle from the most
	// recent successful generation. When set, future requests reference
	// it via `previous_response_id` so we ship only the newest user
	// turn(s) instead of the full Messages array.
	//
	// checkpointLen is the LLMContext.Messages length at the moment we
	// captured previousResponseID. On the next incremental turn we send
	// `Messages[checkpointLen:]` — every message added since the last
	// successful response, not just the trailing one. This matters for
	// turns that bundle a tool result and a follow-up user message.
	prevMu             sync.Mutex
	previousResponseID string
	checkpointLen      int

	// Request-scoped context for cancellable streaming (protected by streamMu).
	requestCtx    context.Context
	requestCancel context.CancelFunc
	isGenerating  bool
	lastContextAt time.Time
	streamMu      sync.Mutex
}

// LLMConfig configures an OpenAI Responses LLM service.
type LLMConfig struct {
	APIKey       string
	Model        string  // e.g. "gpt-4o", "gpt-4.1-mini"
	SystemPrompt string  // sent as `instructions`
	Temperature  float64 // 0 if unset → omitted from request
	BaseURL      string  // optional override; defaults to DefaultResponsesBaseURL
}

const (
	// DefaultResponsesBaseURL is the public OpenAI Responses endpoint.
	DefaultResponsesBaseURL = "https://api.openai.com/v1"
	// DefaultResponsesModel is the default chat model.
	DefaultResponsesModel = "gpt-4o-mini"
)

// NewLLMService creates a new OpenAI Responses LLM service.
func NewLLMService(config LLMConfig) *LLMService {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultResponsesBaseURL
	}
	model := config.Model
	if model == "" {
		model = DefaultResponsesModel
	}

	s := &LLMService{
		apiKey:      config.APIKey,
		baseURL:     baseURL,
		model:       model,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
		log:         logger.WithPrefix("OpenAIResponsesLLM"),
	}
	s.BaseProcessor = processors.NewBaseProcessor("OpenAIResponses", s)
	return s
}

func (s *LLMService) SetModel(model string)         { s.model = model }
func (s *LLMService) SetSystemPrompt(prompt string) { s.context.SystemPrompt = prompt }
func (s *LLMService) SetTemperature(temp float64)   { s.temperature = temp }

func (s *LLMService) AddMessage(role, content string) {
	s.context.Messages = append(s.context.Messages, services.LLMMessage{
		Role:    role,
		Content: content,
	})
}

// ClearContext wipes both the local message history and the
// server-side previous_response_id chain so the next request starts
// fresh.
func (s *LLMService) ClearContext() {
	s.context.Clear()
	s.prevMu.Lock()
	s.previousResponseID = ""
	s.checkpointLen = 0
	s.prevMu.Unlock()
}

// PreviousResponseID returns the most recent server-side response
// handle. Useful for tests and for callers that want to checkpoint
// conversations.
func (s *LLMService) PreviousResponseID() string {
	s.prevMu.Lock()
	defer s.prevMu.Unlock()
	return s.previousResponseID
}

// SetPreviousResponseID seeds the previous_response_id, e.g. when
// resuming a saved conversation. Resets the checkpoint to "no messages
// covered yet" — callers manually seeding an id are typically also
// providing the full message history that produced it.
func (s *LLMService) SetPreviousResponseID(id string) {
	s.prevMu.Lock()
	s.previousResponseID = id
	s.checkpointLen = 0
	s.prevMu.Unlock()
}

// warnToolStreaming logs a one-time warning that the adapter forwards
// tools to the provider but does not yet stream tool-call lifecycle
// frames. Logged once per service instance.
func (s *LLMService) warnToolStreaming() {
	s.toolWarnOnce.Do(func() {
		s.log.Warn("Tools forwarded but tool-call SSE events are not parsed; function calls will arrive as text. Track follow-up before relying on this for production tool-using flows.")
	})
}

// captureCheckpoint atomically records (id, len) so the next incremental
// turn knows which trailing messages have not yet been seen by the
// server.
func (s *LLMService) captureCheckpoint(id string, msgsLen int) {
	s.prevMu.Lock()
	s.previousResponseID = id
	s.checkpointLen = msgsLen
	s.prevMu.Unlock()
}

// readCheckpoint atomically returns (id, len).
func (s *LLMService) readCheckpoint() (string, int) {
	s.prevMu.Lock()
	defer s.prevMu.Unlock()
	return s.previousResponseID, s.checkpointLen
}

func (s *LLMService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.log.Info("Initialized with model %s", s.model)
	return nil
}

func (s *LLMService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	return nil
}

func (s *LLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.streamMu.Lock()
		timeSinceContext := time.Since(s.lastContextAt)
		isNewContext := timeSinceContext < 100*time.Millisecond

		s.log.Warn("Interruption received (isGenerating=%v, timeSinceContext=%v)", s.isGenerating, timeSinceContext)

		if isNewContext {
			s.log.Debug("Ignoring interruption — new context just received (%v ago)", timeSinceContext)
			s.streamMu.Unlock()
			return s.PushFrame(frame, direction)
		}

		if s.isGenerating && s.requestCancel != nil {
			s.log.Warn("Cancelling ongoing stream")
			s.requestCancel()
			s.isGenerating = false
		}
		s.streamMu.Unlock()
		return s.PushFrame(frame, direction)
	}

	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			s.log.Debug("Received LLMContextFrame with %d messages", len(llmContext.Messages))

			s.streamMu.Lock()
			s.lastContextAt = time.Now()
			s.streamMu.Unlock()

			s.context = llmContext

			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			if err := s.generateResponseFromContext(llmContext); err != nil {
				if s.requestCtx != nil && s.requestCtx.Err() == context.Canceled {
					s.log.Debug("Stream cancelled by interruption")
				} else {
					s.log.Error("Error generating response: %v", err)
					s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				}
			}

			s.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
		}
		return nil
	}

	return s.PushFrame(frame, direction)
}

// generateResponseFromContext sends a request to /v1/responses. It tries
// the incremental form first when previousResponseID is non-empty
// (sending only the trailing un-replayed messages); on a 4xx that
// indicates a stale/missing previous_response_id it retries with the
// full conversation.
func (s *LLMService) generateResponseFromContext(llmCtx *services.LLMContext) error {
	parentCtx := s.ctx
	if parentCtx == nil {
		parentCtx = context.Background()
	}

	s.streamMu.Lock()
	s.requestCtx, s.requestCancel = context.WithCancel(parentCtx)
	s.isGenerating = true
	s.streamMu.Unlock()

	defer func() {
		s.streamMu.Lock()
		s.isGenerating = false
		if s.requestCancel != nil {
			s.requestCancel()
		}
		s.requestCancel = nil
		s.streamMu.Unlock()
	}()

	prevID, checkpointLen := s.readCheckpoint()

	// Incremental path: ship every message added since the last
	// captured checkpoint. This handles bundled tool-result + user
	// follow-up turns correctly — sending only [len-1] would drop the
	// tool result silently.
	if prevID != "" && checkpointLen <= len(llmCtx.Messages) {
		newMessages := llmCtx.Messages[checkpointLen:]
		if len(newMessages) > 0 {
			err := s.streamRequest(llmCtx, prevID, newMessages, len(llmCtx.Messages))
			if err == nil {
				return nil
			}
			if isPreviousResponseIDError(err) {
				s.log.Warn("previous_response_id rejected (%v); falling back to full-context replay", err)
				s.SetPreviousResponseID("")
			} else {
				return err
			}
		}
	}

	// Full-context path.
	return s.streamRequest(llmCtx, "", llmCtx.Messages, len(llmCtx.Messages))
}

// streamRequest issues one POST /v1/responses request and consumes the
// SSE stream. totalMessagesAfter is the total LLMContext.Messages
// length at the moment of the request — used to checkpoint the cursor
// when `response.completed` arrives, so the next incremental turn ships
// only messages added after this one.
//
// It updates previousResponseID on `response.completed` and emits
// `LLMTextFrame`s on `response.output_text.delta`.
func (s *LLMService) streamRequest(llmCtx *services.LLMContext, previousResponseID string, messages []services.LLMMessage, totalMessagesAfter int) error {
	requestBody := map[string]interface{}{
		"model":  s.model,
		"input":  buildInputItems(messages),
		"stream": true,
	}
	if llmCtx.SystemPrompt != "" {
		requestBody["instructions"] = llmCtx.SystemPrompt
	}
	if previousResponseID != "" {
		requestBody["previous_response_id"] = previousResponseID
	}
	if s.temperature != 0 {
		requestBody["temperature"] = s.temperature
	}
	if len(llmCtx.Tools) > 0 {
		// Responses API tool shape mirrors chat-completions: top-level
		// type/name/description/parameters under `tools[*].function`.
		tools := make([]map[string]interface{}, 0, len(llmCtx.Tools))
		for _, tool := range llmCtx.Tools {
			tools = append(tools, map[string]interface{}{
				"type": tool.Type,
				"function": map[string]interface{}{
					"name":        tool.Function.Name,
					"description": tool.Function.Description,
					"parameters":  tool.Function.Parameters,
				},
			})
		}
		requestBody["tools"] = tools
		if llmCtx.ToolChoice != nil {
			requestBody["tool_choice"] = llmCtx.ToolChoice
		}
		// Tools are forwarded to the provider so the model can call
		// them, but this adapter does not yet parse `response.output_item`
		// tool-call SSE events into FunctionCallInProgressFrame /
		// FunctionCallResultFrame. Tool-call output will arrive as text
		// instead. Matches the current Groq adapter's gap; full tool
		// streaming is a follow-up that lands alongside the assistant
		// aggregator's tool-flow integration.
		s.warnToolStreaming()
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(s.requestCtx, "POST", s.baseURL+"/responses", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", s.apiKey))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		if s.requestCtx.Err() == context.Canceled {
			return nil
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return &apiError{Status: resp.StatusCode, Body: string(body)}
	}

	return s.consumeSSE(llmCtx, resp.Body, totalMessagesAfter)
}

// consumeSSE reads the Responses-API event stream. Each event is
// `event: <name>\ndata: <json>\n\n`. We care about three kinds:
//   - `response.output_text.delta` — emit LLMTextFrame
//   - `response.completed` — record the response id; the cursor is
//     captured AFTER this function returns so the cursor reflects the
//     real post-append message length (an empty-text completed event
//     would otherwise leave the cursor pointing at a phantom slot)
//   - everything else — ignore
func (s *LLMService) consumeSSE(llmCtx *services.LLMContext, body io.Reader, totalMessagesAfter int) error {
	_ = totalMessagesAfter // kept for caller-side signature symmetry; cursor capture is now post-stream

	var completedID string
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(body)
	// Default buffer is 64KB; raise to 1MB for tool-call argument blobs.
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		select {
		case <-s.requestCtx.Done():
			s.log.Debug("Stream interrupted, stopping generation")
			return nil
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var ev struct {
			Type     string `json:"type"`
			Delta    string `json:"delta"`
			Response struct {
				ID string `json:"id"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}

		switch ev.Type {
		case "response.output_text.delta":
			if ev.Delta == "" {
				continue
			}
			fullResponse.WriteString(ev.Delta)
			s.PushFrame(frames.NewLLMTextFrame(ev.Delta), frames.Downstream)
		case "response.completed":
			if ev.Response.ID != "" {
				completedID = ev.Response.ID
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if s.requestCtx.Err() == context.Canceled {
			return nil
		}
		return err
	}

	response := fullResponse.String()
	if response != "" {
		llmCtx.AddAssistantMessage(response)
		s.log.Debug("Assistant: %s", response)
	}

	// Capture the cursor AFTER any assistant append so the next
	// incremental turn ships exactly Messages[len:] — every message
	// added after this generation. Empty-text completed events leave
	// the cursor at the same position they found it, which is correct.
	if completedID != "" {
		s.captureCheckpoint(completedID, len(llmCtx.Messages))
	}
	return nil
}

// buildInputItems converts strawgo's LLMMessages into Responses-API
// `input` items. The Responses API uses `{role, content}` items where
// content is a list of content parts; we use the simplest text-only
// shape.
func buildInputItems(messages []services.LLMMessage) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(messages))
	for _, msg := range messages {
		// Responses API does not have a "developer" role; map to "user"
		// (matches the existing pattern in other strawgo LLM adapters
		// for async-tool-result developer messages).
		role := msg.Role
		if role == "developer" {
			role = "user"
		}
		// Skip "system" — system messages belong in `instructions`, not
		// in input items. The user aggregator's context already routes
		// the system prompt via SystemPrompt, but if a caller embedded a
		// system message in Messages we drop it here rather than 400'ing.
		if role == "system" {
			continue
		}
		item := map[string]interface{}{"role": role}
		if msg.Content != "" {
			item["content"] = msg.Content
		}
		out = append(out, item)
	}
	return out
}

// apiError is returned from streamRequest on any non-200 HTTP response.
// We sniff its Body for the well-known previous_response_id failure
// modes so the caller can fall back to a full-context replay.
type apiError struct {
	Status int
	Body   string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("OpenAI Responses API error %d: %s", e.Status, e.Body)
}

// isPreviousResponseIDError reports whether a 400/404 error from the
// Responses API indicates that the supplied previous_response_id is
// stale, expired, missing, or otherwise unusable. Triggers the
// fallback to a full-context replay.
//
// Detection is layered: first parse the structured `error` envelope
// (preferred — robust against future humanoid wording changes), then
// fall back to a substring scan for older/uncoded responses.
func isPreviousResponseIDError(err error) bool {
	apiErr, ok := err.(*apiError)
	if !ok {
		return false
	}
	if apiErr.Status != http.StatusBadRequest && apiErr.Status != http.StatusNotFound {
		return false
	}
	// Structured form. OpenAI's Responses API surfaces a wrapped
	// `error: { code, type, param, message }` JSON object.
	var envelope struct {
		Error struct {
			Code    string `json:"code"`
			Type    string `json:"type"`
			Param   string `json:"param"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if jsonErr := json.Unmarshal([]byte(apiErr.Body), &envelope); jsonErr == nil {
		code := strings.ToLower(envelope.Error.Code)
		param := strings.ToLower(envelope.Error.Param)
		if param == "previous_response_id" {
			return true
		}
		if strings.Contains(code, "previous_response") || strings.Contains(code, "response_expired") || strings.Contains(code, "response_not_found") {
			return true
		}
		// Final layer for structured but uncoded errors: scan the
		// message field rather than the whole body.
		msg := strings.ToLower(envelope.Error.Message)
		if matchStaleIDText(msg) {
			return true
		}
	}
	// Unstructured fallback (older responses, proxies, etc.).
	return matchStaleIDText(strings.ToLower(apiErr.Body))
}

func matchStaleIDText(body string) bool {
	return strings.Contains(body, "previous_response_id") ||
		strings.Contains(body, "previous response") ||
		strings.Contains(body, "no longer available") ||
		strings.Contains(body, "expired")
}
