package anthropic

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

const (
	// DefaultBaseURL is the default Anthropic API endpoint
	DefaultBaseURL = "https://api.anthropic.com/v1"
	// DefaultModel is the default Claude model
	DefaultModel = "claude-sonnet-4-6"
	// DefaultMaxTokens is the default max tokens for responses
	DefaultMaxTokens = 4096
	// APIVersion is the Anthropic API version header value
	APIVersion = "2023-06-01"
)

// LLMService provides language model capabilities using Anthropic's Claude API
type LLMService struct {
	*processors.BaseProcessor
	apiKey      string
	baseURL     string
	model       string
	maxTokens   int
	temperature float64
	context     *services.LLMContext
	log         *logger.Logger
	ctx         context.Context
	cancel      context.CancelFunc

	// Request-scoped context for cancellable streaming (protected by streamMu)
	requestCtx    context.Context
	requestCancel context.CancelFunc
	isGenerating  bool
	lastContextAt time.Time  // When we last received a new context (for interruption filtering)
	streamMu      sync.Mutex // Protects requestCancel, isGenerating, and lastContextAt
}

// LLMConfig holds configuration for Anthropic Claude
type LLMConfig struct {
	APIKey       string
	Model        string // e.g., "claude-sonnet-4-6", "claude-3-haiku-20240307"
	SystemPrompt string
	Temperature  float64
	MaxTokens    int    // Default: 4096
	BaseURL      string // Optional: override default Anthropic API URL
}

// NewLLMService creates a new Anthropic LLM service
func NewLLMService(config LLMConfig) *LLMService {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	model := config.Model
	if model == "" {
		model = DefaultModel
	}

	maxTokens := config.MaxTokens
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}

	s := &LLMService{
		apiKey:      config.APIKey,
		baseURL:     baseURL,
		model:       model,
		maxTokens:   maxTokens,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
		log:         logger.WithPrefix("AnthropicLLM"),
	}
	s.BaseProcessor = processors.NewBaseProcessor("Anthropic", s)
	return s
}

func (s *LLMService) SetModel(model string) {
	s.model = model
}

func (s *LLMService) SetSystemPrompt(prompt string) {
	s.context.SystemPrompt = prompt
}

func (s *LLMService) SetTemperature(temp float64) {
	s.temperature = temp
}

func (s *LLMService) AddMessage(role, content string) {
	s.context.Messages = append(s.context.Messages, services.LLMMessage{
		Role:    role,
		Content: content,
	})
}

func (s *LLMService) ClearContext() {
	s.context.Clear()
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
	// Handle InterruptionFrame - CRITICAL: Stop streaming immediately
	// BUT: If we just received a new context (within 100ms), this interruption is for
	// the OLD response, not the new one. Don't cancel in that case.
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.streamMu.Lock()
		timeSinceContext := time.Since(s.lastContextAt)
		isNewContext := timeSinceContext < 100*time.Millisecond

		s.log.Warn("Interruption received (isGenerating=%v, timeSinceContext=%v)", s.isGenerating, timeSinceContext)

		if isNewContext {
			// This interruption is for the OLD response, not the new one we just started
			s.log.Debug("Ignoring interruption - new context was just received (%v ago)", timeSinceContext)
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

	// Handle LLMContextFrame (from aggregators)
	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			s.log.Debug("Received LLMContextFrame with %d messages", len(llmContext.Messages))

			// Record when we received this context (for interruption filtering)
			s.streamMu.Lock()
			s.lastContextAt = time.Now()
			s.streamMu.Unlock()

			// Update our context reference
			s.context = llmContext

			// Send LLM response start marker
			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			// Generate response using the provided context
			if err := s.generateResponseFromContext(llmContext); err != nil {
				// Only log error if not cancelled
				if s.requestCtx != nil && s.requestCtx.Err() == context.Canceled {
					s.log.Debug("Stream cancelled by interruption")
				} else {
					s.log.Error("Error generating response: %v", err)
					s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				}
			}

			// Send LLM response end marker
			s.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
		}
		return nil
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

// generateResponseFromContext generates a response using the Anthropic Messages API
// Supports streaming via SSE, tool calling, and interruption cancellation
func (s *LLMService) generateResponseFromContext(llmCtx *services.LLMContext) error {
	// Create cancellable context for this request
	// Use background context if s.ctx is nil (Initialize not called yet)
	parentCtx := s.ctx
	if parentCtx == nil {
		parentCtx = context.Background()
	}

	// Lock to safely set stream state (read by InterruptionFrame handler)
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

	// Build Anthropic-format messages
	// Anthropic differs from OpenAI:
	// - System prompt is a top-level field, not a message
	// - Tool results use role "user" with tool_result content blocks
	// - Assistant tool calls use content blocks, not separate tool_calls field
	messages := []interface{}{}

	for _, msg := range llmCtx.Messages {
		switch msg.Role {
		case "system":
			// Skip - system prompt goes in top-level "system" field
			continue
		case "tool":
			// Convert to Anthropic tool_result format:
			// {role: "user", content: [{type: "tool_result", tool_use_id: "xxx", content: "result"}]}
			messages = append(messages, map[string]interface{}{
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type":        "tool_result",
						"tool_use_id": msg.ToolCallID,
						"content":     msg.Content,
					},
				},
			})
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				// Assistant message with tool calls -> content blocks
				content := []interface{}{}
				if msg.Content != "" {
					content = append(content, map[string]interface{}{
						"type": "text",
						"text": msg.Content,
					})
				}
				for _, tc := range msg.ToolCalls {
					var input interface{}
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
						input = map[string]interface{}{}
					}
					content = append(content, map[string]interface{}{
						"type":  "tool_use",
						"id":    tc.ID,
						"name":  tc.Function.Name,
						"input": input,
					})
				}
				messages = append(messages, map[string]interface{}{
					"role":    "assistant",
					"content": content,
				})
			} else {
				messages = append(messages, map[string]interface{}{
					"role":    "assistant",
					"content": msg.Content,
				})
			}
		default:
			// "user" and any other roles
			messages = append(messages, map[string]interface{}{
				"role":    msg.Role,
				"content": msg.Content,
			})
		}
	}

	// Build request body
	requestBody := map[string]interface{}{
		"model":      s.model,
		"max_tokens": s.maxTokens,
		"messages":   messages,
		"stream":     true,
	}

	// System prompt goes as top-level field (not a message)
	if llmCtx.SystemPrompt != "" {
		requestBody["system"] = llmCtx.SystemPrompt
	}

	// Add temperature if set
	if s.temperature > 0 {
		requestBody["temperature"] = s.temperature
	}

	// Add tools if present in context
	if len(llmCtx.Tools) > 0 {
		tools := []map[string]interface{}{}
		for _, tool := range llmCtx.Tools {
			tools = append(tools, map[string]interface{}{
				"name":         tool.Function.Name,
				"description":  tool.Function.Description,
				"input_schema": tool.Function.Parameters,
			})
		}
		requestBody["tools"] = tools

		// Map tool_choice from OpenAI format to Anthropic format
		if llmCtx.ToolChoice != nil {
			switch v := llmCtx.ToolChoice.(type) {
			case string:
				switch v {
				case "auto":
					requestBody["tool_choice"] = map[string]interface{}{"type": "auto"}
				case "required":
					requestBody["tool_choice"] = map[string]interface{}{"type": "any"}
				case "none":
					// Anthropic doesn't have "none" - omit tool_choice
				default:
					// Specific function name
					requestBody["tool_choice"] = map[string]interface{}{
						"type": "tool",
						"name": v,
					}
				}
			default:
				// Pass through as-is (already in Anthropic format)
				requestBody["tool_choice"] = llmCtx.ToolChoice
			}
		}
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	// Use cancellable context so interruption can stop the request
	req, err := http.NewRequestWithContext(s.requestCtx, "POST", s.baseURL+"/messages", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	// Anthropic uses x-api-key header (not Bearer token like OpenAI)
	req.Header.Set("x-api-key", s.apiKey)
	req.Header.Set("anthropic-version", APIVersion)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		// Check if cancelled by interruption
		if s.requestCtx.Err() == context.Canceled {
			return nil // Not an error, just interrupted
		}
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Anthropic API error: %s", string(body))
	}

	// Parse SSE stream
	// Anthropic SSE format uses event/data pairs with a "type" field in the data JSON:
	//   event: content_block_delta
	//   data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	// Track tool use accumulation across content blocks
	type toolUseBlock struct {
		id        string
		name      string
		inputJSON strings.Builder
	}
	activeToolUses := map[int]*toolUseBlock{}
	var completedToolCalls []services.ToolCall

	for scanner.Scan() {
		// Check if interrupted
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

		// Parse the type field to determine event kind
		var baseEvent struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(data), &baseEvent); err != nil {
			continue
		}

		switch baseEvent.Type {
		case "content_block_start":
			var event struct {
				Index        int `json:"index"`
				ContentBlock struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}
			if event.ContentBlock.Type == "tool_use" {
				activeToolUses[event.Index] = &toolUseBlock{
					id:   event.ContentBlock.ID,
					name: event.ContentBlock.Name,
				}
			}

		case "content_block_delta":
			var event struct {
				Index int `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
				fullResponse.WriteString(event.Delta.Text)
				// Emit raw LLMTextFrame - sentence splitting handled by SentenceAggregator
				textFrame := frames.NewLLMTextFrame(event.Delta.Text)
				s.PushFrame(textFrame, frames.Downstream)
			} else if event.Delta.Type == "input_json_delta" {
				if tu, ok := activeToolUses[event.Index]; ok {
					tu.inputJSON.WriteString(event.Delta.PartialJSON)
				}
			}

		case "content_block_stop":
			var event struct {
				Index int `json:"index"`
			}
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			if tu, ok := activeToolUses[event.Index]; ok {
				// Parse accumulated JSON arguments
				var args map[string]interface{}
				inputStr := tu.inputJSON.String()
				if inputStr != "" {
					if err := json.Unmarshal([]byte(inputStr), &args); err != nil {
						args = map[string]interface{}{}
					}
				} else {
					args = map[string]interface{}{}
				}

				// Emit FunctionCallInProgressFrame
				funcFrame := frames.NewFunctionCallInProgressFrame(
					tu.id,
					tu.name,
					args,
					true, // cancelOnInterruption
				)
				s.PushFrame(funcFrame, frames.Downstream)

				// Track for context update
				completedToolCalls = append(completedToolCalls, services.ToolCall{
					ID:   tu.id,
					Type: "function",
					Function: services.FunctionCall{
						Name:      tu.name,
						Arguments: inputStr,
					},
				})

				delete(activeToolUses, event.Index)
			}

		case "message_stop":
			// End of message stream
		}
	}

	// Check if scanner error was due to cancellation
	if err := scanner.Err(); err != nil {
		if s.requestCtx.Err() == context.Canceled {
			return nil // Not an error, just interrupted
		}
		return err
	}

	// Add assistant response to context
	response := fullResponse.String()
	if len(completedToolCalls) > 0 {
		// Assistant message with tool calls
		msg := services.LLMMessage{
			Role:      "assistant",
			Content:   response,
			ToolCalls: completedToolCalls,
		}
		llmCtx.Messages = append(llmCtx.Messages, msg)
		s.log.Info("Assistant response with %d tool calls", len(completedToolCalls))
	} else if response != "" {
		llmCtx.AddAssistantMessage(response)
		s.log.Debug("Assistant: %s", response)
	}

	return nil
}
