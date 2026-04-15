package openai

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

// LLMService provides language model capabilities using OpenAI
type LLMService struct {
	*processors.BaseProcessor
	apiKey      string
	model       string
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

// LLMConfig holds configuration for OpenAI
type LLMConfig struct {
	APIKey       string
	Model        string // e.g., "gpt-4-turbo", "gpt-3.5-turbo"
	SystemPrompt string
	Temperature  float64
}

// NewLLMService creates a new OpenAI LLM service
func NewLLMService(config LLMConfig) *LLMService {
	os := &LLMService{
		apiKey:      config.APIKey,
		model:       config.Model,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
		log:         logger.WithPrefix("OpenAILLM"),
	}
	os.BaseProcessor = processors.NewBaseProcessor("OpenAI", os)
	return os
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
		// Extract context from frame
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

// generateResponseFromContext generates a response using the provided context
// Supports full message format including tool calls
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

	// Build messages array from context
	messages := []map[string]interface{}{}

	// Add system prompt if present
	if llmCtx.SystemPrompt != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": llmCtx.SystemPrompt,
		})
	}

	// Add all messages from context
	for _, msg := range llmCtx.Messages {
		message := map[string]interface{}{
			"role": msg.Role,
		}

		// Add content if present
		if msg.Content != "" {
			message["content"] = msg.Content
		}

		// Add tool calls if present (assistant messages)
		if len(msg.ToolCalls) > 0 {
			toolCalls := []map[string]interface{}{}
			for _, tc := range msg.ToolCalls {
				toolCalls = append(toolCalls, map[string]interface{}{
					"id":   tc.ID,
					"type": tc.Type,
					"function": map[string]interface{}{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments,
					},
				})
			}
			message["tool_calls"] = toolCalls
		}

		// Add tool_call_id if present (tool messages)
		if msg.ToolCallID != "" {
			message["tool_call_id"] = msg.ToolCallID
		}

		messages = append(messages, message)
	}

	// Prepare request
	requestBody := map[string]interface{}{
		"model":       s.model,
		"messages":    messages,
		"temperature": s.temperature,
		"stream":      true,
	}

	// Add tools if present in context
	if len(llmCtx.Tools) > 0 {
		tools := []map[string]interface{}{}
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

		// Add tool_choice if specified
		if llmCtx.ToolChoice != nil {
			requestBody["tool_choice"] = llmCtx.ToolChoice
		}
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	// Use cancellable context so interruption can stop the request
	req, err := http.NewRequestWithContext(s.requestCtx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", s.apiKey))
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
		return fmt.Errorf("OpenAI API error: %s", string(body))
	}

	// Stream response
	var fullResponse strings.Builder

	// partialToolCall accumulates streamed fragments of a single function call.
	type partialToolCall struct {
		id        string
		callType  string
		name      string
		arguments strings.Builder
	}
	partialCalls := map[int]*partialToolCall{}
	maxIdx := -1

	scanner := bufio.NewScanner(resp.Body)

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
		if data == "[DONE]" {
			break
		}

		var streamResp struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue
		}

		if len(streamResp.Choices) == 0 {
			continue
		}

		delta := streamResp.Choices[0].Delta

		if delta.Content != "" {
			fullResponse.WriteString(delta.Content)
			// Emit raw LLMTextFrame — sentence splitting handled by SentenceAggregator
			s.PushFrame(frames.NewLLMTextFrame(delta.Content), frames.Downstream)
		}

		// Accumulate streaming tool-call deltas (function.arguments arrive in fragments).
		for _, tcDelta := range delta.ToolCalls {
			idx := tcDelta.Index
			if _, ok := partialCalls[idx]; !ok {
				partialCalls[idx] = &partialToolCall{}
			}
			if idx > maxIdx {
				maxIdx = idx
			}
			pt := partialCalls[idx]
			if tcDelta.ID != "" {
				pt.id = tcDelta.ID
			}
			if tcDelta.Type != "" {
				pt.callType = tcDelta.Type
			}
			if tcDelta.Function.Name != "" {
				pt.name = tcDelta.Function.Name
			}
			pt.arguments.WriteString(tcDelta.Function.Arguments)
		}
	}

	// Check if scanner error was due to cancellation
	if err := scanner.Err(); err != nil {
		if s.requestCtx.Err() == context.Canceled {
			return nil // Not an error, just interrupted
		}
		return err
	}

	// Emit accumulated tool calls as frames and record in context.
	if len(partialCalls) > 0 {
		callInfos := make([]frames.FunctionCallInfo, 0, len(partialCalls))
		completedCalls := make([]services.ToolCall, 0, len(partialCalls))

		for i := 0; i <= maxIdx; i++ {
			pt, ok := partialCalls[i]
			if !ok {
				continue
			}
			callType := pt.callType
			if callType == "" {
				callType = "function"
			}
			argStr := pt.arguments.String()
			callInfos = append(callInfos, frames.FunctionCallInfo{
				ToolCallID:   pt.id,
				FunctionName: pt.name,
			})
			completedCalls = append(completedCalls, services.ToolCall{
				ID:   pt.id,
				Type: callType,
				Function: services.FunctionCall{
					Name:      pt.name,
					Arguments: argStr,
				},
			})
		}

		s.PushFrame(frames.NewFunctionCallsStartedFrame(callInfos), frames.Downstream)

		// Emit FunctionCallInProgressFrame in index order (map iteration is unordered).
		for i := 0; i <= maxIdx; i++ {
			pt, ok := partialCalls[i]
			if !ok {
				continue
			}
			var args map[string]interface{}
			argStr := pt.arguments.String()
			if argStr != "" {
				if err := json.Unmarshal([]byte(argStr), &args); err != nil {
					args = map[string]interface{}{}
				}
			} else {
				args = map[string]interface{}{}
			}
			s.PushFrame(frames.NewFunctionCallInProgressFrame(pt.id, pt.name, args, true), frames.Downstream)
			s.log.Debug("Tool call: %s(%s)", pt.name, argStr)
		}

		llmCtx.AddMessageWithToolCalls(completedCalls)
		s.log.Debug("Emitted %d tool call(s)", len(completedCalls))
		return nil
	}

	// Add text assistant response to context
	response := fullResponse.String()
	if response != "" {
		llmCtx.AddAssistantMessage(response)
		s.log.Debug("Assistant: %s", response)
	}

	return nil
}
