package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
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
	ctx         context.Context
	cancel      context.CancelFunc
}

// LLMConfig holds configuration for OpenAI
type LLMConfig struct {
	APIKey       string
	Model        string  // e.g., "gpt-4-turbo", "gpt-3.5-turbo"
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
	log.Printf("[OpenAI] Initialized with model %s", s.model)
	return nil
}

func (s *LLMService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	return nil
}

func (s *LLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle LLMContextFrame (from aggregators) - PRIMARY MODE
	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		// Extract context from frame
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			log.Printf("[OpenAI] Received LLMContextFrame with %d messages", len(llmContext.Messages))

			// Update our context reference
			s.context = llmContext

			// Send LLM response start marker
			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			// Generate response using the provided context
			if err := s.generateResponseFromContext(llmContext); err != nil {
				log.Printf("[OpenAI] Error generating response: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}

			// Send LLM response end marker
			s.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
		}
		return nil
	}

	// Process transcription frames (user speech) - BACKWARD COMPATIBILITY
	if transcriptionFrame, ok := frame.(*frames.TranscriptionFrame); ok {
		if transcriptionFrame.IsFinal {
			// Add to context and generate response
			s.context.AddUserMessage(transcriptionFrame.Text)
			log.Printf("[OpenAI] User: %s", transcriptionFrame.Text)

			// Send LLM response start marker
			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			// Generate response
			if err := s.generateResponse(); err != nil {
				log.Printf("[OpenAI] Error generating response: %v", err)
				return s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}

			// Send LLM response end marker
			s.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
		}
		return nil
	}

	// Pass all other frames through
	return s.PushFrame(frame, direction)
}

func (s *LLMService) generateResponse() error {
	// Build messages array
	messages := []map[string]string{
		{
			"role":    "system",
			"content": s.context.SystemPrompt,
		},
	}

	for _, msg := range s.context.Messages {
		messages = append(messages, map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}

	// Prepare request
	requestBody := map[string]interface{}{
		"model":       s.model,
		"messages":    messages,
		"temperature": s.temperature,
		"stream":      true,
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", s.apiKey))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("OpenAI API error: %s", string(body))
	}

	// Stream response
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
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
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue
		}

		if len(streamResp.Choices) > 0 {
			content := streamResp.Choices[0].Delta.Content
			if content != "" {
				fullResponse.WriteString(content)
				// Send token as LLM text frame
				textFrame := frames.NewLLMTextFrame(content)
				s.PushFrame(textFrame, frames.Downstream)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// Add assistant response to context
	response := fullResponse.String()
	s.context.AddAssistantMessage(response)
	log.Printf("[OpenAI] Assistant: %s", response)

	return nil
}

// generateResponseFromContext generates a response using the provided context
// Supports full message format including tool calls
func (s *LLMService) generateResponseFromContext(ctx *services.LLMContext) error {
	// Build messages array from context
	messages := []map[string]interface{}{}

	// Add system prompt if present
	if ctx.SystemPrompt != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": ctx.SystemPrompt,
		})
	}

	// Add all messages from context
	for _, msg := range ctx.Messages {
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
	if len(ctx.Tools) > 0 {
		tools := []map[string]interface{}{}
		for _, tool := range ctx.Tools {
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
		if ctx.ToolChoice != nil {
			requestBody["tool_choice"] = ctx.ToolChoice
		}
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", s.apiKey))
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("OpenAI API error: %s", string(body))
	}

	// Stream response
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
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

		if len(streamResp.Choices) > 0 {
			content := streamResp.Choices[0].Delta.Content
			if content != "" {
				fullResponse.WriteString(content)
				// Send token as text frame (not LLMTextFrame - aggregator will handle it)
				textFrame := frames.NewTextFrame(content)
				s.PushFrame(textFrame, frames.Downstream)
			}

			// Handle tool calls (function calling)
			// Note: Full function call support would require accumulating tool_calls
			// and emitting FunctionCallInProgressFrame, but that's beyond current scope
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// Add assistant response to context
	response := fullResponse.String()
	if response != "" {
		ctx.AddAssistantMessage(response)
		log.Printf("[OpenAI] Assistant: %s", response)
	}

	return nil
}
