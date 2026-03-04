package ollama

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
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// OllamaLLMService provides language model capabilities using Ollama's OpenAI-compatible API
// Ollama is a local model hosting service that requires no authentication
type OllamaLLMService struct {
	*processors.BaseProcessor
	baseURL     string
	model       string
	temperature float64
	context     *services.LLMContext
	ctx         context.Context
	cancel      context.CancelFunc

	// Request-scoped context for cancellable streaming (protected by streamMu)
	requestCtx    context.Context
	requestCancel context.CancelFunc
	isGenerating  bool
	lastContextAt time.Time  // When we last received a new context (for interruption filtering)
	streamMu      sync.Mutex // Protects requestCancel, isGenerating, and lastContextAt
}

// OllamaLLMConfig holds configuration for Ollama
type OllamaLLMConfig struct {
	Model        string // e.g., "llama3.2", "mistral", "codellama"
	SystemPrompt string
	Temperature  float64
	BaseURL      string // Optional: override default Ollama URL (default: http://localhost:11434)
}

const (
	// DefaultOllamaBaseURL is the default Ollama API endpoint (OpenAI-compatible)
	DefaultOllamaBaseURL = "http://localhost:11434/v1"
	// DefaultOllamaModel is the default Ollama model
	DefaultOllamaModel = "llama3.2"
)

// NewOllamaLLMService creates a new Ollama LLM service
func NewOllamaLLMService(config OllamaLLMConfig) *OllamaLLMService {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultOllamaBaseURL
	}

	model := config.Model
	if model == "" {
		model = DefaultOllamaModel
	}

	os := &OllamaLLMService{
		baseURL:     baseURL,
		model:       model,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
	}
	os.BaseProcessor = processors.NewBaseProcessor("Ollama", os)
	return os
}

func (s *OllamaLLMService) SetModel(model string) {
	s.model = model
}

func (s *OllamaLLMService) SetSystemPrompt(prompt string) {
	s.context.SystemPrompt = prompt
}

func (s *OllamaLLMService) SetTemperature(temp float64) {
	s.temperature = temp
}

func (s *OllamaLLMService) AddMessage(role, content string) {
	s.context.Messages = append(s.context.Messages, services.LLMMessage{
		Role:    role,
		Content: content,
	})
}

func (s *OllamaLLMService) ClearContext() {
	s.context.Clear()
}

func (s *OllamaLLMService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)
	log.Printf("[Ollama] Initialized with model %s", s.model)
	return nil
}

func (s *OllamaLLMService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	return nil
}

func (s *OllamaLLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle InterruptionFrame - CRITICAL: Stop streaming immediately
	// BUT: If we just received a new context (within 100ms), this interruption is for
	// the OLD response, not the new one. Don't cancel in that case.
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.streamMu.Lock()
		timeSinceContext := time.Since(s.lastContextAt)
		isNewContext := timeSinceContext < 100*time.Millisecond

		log.Printf("[Ollama] INTERRUPTION received (isGenerating=%v, timeSinceContext=%v)", s.isGenerating, timeSinceContext)

		if isNewContext {
			// This interruption is for the OLD response, not the new one we just started
			log.Printf("[Ollama] Ignoring interruption - new context was just received (%v ago)", timeSinceContext)
			s.streamMu.Unlock()
			return s.PushFrame(frame, direction)
		}

		if s.isGenerating && s.requestCancel != nil {
			log.Printf("[Ollama] CANCELLING ongoing stream")
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
			log.Printf("[Ollama] Received LLMContextFrame with %d messages", len(llmContext.Messages))

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
					log.Printf("[Ollama] Stream cancelled by interruption")
				} else {
					log.Printf("[Ollama] Error generating response: %v", err)
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
func (s *OllamaLLMService) generateResponseFromContext(llmCtx *services.LLMContext) error {
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
	// Ollama OpenAI-compatible endpoint
	req, err := http.NewRequestWithContext(s.requestCtx, "POST", s.baseURL+"/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	// No Authorization header needed for Ollama (local service)
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
		return fmt.Errorf("Ollama API error: %s", string(body))
	}

	// Stream response
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		// Check if interrupted
		select {
		case <-s.requestCtx.Done():
			log.Printf("[Ollama] Stream interrupted, stopping generation")
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

		if len(streamResp.Choices) > 0 {
			content := streamResp.Choices[0].Delta.Content
			if content != "" {
				fullResponse.WriteString(content)
				// Emit raw LLMTextFrame - sentence splitting handled by SentenceAggregator
				textFrame := frames.NewLLMTextFrame(content)
				s.PushFrame(textFrame, frames.Downstream)
			}

			// Handle tool calls (function calling)
			// Note: Full function call support would require accumulating tool_calls
			// and emitting FunctionCallInProgressFrame, but that's beyond current scope
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
	if response != "" {
		llmCtx.AddAssistantMessage(response)
		log.Printf("[Ollama] Assistant: %s", response)
	}

	return nil
}
