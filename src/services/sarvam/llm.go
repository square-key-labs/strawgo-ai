// Package sarvam provides Sarvam AI services. Today the package shipped
// only an STT implementation (stt.go); this file adds the chat-completion
// LLM. Sarvam exposes an OpenAI-compatible /chat/completions endpoint
// under https://api.sarvam.ai/v1, so the wire shape mirrors the Groq
// adapter — the differences are model defaults, base URL, and the
// Sarvam-specific reply-language steering header behavior (left to the
// caller via SystemPrompt for now).
package sarvam

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

// LLMService provides Sarvam chat-completion LLM access via the
// OpenAI-compatible Sarvam API. Streaming is server-sent events.
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

	// Request-scoped context for cancellable streaming (protected by streamMu).
	requestCtx    context.Context
	requestCancel context.CancelFunc
	isGenerating  bool
	lastContextAt time.Time
	streamMu      sync.Mutex

	toolWarnOnce sync.Once
}

// LLMConfig configures a Sarvam LLM service.
type LLMConfig struct {
	APIKey       string
	Model        string // e.g. "sarvam-m", "sarvam-30b"
	SystemPrompt string
	Temperature  float64
	BaseURL      string // Optional override; defaults to DefaultSarvamBaseURL
}

const (
	// DefaultSarvamBaseURL is the public Sarvam OpenAI-compatible API endpoint.
	DefaultSarvamBaseURL = "https://api.sarvam.ai/v1"
	// DefaultSarvamModel is the default Sarvam chat model.
	DefaultSarvamModel = "sarvam-m"
)

// NewLLMService creates a new Sarvam LLM service.
func NewLLMService(config LLMConfig) *LLMService {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultSarvamBaseURL
	}

	model := config.Model
	if model == "" {
		model = DefaultSarvamModel
	}

	s := &LLMService{
		apiKey:      config.APIKey,
		baseURL:     baseURL,
		model:       model,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
		log:         logger.WithPrefix("SarvamLLM"),
	}
	s.BaseProcessor = processors.NewBaseProcessor("Sarvam", s)
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

func (s *LLMService) ClearContext() { s.context.Clear() }

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
	// InterruptionFrame — cancel an in-flight stream unless the cancellation
	// is for an OLD response (we just received a new context within 100ms).
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

	// LLMContextFrame from the user aggregator triggers a generation.
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

// generateResponseFromContext streams a chat-completion response from the
// Sarvam endpoint. Streaming is OpenAI-style SSE (`data: {json}\n\n`).
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

	messages := []map[string]interface{}{}

	if llmCtx.SystemPrompt != "" {
		messages = append(messages, map[string]interface{}{
			"role":    "system",
			"content": llmCtx.SystemPrompt,
		})
	}

	for _, msg := range llmCtx.Messages {
		// Sarvam mirrors the OpenAI chat shape but does not accept the
		// "developer" role. Map it to "user" so async-tool-result
		// developer-injected messages from the aggregator still flow
		// through.
		role := msg.Role
		if role == "developer" {
			role = "user"
		}
		message := map[string]interface{}{"role": role}
		if msg.Content != "" {
			message["content"] = msg.Content
		}
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
		if msg.ToolCallID != "" {
			message["tool_call_id"] = msg.ToolCallID
		}
		messages = append(messages, message)
	}

	requestBody := map[string]interface{}{
		"model":       s.model,
		"messages":    messages,
		"temperature": s.temperature,
		"stream":      true,
	}

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
		if llmCtx.ToolChoice != nil {
			requestBody["tool_choice"] = llmCtx.ToolChoice
		}
		// Tools are forwarded to Sarvam so the model can call them, but
		// this adapter does not yet parse streamed tool_calls deltas
		// into FunctionCallInProgressFrame / FunctionCallResultFrame.
		// Tool-call output will arrive as text. Matches the existing
		// Groq adapter's gap (see src/services/groq/llm.go); both wait
		// on full tool-flow integration to land.
		s.toolWarnOnce.Do(func() {
			s.log.Warn("Tools forwarded but tool-call SSE deltas are not parsed; function calls will arrive as text.")
		})
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(s.requestCtx, "POST", s.baseURL+"/chat/completions", bytes.NewReader(bodyBytes))
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
		return fmt.Errorf("Sarvam API error: %s", string(body))
	}

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	// Default 64KB token can clip extra-long SSE lines (image-rich
	// payloads, large tool_calls deltas). Raise to 1MB to match the
	// OpenAI Responses adapter.
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
		if len(streamResp.Choices) == 0 {
			continue
		}
		content := streamResp.Choices[0].Delta.Content
		if content == "" {
			continue
		}
		fullResponse.WriteString(content)
		s.PushFrame(frames.NewLLMTextFrame(content), frames.Downstream)
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

	return nil
}
