package gemini

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

// LLMService provides language model capabilities using Google Gemini
type LLMService struct {
	*processors.BaseProcessor
	apiKey            string
	model             string
	temperature       float64
	systemInstruction string
	context           *services.LLMContext
	ctx               context.Context
	cancel            context.CancelFunc

	// Request-scoped context for cancellable streaming (protected by streamMu)
	requestCtx    context.Context
	requestCancel context.CancelFunc
	isGenerating  bool
	lastContextAt time.Time  // When we last received a new context (for interruption filtering)
	streamMu      sync.Mutex // Protects requestCancel, isGenerating, and lastContextAt
	log           *logger.Logger
}

// LLMConfig holds configuration for Gemini
type LLMConfig struct {
	APIKey       string
	Model        string // e.g., "gemini-1.5-pro", "gemini-1.5-flash"
	SystemPrompt string
	// SystemInstruction, when set, takes precedence over any context-level
	// system prompt. Mirrors pipecat PR #3918 / #3932. Warning logged if both set.
	SystemInstruction string
	Temperature       float64
}

// NewLLMService creates a new Gemini LLM service
func NewLLMService(config LLMConfig) *LLMService {
	gs := &LLMService{
		apiKey:            config.APIKey,
		model:             config.Model,
		temperature:       config.Temperature,
		systemInstruction: config.SystemInstruction,
		context:           services.NewLLMContext(config.SystemPrompt),
		log:               logger.WithPrefix("GeminiLLM"),
	}
	gs.BaseProcessor = processors.NewBaseProcessor("Gemini", gs)
	return gs
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
		isGen := s.isGenerating
		hasCancel := s.requestCancel != nil
		timeSinceContext := time.Since(s.lastContextAt)
		isNewContext := timeSinceContext < 100*time.Millisecond

		s.log.Debug("Interruption received (isGenerating=%v, hasCancel=%v, timeSinceContext=%v)", isGen, hasCancel, timeSinceContext)

		if isNewContext {
			// This interruption is for the OLD response, not the new one we just started
			s.log.Debug("Ignoring interruption - new context was just received (%v ago)", timeSinceContext)
			s.streamMu.Unlock()
			return s.PushFrame(frame, direction)
		}

		if isGen && hasCancel {
			s.log.Info("Cancelling ongoing stream")
			s.requestCancel()
			s.isGenerating = false
			s.log.Info("Stream cancelled, isGenerating=false")
		} else {
			s.log.Debug("No active stream to cancel")
		}
		s.streamMu.Unlock()
		return s.PushFrame(frame, direction)
	}

	// Handle LLMContextFrame (from aggregators)
	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		// Extract context from frame
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			s.log.Info("Received LLMContextFrame with %d messages", len(llmContext.Messages))

			// Record when we received this context (for interruption filtering)
			s.streamMu.Lock()
			s.lastContextAt = time.Now()
			s.streamMu.Unlock()

			// Update our context reference
			s.context = llmContext

			// Send LLM response start marker
			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			// Generate response using the provided context
			if err := s.generateResponse(); err != nil {
				// Only log error if not cancelled
				if s.requestCtx != nil && s.requestCtx.Err() == context.Canceled {
					s.log.Info("Stream cancelled by interruption")
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

func (s *LLMService) generateResponse() error {
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

	s.log.Info("Starting stream generation (isGenerating=true)")
	defer func() {
		s.streamMu.Lock()
		wasGenerating := s.isGenerating
		s.isGenerating = false
		if s.requestCancel != nil {
			s.requestCancel()
		}
		s.requestCancel = nil
		s.streamMu.Unlock()
		s.log.Info("Stream generation ended (wasGenerating=%v)", wasGenerating)
	}()

	// Build contents array (Gemini format)
	contents := []map[string]interface{}{}

	// Resolve effective system prompt: SystemInstruction (service-level) wins.
	effectiveSystem := s.context.SystemPrompt
	if s.systemInstruction != "" {
		if s.context.SystemPrompt != "" {
			s.log.Warn("Both SystemInstruction and LLMContext.SystemPrompt are set; SystemInstruction wins")
		}
		effectiveSystem = s.systemInstruction
	}

	// Add system instruction in first user message if available
	if effectiveSystem != "" && len(s.context.Messages) == 1 {
		contents = append(contents, map[string]interface{}{
			"role": "user",
			"parts": []map[string]string{
				{"text": effectiveSystem + "\n\n" + s.context.Messages[0].Content},
			},
		})
	} else {
		for _, msg := range s.context.Messages {
			role := msg.Role
			if role == "developer" {
				role = "user" // Gemini does not support the "developer" role
			}
			if role == "assistant" {
				role = "model" // Gemini uses "model" instead of "assistant"
			}
			if role == "system" {
				continue // Skip system messages (handled differently)
			}

			contents = append(contents, map[string]interface{}{
				"role": role,
				"parts": []map[string]string{
					{"text": msg.Content},
				},
			})
		}
	}

	// Prepare request
	requestBody := map[string]interface{}{
		"contents": contents,
		"generationConfig": map[string]interface{}{
			"temperature": s.temperature,
		},
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?key=%s&alt=sse",
		s.model, s.apiKey)

	// Use cancellable context so interruption can stop the request
	req, err := http.NewRequestWithContext(s.requestCtx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

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
		return fmt.Errorf("gemini API error: %s", string(body))
	}

	// Stream response (SSE format)
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		// Check if interrupted
		select {
		case <-s.requestCtx.Done():
			s.log.Info("Stream interrupted mid-generation, stopping immediately (tokens so far: %d chars)", fullResponse.Len())
			return nil
		default:
		}

		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var streamResp struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}

		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue
		}

		if len(streamResp.Candidates) > 0 && len(streamResp.Candidates[0].Content.Parts) > 0 {
			content := streamResp.Candidates[0].Content.Parts[0].Text
			if content != "" {
				fullResponse.WriteString(content)
				// Send token as LLM text frame
				textFrame := frames.NewLLMTextFrame(content)
				s.PushFrame(textFrame, frames.Downstream)
			}
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
	s.context.AddAssistantMessage(response)
	s.log.Debug("Assistant response length: %d", len(response))

	return nil
}
