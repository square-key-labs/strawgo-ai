package gemini

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

// LLMService provides language model capabilities using Google Gemini
type LLMService struct {
	*processors.BaseProcessor
	apiKey      string
	model       string
	temperature float64
	context     *services.LLMContext
	ctx         context.Context
	cancel      context.CancelFunc
}

// LLMConfig holds configuration for Gemini
type LLMConfig struct {
	APIKey       string
	Model        string // e.g., "gemini-1.5-pro", "gemini-1.5-flash"
	SystemPrompt string
	Temperature  float64
}

// NewLLMService creates a new Gemini LLM service
func NewLLMService(config LLMConfig) *LLMService {
	gs := &LLMService{
		apiKey:      config.APIKey,
		model:       config.Model,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
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
	log.Printf("[Gemini] Initialized with model %s", s.model)
	return nil
}

func (s *LLMService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	return nil
}

func (s *LLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle LLMContextFrame (from aggregators)
	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		// Extract context from frame
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			log.Printf("[Gemini] Received LLMContextFrame with %d messages", len(llmContext.Messages))

			// Update our context reference
			s.context = llmContext

			// Send LLM response start marker
			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			// Generate response using the provided context
			if err := s.generateResponse(); err != nil {
				log.Printf("[Gemini] Error generating response: %v", err)
				s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
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
	// Build contents array (Gemini format)
	contents := []map[string]interface{}{}

	// Add system instruction in first user message if available
	if s.context.SystemPrompt != "" && len(s.context.Messages) == 1 {
		contents = append(contents, map[string]interface{}{
			"role": "user",
			"parts": []map[string]string{
				{"text": s.context.SystemPrompt + "\n\n" + s.context.Messages[0].Content},
			},
		})
	} else {
		for _, msg := range s.context.Messages {
			role := msg.Role
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

	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
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

	if err := scanner.Err(); err != nil {
		return err
	}

	// Add assistant response to context
	response := fullResponse.String()
	s.context.AddAssistantMessage(response)
	// log.Printf("[Gemini] Assistant: %s", response)
	log.Printf("[Gemini] Assistant Response length: %d", len(response))

	return nil
}
