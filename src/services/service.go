package services

import (
	"context"

	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// AIService is the base interface for all AI services (STT, TTS, LLM)
type AIService interface {
	processors.FrameProcessor

	// Service lifecycle
	Initialize(ctx context.Context) error
	Cleanup() error
}

// STTService converts speech to text
type STTService interface {
	AIService

	// Configuration
	SetLanguage(lang string)
	SetModel(model string)
}

// TTSService converts text to speech
type TTSService interface {
	AIService

	// Configuration
	SetVoice(voiceID string)
	SetModel(model string)
}

// LLMService provides language model capabilities
type LLMService interface {
	AIService

	// Configuration
	SetModel(model string)
	SetSystemPrompt(prompt string)
	SetTemperature(temp float64)

	// Context management
	AddMessage(role, content string)
	ClearContext()
}

// LLMMessage represents a message in the conversation
type LLMMessage struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// LLMContext holds the conversation context
type LLMContext struct {
	Messages    []LLMMessage
	SystemPrompt string
	Model       string
	Temperature float64
}

// NewLLMContext creates a new LLM context
func NewLLMContext(systemPrompt string) *LLMContext {
	return &LLMContext{
		Messages:    make([]LLMMessage, 0),
		SystemPrompt: systemPrompt,
		Temperature: 0.7,
	}
}

func (c *LLMContext) AddUserMessage(content string) {
	c.Messages = append(c.Messages, LLMMessage{
		Role:    "user",
		Content: content,
	})
}

func (c *LLMContext) AddAssistantMessage(content string) {
	c.Messages = append(c.Messages, LLMMessage{
		Role:    "assistant",
		Content: content,
	})
}

func (c *LLMContext) AddSystemMessage(content string) {
	c.Messages = append(c.Messages, LLMMessage{
		Role:    "system",
		Content: content,
	})
}

func (c *LLMContext) Clear() {
	c.Messages = make([]LLMMessage, 0)
}
