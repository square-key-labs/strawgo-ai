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
	Role       string     // "system", "user", "assistant", "tool"
	Content    string
	ToolCalls  []ToolCall // For assistant messages with function calls
	ToolCallID string     // For tool response messages
}

// ToolCall represents a function call made by the LLM
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall represents the function and its arguments
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// Tool represents an available tool/function
type Tool struct {
	Type     string       `json:"type"` // "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a function available to the LLM
type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"` // JSON schema
}

// LLMContext holds the conversation context
type LLMContext struct {
	Messages     []LLMMessage
	SystemPrompt string
	Model        string
	Temperature  float64
	Tools        []Tool      // Available tools/functions
	ToolChoice   interface{} // "auto", "none", "required", or specific function
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

// AddMessageWithToolCalls adds an assistant message with function calls
func (c *LLMContext) AddMessageWithToolCalls(toolCalls []ToolCall) {
	c.Messages = append(c.Messages, LLMMessage{
		Role:      "assistant",
		ToolCalls: toolCalls,
	})
}

// AddToolMessage adds a tool/function response message
func (c *LLMContext) AddToolMessage(toolCallID, content string) {
	c.Messages = append(c.Messages, LLMMessage{
		Role:       "tool",
		Content:    content,
		ToolCallID: toolCallID,
	})
}

// SetTools sets the available tools/functions
func (c *LLMContext) SetTools(tools []Tool) {
	c.Tools = tools
}

// SetToolChoice sets the tool choice strategy
func (c *LLMContext) SetToolChoice(choice interface{}) {
	c.ToolChoice = choice
}

// Clone creates a deep copy of the context
func (c *LLMContext) Clone() *LLMContext {
	clone := &LLMContext{
		SystemPrompt: c.SystemPrompt,
		Model:        c.Model,
		Temperature:  c.Temperature,
		ToolChoice:   c.ToolChoice,
		Messages:     make([]LLMMessage, len(c.Messages)),
		Tools:        make([]Tool, len(c.Tools)),
	}
	copy(clone.Messages, c.Messages)
	copy(clone.Tools, c.Tools)
	return clone
}
