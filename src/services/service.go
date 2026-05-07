package services

import (
	"context"
	"fmt"
	"sync"

	"github.com/google/uuid"
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
	Role       string // "system", "user", "assistant", "tool"
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
		Messages:     make([]LLMMessage, 0),
		SystemPrompt: systemPrompt,
		Temperature:  0.7,
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

// LargeValueThreshold is the byte length above which a string content field is
// considered a "large value" (e.g. base64 image data, binary blobs) and will
// be replaced with a [N bytes] placeholder when GetMessages is called with
// truncateLargeValues=true.
const LargeValueThreshold = 1024

// GetMessages returns the conversation messages. When truncateLargeValues is
// true, any message Content or tool-call Arguments string longer than
// LargeValueThreshold bytes is replaced with a "[N bytes]" placeholder.
// This prevents large binary payloads (base64 images, file data) from bloating
// LLM API requests or debug logs while preserving the message structure.
func (c *LLMContext) GetMessages(truncateLargeValues bool) []LLMMessage {
	if !truncateLargeValues {
		return c.Messages
	}

	truncate := func(s string) string {
		if len(s) > LargeValueThreshold {
			return fmt.Sprintf("[%d bytes]", len(s))
		}
		return s
	}

	out := make([]LLMMessage, len(c.Messages))
	for i, m := range c.Messages {
		msg := m // copy
		msg.Content = truncate(m.Content)
		if len(m.ToolCalls) > 0 {
			calls := make([]ToolCall, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				calls[j] = tc
				calls[j].Function.Arguments = truncate(tc.Function.Arguments)
			}
			msg.ToolCalls = calls
		}
		out[i] = msg
	}
	return out
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

// GenerateContextID generates a unique context ID identifying a single TTS
// Generation (synthesis pass). The ID is stamped on TTSStartedFrame,
// TTSAudioFrame, and TTSStoppedFrame so the AudioGuard processor can drop
// stale audio after an interruption.
//
// See docs/adr/0001-audio-guard-as-dedicated-processor.md for the architectural
// decision and CONTEXT.md for the "Generation" glossary entry.
//
// Returns a UUID v4 string (e.g., "550e8400-e29b-41d4-a716-446655440000").
func GenerateContextID() string {
	return uuid.New().String()
}

// AudioContextManager mints and tracks TTS Generation context IDs for streaming
// services. It provides thread-safe per-turn ID minting and reuse within a
// single LLM turn so multiple TTS invocations share one Generation.
//
// Stale-audio filtering is NOT done here — that lives in the AudioGuard
// processor at the pipeline level (see
// docs/adr/0001-audio-guard-as-dedicated-processor.md). TTS services consume
// this manager to:
//   - Mint a turn-scoped context_id on LLMFullResponseStartFrame
//   - Stamp outgoing TTSAudioFrame metadata with that context_id
//   - Pass it to upstream cancel APIs (e.g., Cartesia "cancel", ElevenLabs
//     "close_context") on InterruptionFrame
//
// Embed *AudioContextManager in TTS service structs to gain shared context
// management behavior. The methods are promoted to the embedding struct.
type AudioContextManager struct {
	contextID            string
	currentTurnContextID string
	mu                   sync.Mutex

	// ReuseContextIDWithinTurn controls whether the same context ID is reused
	// across multiple TTS invocations within a single LLM turn.
	// Default: true.
	ReuseContextIDWithinTurn bool
}

// NewAudioContextManager creates a new AudioContextManager with default settings.
// ReuseContextIDWithinTurn defaults to true.
func NewAudioContextManager() *AudioContextManager {
	return &AudioContextManager{
		ReuseContextIDWithinTurn: true,
	}
}

// HasActiveAudioContext returns true if a context ID is currently active.
func (m *AudioContextManager) HasActiveAudioContext() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.contextID != ""
}

// GetActiveAudioContextID returns the current active context ID.
// Returns empty string if no context is active.
func (m *AudioContextManager) GetActiveAudioContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.contextID
}

// SetActiveAudioContextID sets the active context ID directly.
// Used during service initialization or when explicitly setting a context.
func (m *AudioContextManager) SetActiveAudioContextID(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.contextID = id
}

// ResetActiveAudioContext clears both contextID and currentTurnContextID.
// Called on interruption or normal turn completion.
func (m *AudioContextManager) ResetActiveAudioContext() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.contextID = ""
	m.currentTurnContextID = ""
}

// GetTurnContextID returns the current turn context ID (may be empty).
func (m *AudioContextManager) GetTurnContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.currentTurnContextID
}

// GetOrCreateTurnContextID returns the current turn context ID if set,
// otherwise generates a new one via GenerateContextID() and stores it.
func (m *AudioContextManager) GetOrCreateTurnContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.currentTurnContextID == "" {
		m.currentTurnContextID = GenerateContextID()
	}
	return m.currentTurnContextID
}

// GetOrCreateContextID returns the current contextID. If empty:
//   - If ReuseContextIDWithinTurn is true and a turn context ID exists, reuses it
//   - Otherwise generates a new context ID via GenerateContextID()
func (m *AudioContextManager) GetOrCreateContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.contextID == "" {
		if m.ReuseContextIDWithinTurn && m.currentTurnContextID != "" {
			m.contextID = m.currentTurnContextID
		} else {
			m.contextID = GenerateContextID()
		}
	}
	return m.contextID
}
