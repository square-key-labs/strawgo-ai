package groq

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestNewGroqLLMService(t *testing.T) {
	config := GroqLLMConfig{
		APIKey:       "test-api-key",
		Model:        "llama-3.3-70b-versatile",
		SystemPrompt: "You are a helpful assistant",
		Temperature:  0.7,
	}

	service := NewGroqLLMService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.model != "llama-3.3-70b-versatile" {
		t.Errorf("Expected model llama-3.3-70b-versatile, got %s", service.model)
	}

	if service.baseURL != DefaultGroqBaseURL {
		t.Errorf("Expected Groq base URL %s, got %s", DefaultGroqBaseURL, service.baseURL)
	}

	if service.temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", service.temperature)
	}

	if service.apiKey != "test-api-key" {
		t.Errorf("Expected API key test-api-key, got %s", service.apiKey)
	}
}

func TestNewGroqLLMServiceDefaults(t *testing.T) {
	config := GroqLLMConfig{
		APIKey: "test-api-key",
	}

	service := NewGroqLLMService(config)

	if service.model != DefaultGroqModel {
		t.Errorf("Expected default model %s, got %s", DefaultGroqModel, service.model)
	}

	if service.baseURL != DefaultGroqBaseURL {
		t.Errorf("Expected default base URL %s, got %s", DefaultGroqBaseURL, service.baseURL)
	}
}

func TestGroqLLMServiceBaseURL(t *testing.T) {
	customURL := "https://custom.groq.com/v1"
	config := GroqLLMConfig{
		APIKey:  "test-api-key",
		BaseURL: customURL,
	}

	service := NewGroqLLMService(config)

	if service.baseURL != customURL {
		t.Errorf("Expected custom base URL %s, got %s", customURL, service.baseURL)
	}
}

func TestGroqLLMServiceConfiguration(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	service.SetModel("mixtral-8x7b-32768")
	if service.model != "mixtral-8x7b-32768" {
		t.Errorf("Expected model mixtral-8x7b-32768, got %s", service.model)
	}

	service.SetSystemPrompt("New system prompt")
	if service.context.SystemPrompt != "New system prompt" {
		t.Errorf("Expected system prompt 'New system prompt', got %s", service.context.SystemPrompt)
	}

	service.SetTemperature(0.9)
	if service.temperature != 0.9 {
		t.Errorf("Expected temperature 0.9, got %f", service.temperature)
	}
}

func TestGroqLLMServiceMessageManagement(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	service.AddMessage("user", "Hello")
	service.AddMessage("assistant", "Hi there")

	if len(service.context.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(service.context.Messages))
	}

	if service.context.Messages[0].Role != "user" {
		t.Errorf("Expected first message role 'user', got %s", service.context.Messages[0].Role)
	}

	if service.context.Messages[0].Content != "Hello" {
		t.Errorf("Expected first message content 'Hello', got %s", service.context.Messages[0].Content)
	}

	service.ClearContext()
	if len(service.context.Messages) != 0 {
		t.Errorf("Expected 0 messages after clear, got %d", len(service.context.Messages))
	}
}

func TestGroqLLMServiceInitializeCleanup(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	err := service.Initialize(ctx)
	if err != nil {
		t.Errorf("Initialize failed: %v", err)
	}

	if service.ctx == nil {
		t.Error("Expected context to be set after Initialize")
	}

	if service.cancel == nil {
		t.Error("Expected cancel function to be set after Initialize")
	}

	err = service.Cleanup()
	if err != nil {
		t.Errorf("Cleanup failed: %v", err)
	}
}

func TestGroqLLMServiceFrameLifecycle(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	startFrame := frames.NewStartFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Errorf("StartFrame handling failed: %v", err)
	}

	endFrame := frames.NewEndFrame()
	err = service.HandleFrame(ctx, endFrame, frames.Downstream)
	if err != nil {
		t.Errorf("EndFrame handling failed: %v", err)
	}
}

func TestGroqLLMServiceInterruptionHandling(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	service.streamMu.Lock()
	service.isGenerating = true
	service.requestCtx, service.requestCancel = context.WithCancel(ctx)
	service.streamMu.Unlock()

	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	service.streamMu.Lock()
	wasGenerating := service.isGenerating
	service.streamMu.Unlock()

	if wasGenerating {
		t.Error("Expected isGenerating to be false after interruption")
	}
}

func TestGroqLLMServiceLLMContextFrame(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("You are a helpful assistant")
	llmContext.AddUserMessage("Hello")

	contextFrame := frames.NewLLMContextFrame(llmContext)

	err := service.HandleFrame(ctx, contextFrame, frames.Downstream)
	if err != nil {
		t.Errorf("LLMContextFrame handling failed: %v", err)
	}

	if len(service.context.Messages) != 1 {
		t.Errorf("Expected 1 message in context, got %d", len(service.context.Messages))
	}
}

func TestGroqLLMServicePassthroughFrames(t *testing.T) {
	service := NewGroqLLMService(GroqLLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	textFrame := frames.NewTextFrame("test")
	err := service.HandleFrame(ctx, textFrame, frames.Downstream)
	if err != nil {
		t.Errorf("TextFrame passthrough failed: %v", err)
	}

	audioFrame := frames.NewAudioFrame([]byte{0x00, 0x01}, 16000, 1)
	err = service.HandleFrame(ctx, audioFrame, frames.Downstream)
	if err != nil {
		t.Errorf("AudioFrame passthrough failed: %v", err)
	}
}
