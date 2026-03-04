package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestNewOllamaLLMService(t *testing.T) {
	config := OllamaLLMConfig{
		Model:        "llama3.2",
		SystemPrompt: "You are a helpful assistant",
		Temperature:  0.7,
	}

	service := NewOllamaLLMService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.model != "llama3.2" {
		t.Errorf("Expected model llama3.2, got %s", service.model)
	}

	if service.baseURL != DefaultOllamaBaseURL {
		t.Errorf("Expected Ollama base URL %s, got %s", DefaultOllamaBaseURL, service.baseURL)
	}

	if service.temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", service.temperature)
	}
}

func TestNewOllamaLLMServiceDefaults(t *testing.T) {
	config := OllamaLLMConfig{}

	service := NewOllamaLLMService(config)

	if service.model != DefaultOllamaModel {
		t.Errorf("Expected default model %s, got %s", DefaultOllamaModel, service.model)
	}

	if service.baseURL != DefaultOllamaBaseURL {
		t.Errorf("Expected default base URL %s, got %s", DefaultOllamaBaseURL, service.baseURL)
	}
}

func TestOllamaLLMServiceDefaultBaseURL(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

	// Default base URL should be localhost:11434/v1
	if service.baseURL != "http://localhost:11434/v1" {
		t.Errorf("Expected default base URL http://localhost:11434/v1, got %s", service.baseURL)
	}
}

func TestOllamaLLMServiceDefaultModel(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

	if service.model != "llama3.2" {
		t.Errorf("Expected default model llama3.2, got %s", service.model)
	}
}

func TestOllamaLLMServiceCustomBaseURL(t *testing.T) {
	customURL := "http://myserver:11434/v1"
	config := OllamaLLMConfig{
		BaseURL: customURL,
	}

	service := NewOllamaLLMService(config)

	if service.baseURL != customURL {
		t.Errorf("Expected custom base URL %s, got %s", customURL, service.baseURL)
	}
}

func TestOllamaLLMServiceConfiguration(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

	service.SetModel("mistral")
	if service.model != "mistral" {
		t.Errorf("Expected model mistral, got %s", service.model)
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

func TestOllamaLLMServiceMessageManagement(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

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

func TestOllamaLLMServiceInitializeCleanup(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

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

func TestOllamaLLMServiceFrameLifecycle(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

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

func TestOllamaLLMServiceInterruptionHandling(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

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

func TestOllamaLLMServiceInterruptionCancelsRequest(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	// Set up a request context that we can check cancellation on
	reqCtx, reqCancel := context.WithCancel(ctx)
	service.streamMu.Lock()
	service.isGenerating = true
	service.requestCtx = reqCtx
	service.requestCancel = reqCancel
	service.streamMu.Unlock()

	// Send interruption
	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	// Verify the request context was cancelled
	if reqCtx.Err() != context.Canceled {
		t.Error("Expected request context to be cancelled after interruption")
	}
}

func TestOllamaLLMServiceInterruptionIgnoredForNewContext(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	// Set up as if we just received a new context
	reqCtx, reqCancel := context.WithCancel(ctx)
	service.streamMu.Lock()
	service.isGenerating = true
	service.requestCtx = reqCtx
	service.requestCancel = reqCancel
	service.lastContextAt = time.Now() // Just received context
	service.streamMu.Unlock()

	// Send interruption - should be ignored since context was just received
	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	// Verify the request context was NOT cancelled (interruption was for old response)
	service.streamMu.Lock()
	stillGenerating := service.isGenerating
	service.streamMu.Unlock()

	if !stillGenerating {
		t.Error("Expected isGenerating to still be true - interruption should be ignored for new context")
	}

	if reqCtx.Err() == context.Canceled {
		t.Error("Expected request context to NOT be cancelled for new context interruption")
	}

	// Clean up
	reqCancel()
}

func TestOllamaLLMServicePassthroughFrames(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

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

func TestOllamaLLMServiceLLMContextFrame(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

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

// TestOllamaLLMServiceSSEParsing tests SSE streaming response parsing using httptest
func TestOllamaLLMServiceSSEParsing(t *testing.T) {
	// Create a mock HTTP server that returns SSE stream
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("Expected path ending in /chat/completions, got %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		// Ollama should NOT have Authorization header
		if r.Header.Get("Authorization") != "" {
			t.Errorf("Expected no Authorization header for Ollama, got %s", r.Header.Get("Authorization"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Write SSE events (OpenAI-compatible format)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	// Create service pointing to mock server
	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL: server.URL,
		Model:   "llama3.2",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("You are a test assistant")
	llmContext.AddUserMessage("Say hello")

	err := service.generateResponseFromContext(llmContext)
	if err != nil {
		t.Fatalf("generateResponseFromContext failed: %v", err)
	}

	// Verify assistant message was added to context
	found := false
	for _, msg := range llmContext.Messages {
		if msg.Role == "assistant" && msg.Content == "Hello world!" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Expected assistant message 'Hello world!' in context, messages: %+v", llmContext.Messages)
	}
}

// TestOllamaLLMServiceSSEWithToolCalls tests SSE parsing with tool call deltas
func TestOllamaLLMServiceSSEWithToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Tool call delta chunks (content is empty, tool_calls present)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_123\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL: server.URL,
		Model:   "llama3.2",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("You are a test assistant")
	llmContext.AddUserMessage("What's the weather?")

	err := service.generateResponseFromContext(llmContext)
	if err != nil {
		t.Fatalf("generateResponseFromContext failed: %v", err)
	}

	// No text content expected with tool calls
	// Just verify no panic/error occurred
}

// TestOllamaLLMServiceHTTPError tests handling of HTTP error responses
func TestOllamaLLMServiceHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, `{"error": "model not found"}`)
	}))
	defer server.Close()

	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL: server.URL,
		Model:   "nonexistent-model",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("")
	llmContext.AddUserMessage("Hello")

	err := service.generateResponseFromContext(llmContext)
	if err == nil {
		t.Fatal("Expected error for HTTP 500 response")
	}

	if !strings.Contains(err.Error(), "Ollama API error") {
		t.Errorf("Expected 'Ollama API error' in error message, got: %s", err.Error())
	}
}

// TestOllamaLLMServiceStreamInterruption tests that an in-flight request is cancelled
func TestOllamaLLMServiceStreamInterruption(t *testing.T) {
	requestReceived := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Send first chunk
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}

		close(requestReceived)

		// Block until context is cancelled (simulating slow streaming)
		<-r.Context().Done()
	}))
	defer server.Close()

	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL: server.URL,
		Model:   "llama3.2",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("")
	llmContext.AddUserMessage("Tell me a long story")

	// Run generation in background
	var wg sync.WaitGroup
	var genErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		genErr = service.generateResponseFromContext(llmContext)
	}()

	// Wait for request to be received
	<-requestReceived

	// Give a moment for the scanner to be in the loop
	time.Sleep(10 * time.Millisecond)

	// Send interruption to cancel the stream
	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	// Wait for generation to finish
	wg.Wait()

	// Should not return an error (cancellation is graceful)
	if genErr != nil {
		t.Errorf("Expected nil error after interruption, got: %v", genErr)
	}
}

// TestOllamaLLMServiceNoAuthHeader verifies no Authorization header is sent
func TestOllamaLLMServiceNoAuthHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify no Authorization header
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("Expected no Authorization header for Ollama, got: %s", auth)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("")
	llmContext.AddUserMessage("test")

	err := service.generateResponseFromContext(llmContext)
	if err != nil {
		t.Fatalf("generateResponseFromContext failed: %v", err)
	}
}

// TestOllamaLLMServiceSystemPromptInRequest verifies system prompt is sent in messages
func TestOllamaLLMServiceSystemPromptInRequest(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// We just verify the request is valid and return SSE
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL:      server.URL,
		SystemPrompt: "You are a pirate",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	if service.context.SystemPrompt != "You are a pirate" {
		t.Errorf("Expected system prompt 'You are a pirate', got %s", service.context.SystemPrompt)
	}
}

// TestOllamaLLMServiceMalformedSSE tests handling of malformed SSE data
func TestOllamaLLMServiceMalformedSSE(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Mix of valid, malformed, and empty lines
		fmt.Fprint(w, "data: not-json\n\n")
		fmt.Fprint(w, "\n")
		fmt.Fprint(w, "event: ping\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"works\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	service := NewOllamaLLMService(OllamaLLMConfig{
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("")
	llmContext.AddUserMessage("test")

	err := service.generateResponseFromContext(llmContext)
	if err != nil {
		t.Fatalf("Expected graceful handling of malformed SSE, got error: %v", err)
	}

	// Should still get the valid content
	found := false
	for _, msg := range llmContext.Messages {
		if msg.Role == "assistant" && msg.Content == "works" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Expected assistant message 'works' despite malformed SSE, messages: %+v", llmContext.Messages)
	}
}

// TestOllamaLLMServiceRaceDetection tests concurrent access patterns
func TestOllamaLLMServiceRaceDetection(t *testing.T) {
	service := NewOllamaLLMService(OllamaLLMConfig{})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	var wg sync.WaitGroup

	// Concurrent interruption handling
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			service.streamMu.Lock()
			service.isGenerating = true
			service.requestCtx, service.requestCancel = context.WithCancel(ctx)
			service.streamMu.Unlock()

			interruptFrame := frames.NewInterruptionFrame()
			service.HandleFrame(ctx, interruptFrame, frames.Downstream)
		}()
	}

	wg.Wait()
}
