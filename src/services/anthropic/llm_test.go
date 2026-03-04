package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// frameCapturer captures frames pushed downstream for test verification
type frameCapturer struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (c *frameCapturer) QueueFrame(frame frames.Frame, direction frames.FrameDirection) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, frame)
	return nil
}

func (c *frameCapturer) ProcessFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (c *frameCapturer) PushFrame(frame frames.Frame, direction frames.FrameDirection) error {
	return nil
}

func (c *frameCapturer) Link(next processors.FrameProcessor)    {}
func (c *frameCapturer) SetPrev(prev processors.FrameProcessor) {}
func (c *frameCapturer) Start(ctx context.Context) error        { return nil }
func (c *frameCapturer) Stop() error                            { return nil }
func (c *frameCapturer) Name() string                           { return "TestCapturer" }

func (c *frameCapturer) getFrames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([]frames.Frame, len(c.frames))
	copy(result, c.frames)
	return result
}

// writeSSE writes an SSE event to the response writer and flushes
func writeSSE(w http.ResponseWriter, flusher http.Flusher, eventType string, data interface{}) {
	jsonBytes, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\n", eventType)
	fmt.Fprintf(w, "data: %s\n\n", string(jsonBytes))
	flusher.Flush()
}

// --- Config Validation Tests ---

func TestNewLLMService(t *testing.T) {
	config := LLMConfig{
		APIKey:       "test-api-key",
		Model:        "claude-sonnet-4-6",
		SystemPrompt: "You are a helpful assistant",
		Temperature:  0.7,
		MaxTokens:    2048,
	}

	service := NewLLMService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}
	if service.model != "claude-sonnet-4-6" {
		t.Errorf("Expected model claude-sonnet-4-6, got %s", service.model)
	}
	if service.baseURL != DefaultBaseURL {
		t.Errorf("Expected base URL %s, got %s", DefaultBaseURL, service.baseURL)
	}
	if service.temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", service.temperature)
	}
	if service.apiKey != "test-api-key" {
		t.Errorf("Expected API key test-api-key, got %s", service.apiKey)
	}
	if service.maxTokens != 2048 {
		t.Errorf("Expected max tokens 2048, got %d", service.maxTokens)
	}
	if service.context.SystemPrompt != "You are a helpful assistant" {
		t.Errorf("Expected system prompt set, got %s", service.context.SystemPrompt)
	}
}

func TestNewLLMServiceDefaults(t *testing.T) {
	config := LLMConfig{
		APIKey: "test-api-key",
	}

	service := NewLLMService(config)

	if service.model != DefaultModel {
		t.Errorf("Expected default model %s, got %s", DefaultModel, service.model)
	}
	if service.baseURL != DefaultBaseURL {
		t.Errorf("Expected default base URL %s, got %s", DefaultBaseURL, service.baseURL)
	}
	if service.maxTokens != DefaultMaxTokens {
		t.Errorf("Expected default max tokens %d, got %d", DefaultMaxTokens, service.maxTokens)
	}
}

func TestLLMServiceCustomBaseURL(t *testing.T) {
	customURL := "https://custom.anthropic.com/v1"
	config := LLMConfig{
		APIKey:  "test-api-key",
		BaseURL: customURL,
	}

	service := NewLLMService(config)

	if service.baseURL != customURL {
		t.Errorf("Expected custom base URL %s, got %s", customURL, service.baseURL)
	}
}

// --- Configuration Tests ---

func TestLLMServiceConfiguration(t *testing.T) {
	service := NewLLMService(LLMConfig{
		APIKey: "test-api-key",
	})

	service.SetModel("claude-3-haiku-20240307")
	if service.model != "claude-3-haiku-20240307" {
		t.Errorf("Expected model claude-3-haiku-20240307, got %s", service.model)
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

func TestLLMServiceMessageManagement(t *testing.T) {
	service := NewLLMService(LLMConfig{
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

// --- Lifecycle Tests ---

func TestLLMServiceInitializeCleanup(t *testing.T) {
	service := NewLLMService(LLMConfig{
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

func TestLLMServiceFrameLifecycle(t *testing.T) {
	service := NewLLMService(LLMConfig{
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

func TestLLMServicePassthroughFrames(t *testing.T) {
	service := NewLLMService(LLMConfig{
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

// --- InterruptionFrame Handling Tests ---

func TestLLMServiceInterruptionHandling(t *testing.T) {
	service := NewLLMService(LLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	// Simulate active generation
	service.streamMu.Lock()
	service.isGenerating = true
	service.requestCtx, service.requestCancel = context.WithCancel(ctx)
	service.streamMu.Unlock()

	// Send interruption
	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	// Verify generation was stopped
	service.streamMu.Lock()
	wasGenerating := service.isGenerating
	service.streamMu.Unlock()

	if wasGenerating {
		t.Error("Expected isGenerating to be false after interruption")
	}
}

func TestLLMServiceInterruptionIgnoredForNewContext(t *testing.T) {
	service := NewLLMService(LLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	// Simulate active generation with very recent context
	service.streamMu.Lock()
	service.isGenerating = true
	service.requestCtx, service.requestCancel = context.WithCancel(ctx)
	service.lastContextAt = time.Now() // Just received context
	service.streamMu.Unlock()

	// Send interruption immediately (within 100ms window)
	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	// Verify generation was NOT stopped (interruption was for old response)
	service.streamMu.Lock()
	stillGenerating := service.isGenerating
	service.streamMu.Unlock()

	if !stillGenerating {
		t.Error("Expected isGenerating to remain true when interruption occurs within new context window")
	}

	// Clean up
	service.streamMu.Lock()
	if service.requestCancel != nil {
		service.requestCancel()
	}
	service.streamMu.Unlock()
}

// --- SSE Parsing Tests ---

func TestLLMServiceSSEParsing(t *testing.T) {
	// Create mock Anthropic API server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request method and path
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/messages" {
			t.Errorf("Expected /messages, got %s", r.URL.Path)
		}

		// Verify Anthropic-specific headers
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("Expected x-api-key test-key, got %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != APIVersion {
			t.Errorf("Expected anthropic-version %s, got %s", APIVersion, r.Header.Get("anthropic-version"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}

		// Verify request body
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != DefaultModel {
			t.Errorf("Expected model %s, got %v", DefaultModel, body["model"])
		}
		if body["stream"] != true {
			t.Errorf("Expected stream true, got %v", body["stream"])
		}
		if body["system"] != "You are helpful" {
			t.Errorf("Expected system 'You are helpful', got %v", body["system"])
		}
		// max_tokens should be present (Anthropic requires it)
		if body["max_tokens"] == nil {
			t.Error("Expected max_tokens to be set")
		}

		// Send SSE response
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)

		writeSSE(w, flusher, "message_start", map[string]interface{}{
			"type":    "message_start",
			"message": map[string]interface{}{"id": "msg_test"},
		})
		writeSSE(w, flusher, "content_block_start", map[string]interface{}{
			"type":          "content_block_start",
			"index":         0,
			"content_block": map[string]interface{}{"type": "text", "text": ""},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{"type": "text_delta", "text": "Hello"},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{"type": "text_delta", "text": " world"},
		})
		writeSSE(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 0,
		})
		writeSSE(w, flusher, "message_delta", map[string]interface{}{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "end_turn"},
		})
		writeSSE(w, flusher, "message_stop", map[string]interface{}{
			"type": "message_stop",
		})
	}))
	defer server.Close()

	// Create service pointing to mock server
	service := NewLLMService(LLMConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	// Link a frame capturer to receive downstream frames
	capturer := &frameCapturer{}
	service.Link(capturer)

	// Create context and send LLMContextFrame
	llmContext := services.NewLLMContext("You are helpful")
	llmContext.AddUserMessage("Hello")

	contextFrame := frames.NewLLMContextFrame(llmContext)
	err := service.HandleFrame(ctx, contextFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame failed: %v", err)
	}

	// Verify captured frames
	captured := capturer.getFrames()
	if len(captured) != 4 {
		t.Fatalf("Expected 4 frames (start, text, text, end), got %d", len(captured))
	}

	// Frame 0: LLMFullResponseStartFrame
	if _, ok := captured[0].(*frames.LLMFullResponseStartFrame); !ok {
		t.Errorf("Frame 0: expected LLMFullResponseStartFrame, got %T", captured[0])
	}

	// Frame 1: LLMTextFrame "Hello"
	if tf, ok := captured[1].(*frames.LLMTextFrame); ok {
		if tf.Text != "Hello" {
			t.Errorf("Frame 1: expected text 'Hello', got %q", tf.Text)
		}
	} else {
		t.Errorf("Frame 1: expected LLMTextFrame, got %T", captured[1])
	}

	// Frame 2: LLMTextFrame " world"
	if tf, ok := captured[2].(*frames.LLMTextFrame); ok {
		if tf.Text != " world" {
			t.Errorf("Frame 2: expected text ' world', got %q", tf.Text)
		}
	} else {
		t.Errorf("Frame 2: expected LLMTextFrame, got %T", captured[2])
	}

	// Frame 3: LLMFullResponseEndFrame
	if _, ok := captured[3].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("Frame 3: expected LLMFullResponseEndFrame, got %T", captured[3])
	}

	// Verify context was updated with assistant response
	if len(llmContext.Messages) != 2 {
		t.Fatalf("Expected 2 messages in context (user + assistant), got %d", len(llmContext.Messages))
	}
	lastMsg := llmContext.Messages[1]
	if lastMsg.Role != "assistant" {
		t.Errorf("Expected last message role 'assistant', got %s", lastMsg.Role)
	}
	if lastMsg.Content != "Hello world" {
		t.Errorf("Expected assistant content 'Hello world', got %q", lastMsg.Content)
	}
}

// --- Tool Calling Tests ---

func TestLLMServiceToolCalling(t *testing.T) {
	// Create mock server that returns tool use response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify tools in request body
		var body map[string]interface{}
		json.NewDecoder(r.Body).Decode(&body)

		tools, ok := body["tools"].([]interface{})
		if !ok || len(tools) != 1 {
			t.Errorf("Expected 1 tool in request, got %v", body["tools"])
		}

		// Verify tool uses input_schema (Anthropic format, not parameters)
		if len(tools) > 0 {
			tool := tools[0].(map[string]interface{})
			if tool["input_schema"] == nil {
				t.Error("Expected input_schema in tool definition")
			}
			if tool["name"] != "get_weather" {
				t.Errorf("Expected tool name 'get_weather', got %v", tool["name"])
			}
		}

		// Send SSE response with tool use
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)

		writeSSE(w, flusher, "message_start", map[string]interface{}{
			"type":    "message_start",
			"message": map[string]interface{}{"id": "msg_tool"},
		})
		writeSSE(w, flusher, "content_block_start", map[string]interface{}{
			"type":  "content_block_start",
			"index": 0,
			"content_block": map[string]interface{}{
				"type": "tool_use",
				"id":   "toolu_test123",
				"name": "get_weather",
			},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{
				"type":         "input_json_delta",
				"partial_json": `{"location": `,
			},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{
				"type":         "input_json_delta",
				"partial_json": `"San Francisco"}`,
			},
		})
		writeSSE(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 0,
		})
		writeSSE(w, flusher, "message_delta", map[string]interface{}{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": "tool_use"},
		})
		writeSSE(w, flusher, "message_stop", map[string]interface{}{
			"type": "message_stop",
		})
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	capturer := &frameCapturer{}
	service.Link(capturer)

	// Set up context with tools
	llmContext := services.NewLLMContext("You are helpful")
	llmContext.AddUserMessage("What's the weather in San Francisco?")
	llmContext.SetTools([]services.Tool{
		{
			Type: "function",
			Function: services.ToolFunction{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type": "string",
						},
					},
				},
			},
		},
	})

	contextFrame := frames.NewLLMContextFrame(llmContext)
	err := service.HandleFrame(ctx, contextFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame failed: %v", err)
	}

	captured := capturer.getFrames()
	// Should have: LLMFullResponseStartFrame, FunctionCallInProgressFrame, LLMFullResponseEndFrame
	if len(captured) != 3 {
		t.Fatalf("Expected 3 frames (start, func_call, end), got %d", len(captured))
	}

	// Frame 0: LLMFullResponseStartFrame
	if _, ok := captured[0].(*frames.LLMFullResponseStartFrame); !ok {
		t.Errorf("Frame 0: expected LLMFullResponseStartFrame, got %T", captured[0])
	}

	// Frame 1: FunctionCallInProgressFrame
	fcf, ok := captured[1].(*frames.FunctionCallInProgressFrame)
	if !ok {
		t.Fatalf("Frame 1: expected FunctionCallInProgressFrame, got %T", captured[1])
	}
	if fcf.ToolCallID != "toolu_test123" {
		t.Errorf("Expected tool call ID 'toolu_test123', got %s", fcf.ToolCallID)
	}
	if fcf.FunctionName != "get_weather" {
		t.Errorf("Expected function name 'get_weather', got %s", fcf.FunctionName)
	}
	if fcf.Arguments["location"] != "San Francisco" {
		t.Errorf("Expected location 'San Francisco', got %v", fcf.Arguments["location"])
	}
	if !fcf.CancelOnInterruption {
		t.Error("Expected CancelOnInterruption to be true")
	}

	// Frame 2: LLMFullResponseEndFrame
	if _, ok := captured[2].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("Frame 2: expected LLMFullResponseEndFrame, got %T", captured[2])
	}

	// Verify tool call was added to context
	lastMsg := llmContext.Messages[len(llmContext.Messages)-1]
	if lastMsg.Role != "assistant" {
		t.Errorf("Expected last message role 'assistant', got %s", lastMsg.Role)
	}
	if len(lastMsg.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call in context, got %d", len(lastMsg.ToolCalls))
	}
	if lastMsg.ToolCalls[0].ID != "toolu_test123" {
		t.Errorf("Expected tool call ID 'toolu_test123', got %s", lastMsg.ToolCalls[0].ID)
	}
	if lastMsg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("Expected function name 'get_weather', got %s", lastMsg.ToolCalls[0].Function.Name)
	}
}

func TestLLMServiceToolCallingWithTextAndToolUse(t *testing.T) {
	// Test mixed response: text + tool_use in same message
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)

		writeSSE(w, flusher, "message_start", map[string]interface{}{
			"type":    "message_start",
			"message": map[string]interface{}{"id": "msg_mixed"},
		})
		// Text block first
		writeSSE(w, flusher, "content_block_start", map[string]interface{}{
			"type":          "content_block_start",
			"index":         0,
			"content_block": map[string]interface{}{"type": "text", "text": ""},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{"type": "text_delta", "text": "Let me check."},
		})
		writeSSE(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 0,
		})
		// Then tool_use block
		writeSSE(w, flusher, "content_block_start", map[string]interface{}{
			"type":  "content_block_start",
			"index": 1,
			"content_block": map[string]interface{}{
				"type": "tool_use",
				"id":   "toolu_mixed",
				"name": "lookup",
			},
		})
		writeSSE(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 1,
			"delta": map[string]interface{}{
				"type":         "input_json_delta",
				"partial_json": `{"query": "test"}`,
			},
		})
		writeSSE(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 1,
		})
		writeSSE(w, flusher, "message_stop", map[string]interface{}{
			"type": "message_stop",
		})
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	capturer := &frameCapturer{}
	service.Link(capturer)

	llmContext := services.NewLLMContext("")
	llmContext.AddUserMessage("test")
	llmContext.SetTools([]services.Tool{{
		Type:     "function",
		Function: services.ToolFunction{Name: "lookup", Description: "Lookup"},
	}})

	contextFrame := frames.NewLLMContextFrame(llmContext)
	service.HandleFrame(ctx, contextFrame, frames.Downstream)

	captured := capturer.getFrames()
	// start, text("Let me check."), func_call, end
	if len(captured) != 4 {
		t.Fatalf("Expected 4 frames, got %d", len(captured))
	}

	if tf, ok := captured[1].(*frames.LLMTextFrame); ok {
		if tf.Text != "Let me check." {
			t.Errorf("Expected text 'Let me check.', got %q", tf.Text)
		}
	} else {
		t.Errorf("Frame 1: expected LLMTextFrame, got %T", captured[1])
	}

	if fcf, ok := captured[2].(*frames.FunctionCallInProgressFrame); ok {
		if fcf.FunctionName != "lookup" {
			t.Errorf("Expected function name 'lookup', got %s", fcf.FunctionName)
		}
	} else {
		t.Errorf("Frame 2: expected FunctionCallInProgressFrame, got %T", captured[2])
	}

	// Context should have assistant message with both text and tool calls
	lastMsg := llmContext.Messages[len(llmContext.Messages)-1]
	if lastMsg.Content != "Let me check." {
		t.Errorf("Expected content 'Let me check.', got %q", lastMsg.Content)
	}
	if len(lastMsg.ToolCalls) != 1 {
		t.Errorf("Expected 1 tool call, got %d", len(lastMsg.ToolCalls))
	}
}

// --- Interruption Cancellation Tests ---

func TestLLMServiceInterruptionCancelsRequest(t *testing.T) {
	requestReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}

		close(requestReceived)

		// Block until request context is cancelled
		<-r.Context().Done()
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	capturer := &frameCapturer{}
	service.Link(capturer)

	llmContext := services.NewLLMContext("You are helpful")
	llmContext.AddUserMessage("Hello")

	// Start generation in background
	done := make(chan error, 1)
	go func() {
		contextFrame := frames.NewLLMContextFrame(llmContext)
		done <- service.HandleFrame(ctx, contextFrame, frames.Downstream)
	}()

	// Wait for server to receive the request
	select {
	case <-requestReceived:
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout waiting for server to receive request")
	}

	// Wait past the 100ms new-context window so interruption is not ignored
	time.Sleep(150 * time.Millisecond)

	// Send interruption frame
	interruptFrame := frames.NewInterruptionFrame()
	err := service.HandleFrame(ctx, interruptFrame, frames.Downstream)
	if err != nil {
		t.Errorf("InterruptionFrame handling failed: %v", err)
	}

	// Wait for generation to complete
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Expected nil error after interruption, got: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout waiting for generation to complete after interruption")
	}

	// Verify isGenerating is false
	service.streamMu.Lock()
	stillGenerating := service.isGenerating
	service.streamMu.Unlock()
	if stillGenerating {
		t.Error("Expected isGenerating to be false after interruption")
	}
}

// --- Context Management Tests ---

func TestLLMServiceContextManagement(t *testing.T) {
	service := NewLLMService(LLMConfig{
		APIKey: "test-api-key",
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("You are a helpful assistant")
	llmContext.AddUserMessage("Hello")

	contextFrame := frames.NewLLMContextFrame(llmContext)

	// HandleFrame will try to call generateResponseFromContext which will fail
	// (no real server), but the context should still be updated
	service.HandleFrame(ctx, contextFrame, frames.Downstream)

	// Verify context was set on service
	if service.context != llmContext {
		t.Error("Expected service context to be updated to the provided LLMContext")
	}
	if len(service.context.Messages) != 1 {
		t.Errorf("Expected 1 message in context, got %d", len(service.context.Messages))
	}
}

// --- API Error Tests ---

func TestLLMServiceAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"API key required"}}`)
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:  "bad-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	capturer := &frameCapturer{}
	service.Link(capturer)

	llmContext := services.NewLLMContext("Test")
	llmContext.AddUserMessage("Hello")

	contextFrame := frames.NewLLMContextFrame(llmContext)
	service.HandleFrame(ctx, contextFrame, frames.Downstream)

	// Verify start and end frames were still emitted (error is pushed upstream)
	captured := capturer.getFrames()
	if len(captured) < 2 {
		t.Fatalf("Expected at least 2 frames (start + end), got %d", len(captured))
	}

	if _, ok := captured[0].(*frames.LLMFullResponseStartFrame); !ok {
		t.Errorf("Frame 0: expected LLMFullResponseStartFrame, got %T", captured[0])
	}
	if _, ok := captured[1].(*frames.LLMFullResponseEndFrame); !ok {
		t.Errorf("Frame 1: expected LLMFullResponseEndFrame, got %T", captured[1])
	}
}

// --- Request Body Format Tests ---

func TestLLMServiceRequestBodyFormat(t *testing.T) {
	var capturedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)

		// Return minimal valid SSE
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		writeSSE(w, flusher, "message_stop", map[string]interface{}{"type": "message_stop"})
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:      "test-key",
		BaseURL:     server.URL,
		Temperature: 0.5,
		MaxTokens:   1024,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	capturer := &frameCapturer{}
	service.Link(capturer)

	llmContext := services.NewLLMContext("Be concise")
	llmContext.AddUserMessage("Hi")
	llmContext.AddAssistantMessage("Hello!")
	llmContext.AddUserMessage("How are you?")

	contextFrame := frames.NewLLMContextFrame(llmContext)
	service.HandleFrame(ctx, contextFrame, frames.Downstream)

	// Verify system prompt is top-level (not a message)
	if capturedBody["system"] != "Be concise" {
		t.Errorf("Expected system='Be concise', got %v", capturedBody["system"])
	}

	// Verify messages don't include system role
	msgs := capturedBody["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("Expected 3 messages (user, assistant, user), got %d", len(msgs))
	}
	firstMsg := msgs[0].(map[string]interface{})
	if firstMsg["role"] != "user" {
		t.Errorf("Expected first message role 'user', got %v", firstMsg["role"])
	}

	// Verify temperature and max_tokens
	if capturedBody["temperature"] != 0.5 {
		t.Errorf("Expected temperature 0.5, got %v", capturedBody["temperature"])
	}
	// JSON numbers decode as float64
	if capturedBody["max_tokens"] != float64(1024) {
		t.Errorf("Expected max_tokens 1024, got %v", capturedBody["max_tokens"])
	}
	if capturedBody["stream"] != true {
		t.Errorf("Expected stream true, got %v", capturedBody["stream"])
	}
}

func TestLLMServiceToolResultMessageFormat(t *testing.T) {
	var capturedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		writeSSE(w, flusher, "message_stop", map[string]interface{}{"type": "message_stop"})
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	capturer := &frameCapturer{}
	service.Link(capturer)

	// Build context with tool call + tool result
	llmContext := services.NewLLMContext("")
	llmContext.AddUserMessage("What's the weather?")
	llmContext.AddMessageWithToolCalls([]services.ToolCall{
		{
			ID:   "toolu_abc",
			Type: "function",
			Function: services.FunctionCall{
				Name:      "get_weather",
				Arguments: `{"location": "NYC"}`,
			},
		},
	})
	llmContext.AddToolMessage("toolu_abc", "Sunny, 72F")

	contextFrame := frames.NewLLMContextFrame(llmContext)
	service.HandleFrame(ctx, contextFrame, frames.Downstream)

	// Verify messages format
	msgs := capturedBody["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(msgs))
	}

	// Message 0: user
	msg0 := msgs[0].(map[string]interface{})
	if msg0["role"] != "user" {
		t.Errorf("Expected msg[0] role 'user', got %v", msg0["role"])
	}

	// Message 1: assistant with tool_use content blocks
	msg1 := msgs[1].(map[string]interface{})
	if msg1["role"] != "assistant" {
		t.Errorf("Expected msg[1] role 'assistant', got %v", msg1["role"])
	}
	content1 := msg1["content"].([]interface{})
	if len(content1) != 1 {
		t.Fatalf("Expected 1 content block in assistant message, got %d", len(content1))
	}
	toolUseBlock := content1[0].(map[string]interface{})
	if toolUseBlock["type"] != "tool_use" {
		t.Errorf("Expected content block type 'tool_use', got %v", toolUseBlock["type"])
	}
	if toolUseBlock["id"] != "toolu_abc" {
		t.Errorf("Expected tool_use id 'toolu_abc', got %v", toolUseBlock["id"])
	}

	// Message 2: user with tool_result (Anthropic format)
	msg2 := msgs[2].(map[string]interface{})
	if msg2["role"] != "user" {
		t.Errorf("Expected msg[2] role 'user' (tool_result), got %v", msg2["role"])
	}
	content2 := msg2["content"].([]interface{})
	if len(content2) != 1 {
		t.Fatalf("Expected 1 content block in tool_result message, got %d", len(content2))
	}
	toolResult := content2[0].(map[string]interface{})
	if toolResult["type"] != "tool_result" {
		t.Errorf("Expected content block type 'tool_result', got %v", toolResult["type"])
	}
	if toolResult["tool_use_id"] != "toolu_abc" {
		t.Errorf("Expected tool_use_id 'toolu_abc', got %v", toolResult["tool_use_id"])
	}
	if toolResult["content"] != "Sunny, 72F" {
		t.Errorf("Expected tool result content 'Sunny, 72F', got %v", toolResult["content"])
	}
}

// --- Race Detection Test ---

func TestLLMServiceRaceDetection(t *testing.T) {
	// This test verifies no race conditions when interruption and generation
	// happen concurrently (run with -race flag)
	requestReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		close(requestReceived)
		<-r.Context().Done()
	}))
	defer server.Close()

	service := NewLLMService(LLMConfig{
		APIKey:  "test-key",
		BaseURL: server.URL,
	})

	ctx := context.Background()
	service.Initialize(ctx)
	defer service.Cleanup()

	llmContext := services.NewLLMContext("Test")
	llmContext.AddUserMessage("Hello")

	done := make(chan struct{})
	go func() {
		defer close(done)
		contextFrame := frames.NewLLMContextFrame(llmContext)
		service.HandleFrame(ctx, contextFrame, frames.Downstream)
	}()

	select {
	case <-requestReceived:
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout")
	}

	time.Sleep(150 * time.Millisecond)

	// Concurrent interruption
	service.HandleFrame(ctx, frames.NewInterruptionFrame(), frames.Downstream)

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout waiting for completion")
	}
}
