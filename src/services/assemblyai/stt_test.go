package assemblyai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

// mockCollector captures frames pushed by the service for test assertions
type mockCollector struct {
	*processors.BaseProcessor
	mu     sync.Mutex
	frames []frames.Frame
}

func newMockCollector() *mockCollector {
	c := &mockCollector{
		frames: make([]frames.Frame, 0),
	}
	c.BaseProcessor = processors.NewBaseProcessor("MockCollector", c)
	return c
}

func (c *mockCollector) HandleFrame(_ context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	c.mu.Lock()
	c.frames = append(c.frames, frame)
	c.mu.Unlock()
	return c.PushFrame(frame, direction)
}

func (c *mockCollector) getFrames() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	result := make([]frames.Frame, len(c.frames))
	copy(result, c.frames)
	return result
}

// startMockWSServer creates a mock WebSocket server for testing.
// handler receives the upgraded connection for custom behavior.
func startMockWSServer(t *testing.T, handler func(conn *websocket.Conn)) *httptest.Server {
	t.Helper()
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Logf("Mock server upgrade error: %v", err)
			return
		}
		defer conn.Close()
		handler(conn)
	}))
	return server
}

// wsURL converts an HTTP test server URL to a WebSocket URL
func wsURL(server *httptest.Server) string {
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func TestNewSTTService(t *testing.T) {
	config := STTConfig{
		APIKey:   "test-api-key",
		Language: "en",
		Model:    "u3-rt-pro",
	}

	service := NewSTTService(config)

	if service == nil {
		t.Fatal("Expected service to be created")
	}

	if service.apiKey != "test-api-key" {
		t.Errorf("Expected API key 'test-api-key', got %s", service.apiKey)
	}

	if service.language != "en" {
		t.Errorf("Expected language 'en', got %s", service.language)
	}

	if service.model != "u3-rt-pro" {
		t.Errorf("Expected model 'u3-rt-pro', got %s", service.model)
	}
}

func TestNewSTTServiceDefaults(t *testing.T) {
	config := STTConfig{
		APIKey: "test-key",
	}

	service := NewSTTService(config)

	if service.model != DefaultModel {
		t.Errorf("Expected default model %s, got %s", DefaultModel, service.model)
	}

	if service.sampleRate != DefaultSampleRate {
		t.Errorf("Expected default sample rate %d, got %d", DefaultSampleRate, service.sampleRate)
	}

	if service.endUtteranceSilenceThreshold != DefaultEndUtteranceSilenceThreshold {
		t.Errorf("Expected default silence threshold %d, got %d",
			DefaultEndUtteranceSilenceThreshold, service.endUtteranceSilenceThreshold)
	}

	if service.baseURL != DefaultBaseURL {
		t.Errorf("Expected default base URL %s, got %s", DefaultBaseURL, service.baseURL)
	}
}

func TestSTTServiceConfiguration(t *testing.T) {
	tests := []struct {
		name                     string
		config                   STTConfig
		expectedModel            string
		expectedSampleRate       int
		expectedSilenceThreshold int
		expectedLang             string
	}{
		{
			name: "Custom configuration",
			config: STTConfig{
				APIKey:                       "test-key",
				Language:                     "es",
				Model:                        "u3-rt-pro",
				SampleRate:                   8000,
				EndUtteranceSilenceThreshold: 500,
			},
			expectedModel:            "u3-rt-pro",
			expectedSampleRate:       8000,
			expectedSilenceThreshold: 500,
			expectedLang:             "es",
		},
		{
			name: "Default configuration",
			config: STTConfig{
				APIKey: "test-key",
			},
			expectedModel:            DefaultModel,
			expectedSampleRate:       DefaultSampleRate,
			expectedSilenceThreshold: DefaultEndUtteranceSilenceThreshold,
			expectedLang:             "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := NewSTTService(tt.config)

			if service.model != tt.expectedModel {
				t.Errorf("Expected model %s, got %s", tt.expectedModel, service.model)
			}

			if service.sampleRate != tt.expectedSampleRate {
				t.Errorf("Expected sample rate %d, got %d", tt.expectedSampleRate, service.sampleRate)
			}

			if service.endUtteranceSilenceThreshold != tt.expectedSilenceThreshold {
				t.Errorf("Expected silence threshold %d, got %d",
					tt.expectedSilenceThreshold, service.endUtteranceSilenceThreshold)
			}

			if service.language != tt.expectedLang {
				t.Errorf("Expected language %s, got %s", tt.expectedLang, service.language)
			}
		})
	}
}

func TestSTTSetLanguage(t *testing.T) {
	service := NewSTTService(STTConfig{APIKey: "test-key", Language: "en"})

	if service.language != "en" {
		t.Errorf("Expected initial language 'en', got %s", service.language)
	}

	service.SetLanguage("fr")

	if service.language != "fr" {
		t.Errorf("Expected language to be updated to 'fr', got %s", service.language)
	}
}

func TestSTTSetModel(t *testing.T) {
	service := NewSTTService(STTConfig{APIKey: "test-key"})
	service.SetModel("nano")

	if service.model != "nano" {
		t.Errorf("Expected model 'nano', got %s", service.model)
	}
}

func TestPartialTranscriptEmitsNonFinalFrame(t *testing.T) {
	// Mock server sends a PartialTranscript
	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config
		_, _, err := conn.ReadMessage()
		if err != nil {
			return
		}

		// Send SessionBegins
		sessionBegins := map[string]string{"message_type": "SessionBegins"}
		conn.WriteJSON(sessionBegins)

		// Read audio message
		_, _, err = conn.ReadMessage()
		if err != nil {
			return
		}

		// Send PartialTranscript
		partial := transcriptMessage{
			MessageType: "PartialTranscript",
			Text:        "hello",
			Confidence:  0.85,
		}
		conn.WriteJSON(partial)

		// Keep connection alive briefly
		time.Sleep(200 * time.Millisecond)
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	// Initialize manually
	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Send an audio frame to trigger transcript
	audioFrame := frames.NewAudioFrame([]byte{0x00, 0x01, 0x02}, 16000, 1)
	service.HandleFrame(ctx, audioFrame, frames.Downstream)

	// Wait for transcript to arrive
	deadline := time.After(2 * time.Second)
	for {
		collectedFrames := collector.getFrames()
		for _, f := range collectedFrames {
			if tf, ok := f.(*frames.TranscriptionFrame); ok {
				if tf.Text != "hello" {
					t.Errorf("Expected text 'hello', got %s", tf.Text)
				}
				if tf.IsFinal {
					t.Error("Expected IsFinal=false for PartialTranscript")
				}
				// Success - cleanup and return
				service.Cleanup()
				return
			}
		}

		select {
		case <-deadline:
			service.Cleanup()
			t.Fatal("Timed out waiting for partial transcript frame")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func TestFinalTranscriptEmitsFinalFrame(t *testing.T) {
	// Mock server sends a FinalTranscript
	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config
		_, _, err := conn.ReadMessage()
		if err != nil {
			return
		}

		// Send SessionBegins
		sessionBegins := map[string]string{"message_type": "SessionBegins"}
		conn.WriteJSON(sessionBegins)

		// Read audio message
		_, _, err = conn.ReadMessage()
		if err != nil {
			return
		}

		// Send FinalTranscript
		final := transcriptMessage{
			MessageType: "FinalTranscript",
			Text:        "hello world",
			Confidence:  0.95,
		}
		conn.WriteJSON(final)

		// Keep connection alive briefly
		time.Sleep(200 * time.Millisecond)
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	// Initialize manually
	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Send an audio frame
	audioFrame := frames.NewAudioFrame([]byte{0x00, 0x01, 0x02}, 16000, 1)
	service.HandleFrame(ctx, audioFrame, frames.Downstream)

	// Wait for transcript
	deadline := time.After(2 * time.Second)
	for {
		collectedFrames := collector.getFrames()
		for _, f := range collectedFrames {
			if tf, ok := f.(*frames.TranscriptionFrame); ok {
				if tf.Text != "hello world" {
					t.Errorf("Expected text 'hello world', got %s", tf.Text)
				}
				if !tf.IsFinal {
					t.Error("Expected IsFinal=true for FinalTranscript")
				}
				service.Cleanup()
				return
			}
		}

		select {
		case <-deadline:
			service.Cleanup()
			t.Fatal("Timed out waiting for final transcript frame")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func TestEndUtteranceSilenceThresholdSentInConfig(t *testing.T) {
	var receivedConfig sessionConfig
	configReceived := make(chan struct{})

	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config message
		_, message, err := conn.ReadMessage()
		if err != nil {
			return
		}

		if err := json.Unmarshal(message, &receivedConfig); err != nil {
			t.Errorf("Failed to parse session config: %v", err)
			return
		}
		close(configReceived)

		// Keep alive
		time.Sleep(500 * time.Millisecond)
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:                       "test-key",
		BaseURL:                      wsURL(server),
		EndUtteranceSilenceThreshold: 500,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer service.Cleanup()

	// Wait for config to be received
	select {
	case <-configReceived:
		if receivedConfig.EndUtteranceSilenceThreshold != 500 {
			t.Errorf("Expected end_utterance_silence_threshold=500, got %d",
				receivedConfig.EndUtteranceSilenceThreshold)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Timed out waiting for session config")
	}
}

func TestEndUtteranceSilenceThresholdDefault(t *testing.T) {
	var receivedConfig sessionConfig
	configReceived := make(chan struct{})

	server := startMockWSServer(t, func(conn *websocket.Conn) {
		_, message, err := conn.ReadMessage()
		if err != nil {
			return
		}

		if err := json.Unmarshal(message, &receivedConfig); err != nil {
			t.Errorf("Failed to parse session config: %v", err)
			return
		}
		close(configReceived)

		time.Sleep(500 * time.Millisecond)
	})
	defer server.Close()

	// Use defaults (no EndUtteranceSilenceThreshold set)
	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer service.Cleanup()

	select {
	case <-configReceived:
		if receivedConfig.EndUtteranceSilenceThreshold != DefaultEndUtteranceSilenceThreshold {
			t.Errorf("Expected default end_utterance_silence_threshold=%d, got %d",
				DefaultEndUtteranceSilenceThreshold, receivedConfig.EndUtteranceSilenceThreshold)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Timed out waiting for session config")
	}
}

func TestStartFramePassthrough(t *testing.T) {
	service := NewSTTService(STTConfig{APIKey: "test-key"})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	startFrame := frames.NewStartFrame()
	service.HandleFrame(ctx, startFrame, frames.Downstream)

	// Wait for frame to pass through
	time.Sleep(100 * time.Millisecond)

	collectedFrames := collector.getFrames()
	found := false
	for _, f := range collectedFrames {
		if _, ok := f.(*frames.StartFrame); ok {
			found = true
			break
		}
	}

	if !found {
		t.Error("Expected StartFrame to pass through")
	}
}

func TestEndFrameTriggersCleanup(t *testing.T) {
	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config
		_, _, _ = conn.ReadMessage()
		// Keep alive until closed
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				return
			}
		}
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	// Initialize connection
	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Verify connection exists
	if service.conn == nil {
		t.Fatal("Expected connection to be established")
	}

	// Send EndFrame
	endFrame := frames.NewEndFrame()
	service.HandleFrame(ctx, endFrame, frames.Downstream)

	// Wait for cleanup
	time.Sleep(200 * time.Millisecond)

	// Verify connection is cleaned up
	if service.conn != nil {
		t.Error("Expected connection to be nil after EndFrame cleanup")
	}
}

func TestConnectionErrorReturnsError(t *testing.T) {
	service := NewSTTService(STTConfig{
		APIKey:  "invalid-key",
		BaseURL: "ws://localhost:1", // Port that won't be listening
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	err := service.Initialize(ctx)
	if err == nil {
		t.Error("Expected Initialize to return an error for invalid endpoint")
		service.Cleanup()
	}
}

func TestEmptyTranscriptIgnored(t *testing.T) {
	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config
		_, _, err := conn.ReadMessage()
		if err != nil {
			return
		}

		// Send SessionBegins
		conn.WriteJSON(map[string]string{"message_type": "SessionBegins"})

		// Read audio
		_, _, err = conn.ReadMessage()
		if err != nil {
			return
		}

		// Send empty PartialTranscript (should be ignored)
		conn.WriteJSON(transcriptMessage{
			MessageType: "PartialTranscript",
			Text:        "",
			Confidence:  0.0,
		})

		// Send non-empty FinalTranscript
		conn.WriteJSON(transcriptMessage{
			MessageType: "FinalTranscript",
			Text:        "actual text",
			Confidence:  0.9,
		})

		time.Sleep(200 * time.Millisecond)
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	audioFrame := frames.NewAudioFrame([]byte{0x00}, 16000, 1)
	service.HandleFrame(ctx, audioFrame, frames.Downstream)

	// Wait for frames
	deadline := time.After(2 * time.Second)
	for {
		collectedFrames := collector.getFrames()
		for _, f := range collectedFrames {
			if tf, ok := f.(*frames.TranscriptionFrame); ok {
				if tf.Text == "" {
					t.Error("Empty transcript should not produce a frame")
				}
				if tf.Text == "actual text" && tf.IsFinal {
					// Success - only the non-empty final came through
					service.Cleanup()
					return
				}
			}
		}

		select {
		case <-deadline:
			service.Cleanup()
			t.Fatal("Timed out waiting for non-empty transcript frame")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func TestAPIKeyInWebSocketURL(t *testing.T) {
	var receivedURL string
	urlReceived := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedURL = r.URL.String()
		close(urlReceived)

		upgrader := websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()

		// Read session config
		_, _, _ = conn.ReadMessage()
		time.Sleep(500 * time.Millisecond)
	}))
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "my-secret-token",
		BaseURL: wsURL(server),
	})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	defer service.Cleanup()

	select {
	case <-urlReceived:
		if !strings.Contains(receivedURL, "token=my-secret-token") {
			t.Errorf("Expected URL to contain 'token=my-secret-token', got %s", receivedURL)
		}
		if !strings.Contains(receivedURL, "sample_rate=16000") {
			t.Errorf("Expected URL to contain 'sample_rate=16000', got %s", receivedURL)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Timed out waiting for URL capture")
	}
}

func TestLazyInitialization(t *testing.T) {
	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config
		_, _, _ = conn.ReadMessage()
		// Read audio
		_, _, _ = conn.ReadMessage()
		time.Sleep(500 * time.Millisecond)
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	// Connection should not exist yet
	if service.conn != nil {
		t.Error("Expected no connection before first audio frame")
	}

	// Send audio frame - should trigger lazy initialization
	audioFrame := frames.NewAudioFrame([]byte{0x00, 0x01}, 16000, 1)
	service.HandleFrame(ctx, audioFrame, frames.Downstream)

	// Wait for initialization
	time.Sleep(200 * time.Millisecond)

	// Connection should now exist
	if service.conn == nil {
		t.Error("Expected connection after first audio frame (lazy init)")
	}

	service.Cleanup()
}

func TestAudioFramePassedDownstream(t *testing.T) {
	server := startMockWSServer(t, func(conn *websocket.Conn) {
		// Read session config
		_, _, _ = conn.ReadMessage()
		// Read audio
		_, _, _ = conn.ReadMessage()
		time.Sleep(500 * time.Millisecond)
	})
	defer server.Close()

	service := NewSTTService(STTConfig{
		APIKey:  "test-key",
		BaseURL: wsURL(server),
	})

	collector := newMockCollector()
	service.Link(collector)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := service.Start(ctx); err != nil {
		t.Fatalf("Failed to start: %v", err)
	}
	if err := collector.Start(ctx); err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	if err := service.Initialize(ctx); err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Send audio frame
	audioFrame := frames.NewAudioFrame([]byte{0xAA, 0xBB}, 16000, 1)
	service.HandleFrame(ctx, audioFrame, frames.Downstream)

	// Wait for passthrough
	time.Sleep(200 * time.Millisecond)

	collectedFrames := collector.getFrames()
	found := false
	for _, f := range collectedFrames {
		if af, ok := f.(*frames.AudioFrame); ok {
			if len(af.Data) == 2 && af.Data[0] == 0xAA && af.Data[1] == 0xBB {
				found = true
				break
			}
		}
	}

	if !found {
		t.Error("Expected AudioFrame to be passed downstream")
	}

	service.Cleanup()
}
