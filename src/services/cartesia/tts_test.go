package cartesia

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

func TestCartesiaTTSContextIDGeneration(t *testing.T) {
	service := NewTTSService(TTSConfig{
		APIKey:  "test-key",
		VoiceID: "test-voice",
		Model:   "sonic-3",
	})

	if service.HasActiveAudioContext() {
		t.Errorf("Expected no active audio context before initialization, got: %s", service.GetActiveAudioContextID())
	}

	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()

	if contextID1 == "" || contextID2 == "" {
		t.Error("GenerateContextID returned empty string")
	}

	if contextID1 == contextID2 {
		t.Errorf("GenerateContextID should generate unique IDs, got: %s == %s", contextID1, contextID2)
	}

	if len(contextID1) != 36 {
		t.Errorf("Expected UUID length 36, got: %d", len(contextID1))
	}
	if contextID1[8] != '-' || contextID1[13] != '-' || contextID1[18] != '-' || contextID1[23] != '-' {
		t.Errorf("Invalid UUID format: %s", contextID1)
	}
}

func TestCartesiaTTSContextIDConsistency(t *testing.T) {
	contextID1 := services.GenerateContextID()
	contextID2 := services.GenerateContextID()
	contextID3 := services.GenerateContextID()

	contextIDs := []string{contextID1, contextID2, contextID3}

	for i, id := range contextIDs {
		if id == "" {
			t.Errorf("Context ID %d is empty", i)
		}

		if len(id) != 36 {
			t.Errorf("Context ID %d has wrong length: expected 36, got %d", i, len(id))
		}

		for j := i + 1; j < len(contextIDs); j++ {
			if id == contextIDs[j] {
				t.Errorf("Context IDs %d and %d are identical: %s", i, j, id)
			}
		}
	}
}

func TestCartesiaTTSContextIDPattern(t *testing.T) {
	contextIDs := make([]string, 10)
	for i := 0; i < 10; i++ {
		contextIDs[i] = services.GenerateContextID()
	}

	for i, id := range contextIDs {
		if len(id) != 36 {
			t.Errorf("Expected UUID length 36, got: %d", len(id))
		}
		hyphenCount := strings.Count(id, "-")
		if hyphenCount != 4 {
			t.Errorf("Expected 4 hyphens in UUID, got: %d", hyphenCount)
		}

		for j := i + 1; j < len(contextIDs); j++ {
			if id == contextIDs[j] {
				t.Errorf("Duplicate context IDs found at positions %d and %d: %s", i, j, id)
			}
		}
	}
}

func TestCartesiaTTSUnifiedPattern(t *testing.T) {
	id1 := services.GenerateContextID()
	id2 := services.GenerateContextID()

	if len(id1) != len(id2) {
		t.Errorf("Context ID lengths don't match: id1=%d, id2=%d", len(id1), len(id2))
	}

	for i := 0; i < len(id1); i++ {
		if (id1[i] == '-') != (id2[i] == '-') {
			t.Errorf("Hyphen mismatch at position %d", i)
			break
		}
	}
}
func TestCartesiaTTSContextCleanupOnCompletion(t *testing.T) {
	// Test that context is cleaned up on normal completion (LLMFullResponseEndFrame)
	// not just on interruption

	service := NewTTSService(TTSConfig{
		APIKey:  "test-key",
		VoiceID: "test-voice",
		Model:   "sonic-3",
	})

	// Simulate state changes that would occur during normal completion
	ctx := context.Background()

	// 1. Manually set up state as if synthesis had started
	service.mu.Lock()
	service.isSpeaking = true
	service.mu.Unlock()
	service.SetActiveAudioContextID("test-context-id")

	// 2. Send LLMFullResponseEndFrame - this should reset isSpeaking and contextID
	llmEndFrame := frames.NewLLMFullResponseEndFrame()
	err := service.HandleFrame(ctx, llmEndFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) failed: %v", err)
	}

	// 3. Verify isSpeaking was reset to false and contextID was cleared
	if service.isSpeaking {
		t.Error("Expected isSpeaking to be false after LLMFullResponseEndFrame")
	}
	if service.HasActiveAudioContext() {
		t.Errorf("Expected contextID to be empty after LLMFullResponseEndFrame, got: %s", service.GetActiveAudioContextID())
	}
}

func TestCartesiaTTSContextIDReuse(t *testing.T) {
	// Test that context_id is reused within a single LLM turn
	// (between LLMFullResponseStartFrame and LLMFullResponseEndFrame)

	service := NewTTSService(TTSConfig{
		APIKey:  "test-key",
		VoiceID: "test-voice",
		Model:   "sonic-3",
	})

	ctx := context.Background()

	// 1. Send LLMFullResponseStartFrame - should generate turn context ID
	startFrame := frames.NewLLMFullResponseStartFrame()
	err := service.HandleFrame(ctx, startFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseStartFrame) failed: %v", err)
	}

	// Capture the turn context ID via AudioContextManager
	turnContextID := service.GetTurnContextID()

	if turnContextID == "" {
		t.Error("Expected currentTurnContextID to be set after LLMFullResponseStartFrame")
	}

	// 2. Simulate 3 text chunks by calling GetOrCreateContextID (same as synthesizeText)
	collectedContextIDs := []string{}

	for i := 0; i < 3; i++ {
		// Use AudioContextManager method (same as refactored synthesizeText)
		currentContextID := service.GetOrCreateContextID()

		if currentContextID == "" {
			t.Errorf("Expected contextID to be set for text chunk %d", i)
		}
		collectedContextIDs = append(collectedContextIDs, currentContextID)
	}

	// 3. Verify all 3 text chunks used the SAME context ID
	for i := 1; i < len(collectedContextIDs); i++ {
		if collectedContextIDs[i] != collectedContextIDs[0] {
			t.Errorf("Context ID mismatch: chunk 0 used %s, chunk %d used %s",
				collectedContextIDs[0], i, collectedContextIDs[i])
		}
	}

	// 4. Verify the context ID matches the turn context ID
	if collectedContextIDs[0] != turnContextID {
		t.Errorf("Context ID %s does not match turn context ID %s",
			collectedContextIDs[0], turnContextID)
	}

	// 5. Manually set isSpeaking to true to simulate that synthesis occurred
	service.mu.Lock()
	service.isSpeaking = true
	service.mu.Unlock()

	// 6. Send LLMFullResponseEndFrame - should reset context IDs
	endFrame := frames.NewLLMFullResponseEndFrame()
	err = service.HandleFrame(ctx, endFrame, frames.Downstream)
	if err != nil {
		t.Fatalf("HandleFrame(LLMFullResponseEndFrame) failed: %v", err)
	}

	// 7. Verify context IDs were reset via AudioContextManager
	if service.HasActiveAudioContext() {
		t.Errorf("Expected contextID to be reset after LLMFullResponseEndFrame, got: %s", service.GetActiveAudioContextID())
	}
	if service.GetTurnContextID() != "" {
		t.Errorf("Expected currentTurnContextID to be reset after LLMFullResponseEndFrame, got: %s", service.GetTurnContextID())
	}
}

func testDialWebSocket(url string) func() (*websocket.Conn, error) {
	return func() (*websocket.Conn, error) {
		conn, _, err := websocket.DefaultDialer.Dial(url, nil)
		return conn, err
	}
}

func testServiceWithContext() *TTSService {
	s := NewTTSService(TTSConfig{APIKey: "test-key", VoiceID: "test-voice", Model: "sonic-3"})
	s.ctx, s.cancel = context.WithCancel(context.Background())
	return s
}

func closeTestService(s *TTSService) {
	if s.cancel != nil {
		s.cancel()
	}
	s.wsMu.Lock()
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}
	s.wsMu.Unlock()
}

func TestWriteJSONReconnectsOnNilConn(t *testing.T) {
	upgrader := websocket.Upgrader{}
	received := make(chan map[string]interface{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		var msg map[string]interface{}
		if err := conn.ReadJSON(&msg); err == nil {
			received <- msg
		}
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	s := testServiceWithContext()
	defer closeTestService(s)
	s.dialFunc = testDialWebSocket(wsURL)
	s.conn = nil

	msg := map[string]interface{}{"type": "test", "value": "hello"}
	if err := s.writeJSON(msg); err != nil {
		t.Fatalf("writeJSON failed: %v", err)
	}

	s.wsMu.Lock()
	conn := s.conn
	s.wsMu.Unlock()
	if conn == nil {
		t.Fatal("expected writeJSON to reconnect and install conn")
	}

	select {
	case got := <-received:
		if got["type"] != "test" || got["value"] != "hello" {
			t.Fatalf("unexpected message received: %#v", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for server to receive message")
	}
}

func TestWriteJSONBestEffortFailsOnNilConn(t *testing.T) {
	s := testServiceWithContext()
	defer closeTestService(s)

	dialCalls := 0
	s.dialFunc = func() (*websocket.Conn, error) {
		dialCalls++
		return nil, nil
	}
	s.conn = nil

	err := s.writeJSONBestEffort(map[string]interface{}{"type": "cancel"})
	if err == nil {
		t.Fatal("expected writeJSONBestEffort to fail on nil conn")
	}
	if !strings.Contains(err.Error(), "not established") {
		t.Fatalf("expected error to contain 'not established', got: %v", err)
	}
	if dialCalls != 0 {
		t.Fatalf("expected no reconnect attempt, dial calls=%d", dialCalls)
	}
	if s.conn != nil {
		t.Fatal("expected connection to remain nil")
	}
}

func TestReceiveAudioPointerGuard(t *testing.T) {
	upgrader := websocket.Upgrader{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer conn.Close()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	conn1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("failed to dial conn1: %v", err)
	}
	defer conn1.Close()
	conn2, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("failed to dial conn2: %v", err)
	}
	defer conn2.Close()

	s := testServiceWithContext()
	defer closeTestService(s)

	s.wsMu.Lock()
	s.conn = conn1
	myConn := s.conn
	s.conn = conn2
	s.wsMu.Unlock()

	_ = conn1.Close()

	s.wsMu.Lock()
	if s.conn == myConn {
		s.conn = nil
	}
	current := s.conn
	s.wsMu.Unlock()

	if current != conn2 {
		t.Fatal("expected pointer guard to preserve newer conn2")
	}
}

func TestReconnectLockedDiscardsOnShutdown(t *testing.T) {
	upgrader := websocket.Upgrader{}
	accepted := make(chan struct{}, 1)
	closed := make(chan struct{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		accepted <- struct{}{}
		defer conn.Close()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				closed <- struct{}{}
				return
			}
		}
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	s := testServiceWithContext()
	defer closeTestService(s)
	s.dialFunc = testDialWebSocket(wsURL)
	s.cancel()

	s.wsMu.Lock()
	err := s.reconnectLocked()
	s.wsMu.Unlock()

	if err == nil || !strings.Contains(err.Error(), "shutting down") {
		t.Fatalf("expected shutdown discard error, got: %v", err)
	}
	if s.conn != nil {
		t.Fatal("expected s.conn to remain nil")
	}

	select {
	case <-accepted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for test server accept")
	}

	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for discarded conn to close")
	}
}

func TestReconnectLockedConcurrentDial(t *testing.T) {
	upgrader := websocket.Upgrader{}
	acceptedIDs := make(chan int, 4)
	closedIDs := make(chan int, 4)
	var mu sync.Mutex
	nextID := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		mu.Lock()
		nextID++
		id := nextID
		mu.Unlock()
		acceptedIDs <- id
		defer conn.Close()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				closedIDs <- id
				return
			}
		}
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")
	s := testServiceWithContext()
	defer closeTestService(s)

	dialStarted := make(chan struct{})
	allowDialReturn := make(chan struct{})
	s.dialFunc = func() (*websocket.Conn, error) {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		if err != nil {
			return nil, err
		}
		close(dialStarted)
		<-allowDialReturn
		return conn, nil
	}

	var wg sync.WaitGroup
	var errA error
	wg.Add(1)
	go func() {
		defer wg.Done()
		s.wsMu.Lock()
		errA = s.reconnectLocked()
		s.wsMu.Unlock()
	}()

	select {
	case <-dialStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for reconnect dial to start")
	}

	installedConn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("failed to dial installed conn: %v", err)
	}

	s.wsMu.Lock()
	s.conn = installedConn
	s.wsMu.Unlock()

	close(allowDialReturn)
	wg.Wait()

	if errA != nil {
		t.Fatalf("expected reconnectLocked to return nil, got: %v", errA)
	}

	s.wsMu.Lock()
	current := s.conn
	s.wsMu.Unlock()
	if current != installedConn {
		t.Fatal("expected concurrently installed conn to remain active")
	}

	seenClosed := false
	deadline := time.After(2 * time.Second)
	for !seenClosed {
		select {
		case <-closedIDs:
			seenClosed = true
		case <-deadline:
			t.Fatal("timed out waiting for reconnectLocked to close its redundant conn")
		}
	}
}
