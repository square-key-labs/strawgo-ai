package transports

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

// WebSocketTransport is a generic WebSocket transport that uses
// an injected serializer for protocol-specific message handling
type WebSocketTransport struct {
	port       int
	path       string
	serializer serializers.FrameSerializer
	inputProc  *WebSocketInputProcessor
	outputProc *WebSocketOutputProcessor
	server     *http.Server
	upgrader   websocket.Upgrader
	conns      map[string]*wsConnection
	connMu     sync.RWMutex
}

type wsConnection struct {
	id      string
	conn    *websocket.Conn
	ctx     context.Context
	cancel  context.CancelFunc
	writeMu sync.Mutex // Protect concurrent writes to WebSocket
}

// WebSocketConfig holds configuration for the WebSocket transport
type WebSocketConfig struct {
	Port       int                         // Port to listen on (e.g., 8080)
	Path       string                      // WebSocket path (e.g., "/ws")
	Serializer serializers.FrameSerializer // Protocol serializer (Twilio, Asterisk, etc.)
}

// NewWebSocketTransport creates a new generic WebSocket transport
func NewWebSocketTransport(config WebSocketConfig) *WebSocketTransport {
	if config.Path == "" {
		config.Path = "/ws"
	}
	if config.Serializer == nil {
		panic("WebSocketTransport requires a serializer")
	}

	t := &WebSocketTransport{
		port:       config.Port,
		path:       config.Path,
		serializer: config.Serializer,
		conns:      make(map[string]*wsConnection),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins (configure based on security needs)
			},
		},
	}

	t.inputProc = newWebSocketInputProcessor(t)
	t.outputProc = newWebSocketOutputProcessor(t)

	return t
}

// Input returns the input processor
func (t *WebSocketTransport) Input() processors.FrameProcessor {
	return t.inputProc
}

// Output returns the output processor
func (t *WebSocketTransport) Output() processors.FrameProcessor {
	return t.outputProc
}

// Start begins listening for WebSocket connections
func (t *WebSocketTransport) Start(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc(t.path, t.handleWebSocket)

	t.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", t.port),
		Handler: mux,
	}

	go func() {
		<-ctx.Done()
		if err := t.server.Shutdown(context.Background()); err != nil {
			log.Printf("WebSocket server shutdown error: %v", err)
		}
	}()

	log.Printf("WebSocket transport listening on %s%s", t.server.Addr, t.path)
	if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("WebSocket server error: %w", err)
	}

	return nil
}

// handleWebSocket upgrades HTTP connections to WebSocket
func (t *WebSocketTransport) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := t.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	// Create connection context
	ctx, cancel := context.WithCancel(context.Background())
	connID := fmt.Sprintf("ws-%p", conn)

	wsConn := &wsConnection{
		id:     connID,
		conn:   conn,
		ctx:    ctx,
		cancel: cancel,
	}

	t.connMu.Lock()
	t.conns[connID] = wsConn
	t.connMu.Unlock()

	defer func() {
		t.connMu.Lock()
		delete(t.conns, connID)
		t.connMu.Unlock()
		cancel()
		conn.Close()
	}()

	log.Printf("WebSocket connection established: %s", connID)

	// Emit ClientConnectedFrame to notify downstream services
	if err := t.inputProc.pushFrame(frames.NewClientConnectedFrame()); err != nil {
		log.Printf("Error pushing ClientConnectedFrame: %v", err)
	}

	// Handle incoming messages
	for {
		select {
		case <-ctx.Done():
			return
		default:
			var data interface{}
			var err error

			// Read message and check ACTUAL WebSocket frame type (not serializer type)
			// This supports hybrid protocols like Asterisk (BINARY for audio, TEXT for control)
			msgType, msgBytes, readErr := conn.ReadMessage()
			if readErr != nil {
				if websocket.IsUnexpectedCloseError(readErr, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("WebSocket read error: %v", readErr)
				}
				// Push EndFrame to notify downstream services to cleanup
				if err := t.inputProc.pushFrame(frames.NewEndFrame()); err != nil {
					log.Printf("Error pushing end frame: %v", err)
				}
				return
			}

			// Convert based on WebSocket message type
			if msgType == websocket.BinaryMessage {
				data = msgBytes
			} else {
				// TEXT message
				data = string(msgBytes)
			}

			// Deserialize using the protocol-specific serializer
			frame, err := t.serializer.Deserialize(data)
			if err != nil {
				log.Printf("Deserialization error: %v", err)
				continue
			}

			if frame == nil {
				// Serializer returned nil (e.g., ignored message type)
				continue
			}

			// Handle different frame types
			switch f := frame.(type) {
			case *frames.AudioFrame:
				// Send audio to input processor
				if err := t.inputProc.pushAudioFrame(f); err != nil {
					log.Printf("Error pushing audio frame: %v", err)
				}

			case *frames.StartFrame:
				// Send start frame
				if err := t.inputProc.pushFrame(f); err != nil {
					log.Printf("Error pushing start frame: %v", err)
				}

			case *frames.EndFrame:
				// Send end frame and close connection
				if err := t.inputProc.pushFrame(f); err != nil {
					log.Printf("Error pushing end frame: %v", err)
				}
				return

			default:
				// Send other frames
				if err := t.inputProc.pushFrame(f); err != nil {
					log.Printf("Error pushing frame: %v", err)
				}
			}
		}
	}
}

// sendMessage sends a serialized message to all active connections
func (t *WebSocketTransport) sendMessage(data interface{}) error {
	t.connMu.RLock()
	defer t.connMu.RUnlock()

	for _, wsConn := range t.conns {
		var err error

		// Protect concurrent writes to the same connection
		wsConn.writeMu.Lock()

		// Determine message type based on actual data type
		// This supports hybrid protocols (e.g., Asterisk: BINARY for audio, TEXT for control)
		switch v := data.(type) {
		case []byte:
			err = wsConn.conn.WriteMessage(websocket.BinaryMessage, v)
		case string:
			// Send as TEXT frame
			log.Printf("[WebSocketTransport] Sending TEXT frame: '%s'", v)
			err = wsConn.conn.WriteMessage(websocket.TextMessage, []byte(v))
		default:
			wsConn.writeMu.Unlock()
			return fmt.Errorf("unsupported data type for WebSocket message: %T", data)
		}

		wsConn.writeMu.Unlock()

		if err != nil {
			log.Printf("Error sending to connection %s: %v", wsConn.id, err)
		}
	}

	return nil
}

// WebSocketInputProcessor handles incoming frames from WebSocket
type WebSocketInputProcessor struct {
	*processors.BaseProcessor
	transport *WebSocketTransport
}

func newWebSocketInputProcessor(transport *WebSocketTransport) *WebSocketInputProcessor {
	p := &WebSocketInputProcessor{
		transport: transport,
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketInput", p)
	return p
}

func (p *WebSocketInputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - configure interruption settings
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
		log.Printf("[WebSocketInput] Interruptions configured: allowed=%v, strategies=%d",
			p.InterruptionsAllowed(), len(p.InterruptionStrategies()))
	}
	// Input processor just passes frames through
	return p.PushFrame(frame, direction)
}

func (p *WebSocketInputProcessor) pushFrame(frame frames.Frame) error {
	return p.BaseProcessor.PushFrame(frame, frames.Downstream)
}

func (p *WebSocketInputProcessor) pushAudioFrame(frame *frames.AudioFrame) error {
	return p.BaseProcessor.PushFrame(frame, frames.Downstream)
}

// audioChunk represents a pre-serialized audio chunk ready to send
type audioChunk struct {
	data         interface{} // Pre-serialized data ([]byte or string)
	chunkSize    int
	sampleRate   int
	sendInterval time.Duration
}

// WebSocketOutputProcessor handles outgoing frames to WebSocket
type WebSocketOutputProcessor struct {
	*processors.BaseProcessor
	transport   *WebSocketTransport
	audioBuffer []byte
	chunkSize   int
	mu          sync.Mutex

	// Rate-limited sender
	chunkQueue   chan *audioChunk
	senderCtx    context.Context
	senderCancel context.CancelFunc
	senderWg     sync.WaitGroup
	cleanupOnce  sync.Once // Ensure cleanup only runs once

	// Track LLM response state for bot speaking detection
	llmResponseEnded bool
	llmMu            sync.Mutex

	// Interruption state - block new audio after interruption
	// Uses context_id to distinguish old vs new audio
	interrupted       bool
	currentContextID  string // The context_id we're currently accepting audio from
	expectedContextID string // The context_id we expect from TTSStartedFrame (set before audio arrives)
	interruptionMu    sync.Mutex

	// Track if cleanup has been done to prevent send on closed channel
	cleanupDone   bool
	cleanupLogged bool // Only log cleanup warning once

	// Track stale audio blocking to avoid log spam
	staleAudioBlockedCount int
	lastStaleContextID     string
}

func newWebSocketOutputProcessor(transport *WebSocketTransport) *WebSocketOutputProcessor {
	p := &WebSocketOutputProcessor{
		transport:   transport,
		audioBuffer: make([]byte, 0),
		chunkSize:   320,                          // Default chunk size (can be configured per codec)
		chunkQueue:  make(chan *audioChunk, 1000), // Larger buffer for streaming TTS
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketOutput", p)

	// Start the rate-limited sender goroutine
	p.senderCtx, p.senderCancel = context.WithCancel(context.Background())
	p.startChunkSender()

	return p
}

// calculateSendInterval computes the real-time pacing interval for audio chunks
// Formula: chunk_duration = chunk_size / (sample_rate * bytes_per_sample)
// For 160-byte chunks at 8kHz: 160/8000 = 0.02s = 20ms
func calculateSendInterval(chunkSize int, sampleRate int) time.Duration {
	if sampleRate == 0 {
		sampleRate = 8000 // Default fallback
	}
	// For telephony codecs: 1 byte per sample
	// For linear16: 2 bytes per sample, but chunk size already accounts for this
	bytesPerSample := 1
	if sampleRate > 8000 {
		bytesPerSample = 2
	}

	// Calculate real-time playback interval: chunk_duration = chunk_size / (sample_rate * bytes_per_sample)
	// Example: 160 bytes / 8000 samples/sec = 0.02 sec = 20ms
	intervalSecs := float64(chunkSize) / float64(sampleRate*bytesPerSample)
	interval := time.Duration(intervalSecs * float64(time.Second))

	// Ensure minimum interval to prevent tight loops
	if interval < time.Millisecond {
		interval = time.Millisecond
	}

	return interval
}

// startChunkSender starts the rate-limited audio sender goroutine
// This goroutine consumes chunks from the queue and sends them with proper pacing
// to prevent overwhelming the WebSocket/Asterisk buffer
// Also implements timeout-based bot speech detection
func (p *WebSocketOutputProcessor) startChunkSender() {
	p.senderWg.Add(1)
	go func() {
		defer p.senderWg.Done()

		var nextSendTime time.Time
		firstChunk := true
		botSpeaking := false

		// BOT_VAD_STOP_SECS = 0.35
		// If no audio chunks for this duration, bot is considered to have stopped speaking
		vadStopDuration := 350 * time.Millisecond
		vadTimer := time.NewTimer(vadStopDuration)
		vadTimer.Stop() // Don't start timer until first chunk

		defer vadTimer.Stop()

		for {
			select {
			case <-p.senderCtx.Done():
				log.Printf("[WebSocketOutput] Sender goroutine stopped")
				return

			case chunk := <-p.chunkQueue:
				// CRITICAL: Check if interrupted before sending - discard chunk if so
				// This prevents sending chunks that were picked up just before/during interruption
				p.interruptionMu.Lock()
				if p.interrupted {
					p.interruptionMu.Unlock()
					logger.Debug("[WebSocketOutput] Sender: discarding chunk - interrupted")
					continue // Skip this chunk, don't send it
				}
				p.interruptionMu.Unlock()

				// Rate-limiting algorithm:
				// current_time = time.monotonic()
				// sleep_duration = max(0, self._next_send_time - current_time)
				// await asyncio.sleep(sleep_duration)
				// if sleep_duration == 0:
				//     self._next_send_time = time.monotonic() + self._send_interval
				// else:
				//     self._next_send_time += self._send_interval

				now := time.Now()

				// First chunk - initialize next send time and start VAD timer
				if firstChunk {
					nextSendTime = now
					firstChunk = false
				}

				// Calculate sleep duration
				sleepDuration := nextSendTime.Sub(now)
				if sleepDuration > 0 {
					time.Sleep(sleepDuration)
				}

				// Send the chunk
				if err := p.transport.sendMessage(chunk.data); err != nil {
					log.Printf("[WebSocketOutput] Error sending chunk: %v", err)
					// Check for broken pipe or connection closed errors - stop sending
					errStr := err.Error()
					if strings.Contains(errStr, "broken pipe") ||
						strings.Contains(errStr, "connection reset") ||
						strings.Contains(errStr, "closed network connection") ||
						strings.Contains(errStr, "use of closed") {
						log.Printf("[WebSocketOutput] 🔴 Connection lost, stopping sender")
						return // Stop the sender goroutine
					}
				}

				// Update next send time
				if sleepDuration <= 0 {
					// We're behind schedule - reset to current time + interval
					nextSendTime = time.Now().Add(chunk.sendInterval)
				} else {
					// We're on schedule - add interval to maintain consistent pacing
					nextSendTime = nextSendTime.Add(chunk.sendInterval)
				}

				// Reset VAD timer
				// If no more chunks arrive within vadStopDuration, emit BotStoppedSpeakingFrame
				if !vadTimer.Stop() {
					select {
					case <-vadTimer.C:
					default:
					}
				}
				vadTimer.Reset(vadStopDuration)

				// Emit BotStartedSpeakingFrame on first audio chunk
				if !botSpeaking {
					log.Printf("[WebSocketOutput] 🎤 Bot started speaking")
					p.PushFrame(frames.NewBotStartedSpeakingFrame(), frames.Upstream)
					botSpeaking = true
				}

			case <-vadTimer.C:
				// Timeout - no audio chunks for vadStopDuration
				// IMPORTANT: Only emit BotStoppedSpeakingFrame if LLM has finished generating
				// This prevents premature stopping while TTS is still processing LLM chunks
				if botSpeaking {
					p.llmMu.Lock()
					llmEnded := p.llmResponseEnded
					p.llmMu.Unlock()

					if llmEnded {
						log.Printf("[WebSocketOutput] 🔇 Bot stopped speaking (no audio for %v, LLM response ended)", vadStopDuration)
						p.PushFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)
						botSpeaking = false
					} else {
						log.Printf("[WebSocketOutput] No audio for %v but LLM still generating, waiting...", vadStopDuration)
						// Reset timer to check again
						vadTimer.Reset(vadStopDuration)
					}
				}
			}
		}
	}()
}

// Cleanup stops the sender goroutine and releases resources
// Safe to call multiple times - only executes once
func (p *WebSocketOutputProcessor) Cleanup() error {
	p.cleanupOnce.Do(func() {
		log.Printf("[WebSocketOutput] Cleaning up sender goroutine")

		// Mark cleanup as done BEFORE closing channel to prevent send on closed channel
		p.mu.Lock()
		p.cleanupDone = true
		p.mu.Unlock()

		if p.senderCancel != nil {
			p.senderCancel()
		}
		p.senderWg.Wait()
		close(p.chunkQueue)
		log.Printf("[WebSocketOutput] Cleanup complete")
	})
	return nil
}

func (p *WebSocketOutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - configure interruption settings
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
		log.Printf("[WebSocketOutput] Interruptions configured: allowed=%v, strategies=%d",
			p.InterruptionsAllowed(), len(p.InterruptionStrategies()))
		// Pass frame downstream
		return p.PushFrame(frame, direction)
	}

	// Handle EndFrame - cleanup sender goroutine and stop processing
	if _, ok := frame.(*frames.EndFrame); ok {
		log.Printf("[WebSocketOutput] Received EndFrame, cleaning up sender goroutine")
		if err := p.Cleanup(); err != nil {
			log.Printf("[WebSocketOutput] Error during cleanup: %v", err)
		}
		// Don't process any more frames after EndFrame
		return nil
	}

	// Handle LLMFullResponseEndFrame - mark that LLM has finished generating
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		p.llmMu.Lock()
		p.llmResponseEnded = true
		p.llmMu.Unlock()
		log.Printf("[WebSocketOutput] LLM response ended - bot will stop speaking after final audio")
		// Pass frame downstream
		return p.PushFrame(frame, direction)
	}

	// Handle TTSStartedFrame - reset LLM response state for new generation
	// CRITICAL: Store the expected context ID from the frame. This tells us exactly
	// which context to accept, preventing old audio from cancelled contexts from
	// being treated as the "new" response.
	// NOTE: We do NOT clear interrupted flag here! The flag is cleared when we
	// receive the first audio frame with the EXPECTED context_id.
	if ttsFrame, ok := frame.(*frames.TTSStartedFrame); ok {
		p.llmMu.Lock()
		p.llmResponseEnded = false
		p.llmMu.Unlock()

		p.interruptionMu.Lock()
		wasInterrupted := p.interrupted
		oldContextID := p.currentContextID
		// Reset currentContextID - will be set when matching audio arrives
		p.currentContextID = ""
		// Store expected context ID from the TTS service
		// Only accept audio frames with this exact context ID
		p.expectedContextID = ttsFrame.ContextID
		// Log summary of blocked stale audio before resetting counters
		if p.staleAudioBlockedCount > 0 {
			log.Printf("[WebSocketOutput] ⛔ Blocked %d stale audio frames from context %s",
				p.staleAudioBlockedCount, p.lastStaleContextID)
		}
		p.staleAudioBlockedCount = 0
		p.lastStaleContextID = ""
		p.interruptionMu.Unlock()

		if wasInterrupted {
			log.Printf("[WebSocketOutput] TTS started - expecting context %s (was %s), keeping interrupted=true", ttsFrame.ContextID, oldContextID)
		} else {
			log.Printf("[WebSocketOutput] TTS started - expecting context %s (was %s)", ttsFrame.ContextID, oldContextID)
		}
		// Pass frame upstream (to aggregators)
		return p.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - clear local buffer, drain queue, and send flush command to server
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		// Check if interruptions are allowed
		if !p.InterruptionsAllowed() {
			log.Printf("[WebSocketOutput] ⚪ Interruptions not allowed, ignoring InterruptionFrame")
			return nil
		}

		log.Printf("[WebSocketOutput] ════════════════════════════════════════════")
		log.Printf("[WebSocketOutput] 🔴 INTERRUPTION SEQUENCE STARTED")
		log.Printf("[WebSocketOutput] ════════════════════════════════════════════")

		// Emit BotStoppedSpeakingFrame if we were speaking
		// This notifies upstream processors that bot audio has stopped
		log.Printf("[WebSocketOutput]   ├─ Step 1: Pushing BotStoppedSpeakingFrame upstream")
		p.PushFrame(frames.NewBotStoppedSpeakingFrame(), frames.Upstream)

		// CRITICAL: Set interrupted flag to block audio from being queued
		// The flag will be cleared when we receive audio with a NEW context_id
		// This ensures old audio (still in pipeline) doesn't slip through
		p.interruptionMu.Lock()
		wasAlreadyInterrupted := p.interrupted
		p.interrupted = true
		oldContextID := p.currentContextID
		log.Printf("[WebSocketOutput]   ├─ Step 2: Set interrupted=true (was=%v, blocking context: %s)", wasAlreadyInterrupted, oldContextID)
		p.interruptionMu.Unlock()

		// Clear local audio buffer
		p.mu.Lock()
		bufferSize := len(p.audioBuffer)
		if bufferSize > 0 {
			log.Printf("[WebSocketOutput]   ├─ Step 3: Clearing local audio buffer (%d bytes)", bufferSize)
			p.audioBuffer = make([]byte, 0)
		} else {
			log.Printf("[WebSocketOutput]   ├─ Step 3: Local audio buffer already empty")
		}
		p.mu.Unlock()

		// Drain the chunk queue (remove all pending chunks)
		log.Printf("[WebSocketOutput]   ├─ Step 4: Draining pending chunk queue...")
		drainedChunks := 0
		drainedBytes := 0
	drainLoop:
		for {
			select {
			case chunk := <-p.chunkQueue:
				drainedChunks++
				drainedBytes += chunk.chunkSize
			default:
				break drainLoop
			}
		}
		if drainedChunks > 0 {
			log.Printf("[WebSocketOutput]   ├─ Step 4: Drained %d pending chunks (%d bytes) from queue", drainedChunks, drainedBytes)
		} else {
			log.Printf("[WebSocketOutput]   ├─ Step 4: Chunk queue already empty")
		}

		// Serialize the interruption frame (serializer knows what commands to send)
		data, err := p.transport.serializer.Serialize(frame)
		if err != nil {
			return fmt.Errorf("serialization error: %w", err)
		}

		if data != nil {
			// Handle both single message and multiple messages (slice)
			if commands, ok := data.([]string); ok {
				// Multiple commands - send each one
				log.Printf("[WebSocketOutput]   └─ Sending %d server-side flush commands", len(commands))
				for _, cmd := range commands {
					log.Printf("[WebSocketOutput]      └─ Sending: %s", cmd)
					if err := p.transport.sendMessage(cmd); err != nil {
						return fmt.Errorf("send error: %w", err)
					}
				}
			} else {
				// Single message - send it
				log.Printf("[WebSocketOutput]   └─ Sending server-side flush command")
				if err := p.transport.sendMessage(data); err != nil {
					return fmt.Errorf("send error: %w", err)
				}
			}
		} else {
			log.Printf("[WebSocketOutput]   └─ No server-side flush command needed")
		}

		log.Printf("[WebSocketOutput] ✓ Interruption handling complete (cleared %d bytes buffer + %d chunks)", bufferSize, drainedChunks)
		return nil
	}

	// Handle TTSAudioFrame with buffering and chunking (TTS output to send to client)
	if audioFrame, ok := frame.(*frames.TTSAudioFrame); ok {
		return p.handleAudioFrame(audioFrame)
	}

	// IMPORTANT: Ignore user's AudioFrames - do NOT send them back to client!
	// User AudioFrames flow through pipeline for interruption detection but should not be echoed back
	if _, ok := frame.(*frames.AudioFrame); ok {
		// Silently consume user's audio - don't send back to phone
		return nil
	}

	// For all other frames, serialize and send normally
	data, err := p.transport.serializer.Serialize(frame)
	if err != nil {
		return fmt.Errorf("serialization error: %w", err)
	}

	if data == nil {
		// Serializer returned nil (frame type not supported for output)
		return nil
	}

	// Send to WebSocket connections
	if err := p.transport.sendMessage(data); err != nil {
		return fmt.Errorf("send error: %w", err)
	}

	return nil
}

func (p *WebSocketOutputProcessor) handleAudioFrame(audioFrame *frames.TTSAudioFrame) error {
	// CRITICAL: Check if cleanup has been done - prevent send on closed channel
	p.mu.Lock()
	if p.cleanupDone {
		// Only log once to avoid spam
		if !p.cleanupLogged {
			log.Printf("[WebSocketOutput] ⛔ Ignoring audio frames - cleanup already done (suppressing further logs)")
			p.cleanupLogged = true
		}
		p.mu.Unlock()
		return nil
	}
	p.mu.Unlock()

	// Get context_id from frame metadata (set by TTS service like Cartesia)
	frameContextID := ""
	if ctxIDRaw, exists := audioFrame.Metadata()["context_id"]; exists {
		if ctxIDStr, ok := ctxIDRaw.(string); ok {
			frameContextID = ctxIDStr
		}
	}

	// CRITICAL: Context-based filtering with expected context ID
	// - TTSStartedFrame sets expectedContextID telling us exactly which context to accept
	// - This prevents old audio from cancelled contexts being accepted as "new" response
	// - Normal flow: Accept audio matching expected/current context
	// - Interruption flow: Block all audio until matching expected context arrives
	p.interruptionMu.Lock()
	isInterrupted := p.interrupted
	currentCtxID := p.currentContextID
	expectedCtxID := p.expectedContextID

	if frameContextID != "" {
		if currentCtxID == "" {
			// Context was reset (by TTSStartedFrame) - waiting for first audio from new TTS response
			// CRITICAL: Only accept if it matches expected context (if set)
			if expectedCtxID != "" && frameContextID != expectedCtxID {
				// This is old audio from a cancelled context - BLOCK IT
				// Only log first occurrence and summary to avoid spam
				if p.lastStaleContextID != frameContextID {
					if p.staleAudioBlockedCount > 0 {
						log.Printf("[WebSocketOutput] ⛔ Blocked %d stale audio frames from context %s",
							p.staleAudioBlockedCount, p.lastStaleContextID)
					}
					log.Printf("[WebSocketOutput] ⛔ BLOCKED old audio (context %s != expected %s)",
						frameContextID, expectedCtxID)
					p.lastStaleContextID = frameContextID
					p.staleAudioBlockedCount = 1
				} else {
					p.staleAudioBlockedCount++
				}
				p.interruptionMu.Unlock()
				return nil
			}

			// Accept this frame - either matches expected or no expected set (backward compat)
			p.currentContextID = frameContextID
			currentCtxID = frameContextID
			if isInterrupted {
				p.interrupted = false
				isInterrupted = false
				log.Printf("[WebSocketOutput] ════════════════════════════════════════════")
				log.Printf("[WebSocketOutput] ✅ INTERRUPTION CLEARED - New context: %s (matched expected)", frameContextID)
				log.Printf("[WebSocketOutput] ════════════════════════════════════════════")
			} else {
				log.Printf("[WebSocketOutput] 📝 New context set: %s", frameContextID)
			}
		} else if isInterrupted && frameContextID != currentCtxID {
			// Different context while interrupted - block old audio
			p.interruptionMu.Unlock()
			log.Printf("[WebSocketOutput] ⛔ BLOCKED old audio during interruption (context %s, waiting for %s)",
				frameContextID, expectedCtxID)
			return nil
		} else if frameContextID != currentCtxID {
			// Different context but not interrupted - this is OLD audio from a previous
			// response that's still in the pipeline. BLOCK IT!
			// Only log first occurrence and summary to avoid spam
			if p.lastStaleContextID != frameContextID {
				if p.staleAudioBlockedCount > 0 {
					log.Printf("[WebSocketOutput] ⛔ Blocked %d stale audio frames from context %s",
						p.staleAudioBlockedCount, p.lastStaleContextID)
				}
				log.Printf("[WebSocketOutput] ⛔ BLOCKED stale audio (context %s != current %s)",
					frameContextID, currentCtxID)
				p.lastStaleContextID = frameContextID
				p.staleAudioBlockedCount = 1
			} else {
				p.staleAudioBlockedCount++
			}
			p.interruptionMu.Unlock()
			return nil
		}
		// else: frameContextID == currentCtxID - same context, allow through
	}
	p.interruptionMu.Unlock()

	// Block audio if still interrupted AND we don't have a valid context yet
	// (shouldn't happen normally since TTSStartedFrame resets context)
	if isInterrupted {
		log.Printf("[WebSocketOutput] ⛔ BLOCKED audio frame (%d bytes) - interrupted, no context (frame: %s)",
			len(audioFrame.Data), frameContextID)
		return nil
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Determine chunk size based on codec
	codec := "linear16"
	if codecRaw, exists := audioFrame.Metadata()["codec"]; exists {
		if codecStr, ok := codecRaw.(string); ok {
			codec = codecStr
		}
	}

	// Set chunk size based on codec
	// For telephony codecs (mulaw/alaw): 160 bytes = 20ms at 8kHz
	// For PCM: 320 bytes = 10ms at 16kHz
	chunkSize := 320
	if codec == "mulaw" || codec == "alaw" {
		chunkSize = 160
	}

	// Calculate send interval for rate limiting
	sendInterval := calculateSendInterval(chunkSize, audioFrame.SampleRate)

	// IMMEDIATE STREAMING MODE:
	// Process THIS frame's data immediately, combining with any small remainder from previous frame
	// This ensures each TTS chunk is sent as soon as it arrives, not accumulated
	currentData := append(p.audioBuffer, audioFrame.Data...)
	p.audioBuffer = make([]byte, 0) // Clear old buffer

	numChunks := 0

	// Chunk and send immediately from current frame
	for len(currentData) >= chunkSize {
		// CRITICAL: Check if interrupted before queuing each chunk
		// This prevents race condition where audio continues to queue during interruption
		p.interruptionMu.Lock()
		if p.interrupted {
			p.interruptionMu.Unlock()
			logger.Debug("[WebSocketOutput] Aborting audio streaming - interrupted")
			p.audioBuffer = make([]byte, 0) // Clear any remainder
			return nil
		}
		p.interruptionMu.Unlock()

		chunk := currentData[:chunkSize]
		currentData = currentData[chunkSize:]
		numChunks++

		// Create a new audio frame for this chunk
		chunkFrame := frames.NewTTSAudioFrame(chunk, audioFrame.SampleRate, audioFrame.Channels)
		// Copy metadata
		for k, v := range audioFrame.Metadata() {
			chunkFrame.SetMetadata(k, v)
		}

		// Pre-serialize the chunk
		data, err := p.transport.serializer.Serialize(chunkFrame)
		if err != nil {
			log.Printf("[WebSocketOutput] Serialization error: %v", err)
			continue
		}

		if data == nil {
			continue
		}

		// BLOCKING send to queue for immediate transmission
		select {
		case p.chunkQueue <- &audioChunk{
			data:         data,
			chunkSize:    chunkSize,
			sampleRate:   audioFrame.SampleRate,
			sendInterval: sendInterval,
		}:
			// Chunk queued successfully
		case <-p.senderCtx.Done():
			// Sender stopped (EndFrame received), abort processing
			log.Printf("[WebSocketOutput] Sender stopped, discarding remaining audio")
			return nil
		}
	}

	// Keep ONLY the small remainder (< chunkSize) for next frame
	// This ensures we don't accumulate large buffers across frames
	p.audioBuffer = currentData

	// Only log for significant chunks (reduces noise)
	if numChunks > 0 {
		logger.Debug("[WebSocketOutput] Streamed %d chunks (%d bytes) immediately (buffer_remainder=%d bytes)",
			numChunks, numChunks*chunkSize, len(p.audioBuffer))
	}

	return nil
}
