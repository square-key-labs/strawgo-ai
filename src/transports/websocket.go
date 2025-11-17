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
	interrupted    bool
	interruptionMu sync.Mutex

	// Track if cleanup has been done to prevent send on closed channel
	cleanupDone bool
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
				// If no more chunks arrive within vadStopDuration, emit TTSStoppedFrame
				if !vadTimer.Stop() {
					select {
					case <-vadTimer.C:
					default:
					}
				}
				vadTimer.Reset(vadStopDuration)
				botSpeaking = true

			case <-vadTimer.C:
				// Timeout - no audio chunks for vadStopDuration
				// IMPORTANT: Only emit TTSStoppedFrame if LLM has finished generating
				// This prevents premature stopping while TTS is still processing LLM chunks
				if botSpeaking {
					p.llmMu.Lock()
					llmEnded := p.llmResponseEnded
					p.llmMu.Unlock()

					if llmEnded {
						log.Printf("[WebSocketOutput] No audio for %v (LLM response ended), emitting TTSStoppedFrame", vadStopDuration)
						p.PushFrame(frames.NewTTSStoppedFrame(), frames.Upstream)
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
	if _, ok := frame.(*frames.TTSStartedFrame); ok {
		p.llmMu.Lock()
		p.llmResponseEnded = false
		p.llmMu.Unlock()

		// CRITICAL: Reset interrupted flag to allow new audio to be queued
		p.interruptionMu.Lock()
		wasInterrupted := p.interrupted
		p.interrupted = false
		p.interruptionMu.Unlock()

		if wasInterrupted {
			log.Printf("[WebSocketOutput] TTS started - reset LLM response state AND cleared interrupted flag")
		} else {
			log.Printf("[WebSocketOutput] TTS started - reset LLM response state")
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

		log.Printf("[WebSocketOutput] 🔴 INTERRUPTION RECEIVED - Starting audio buffer flush")

		// CRITICAL: Set interrupted flag to block new audio from being queued
		p.interruptionMu.Lock()
		p.interrupted = true
		log.Printf("[WebSocketOutput]   ├─ Set interrupted=true (blocking new audio)")
		p.interruptionMu.Unlock()

		// Clear local audio buffer
		p.mu.Lock()
		bufferSize := len(p.audioBuffer)
		if bufferSize > 0 {
			log.Printf("[WebSocketOutput]   ├─ Clearing local audio buffer (%d bytes)", bufferSize)
			p.audioBuffer = make([]byte, 0)
		} else {
			log.Printf("[WebSocketOutput]   ├─ Local audio buffer already empty")
		}
		p.mu.Unlock()

		// Drain the chunk queue (remove all pending chunks)
		log.Printf("[WebSocketOutput]   ├─ Draining pending chunk queue...")
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
			log.Printf("[WebSocketOutput]   ├─ Drained %d pending chunks (%d bytes) from queue", drainedChunks, drainedBytes)
		} else {
			log.Printf("[WebSocketOutput]   ├─ Chunk queue already empty")
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
		p.mu.Unlock()
		log.Printf("[WebSocketOutput] ⛔ Ignoring audio frame - cleanup already done")
		return nil
	}
	p.mu.Unlock()

	// CRITICAL: Check if we're in interrupted state - block new audio
	p.interruptionMu.Lock()
	isInterrupted := p.interrupted
	p.interruptionMu.Unlock()

	if isInterrupted {
		log.Printf("[WebSocketOutput] ⛔ BLOCKED audio frame (%d bytes) - interrupted state active", len(audioFrame.Data))
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
		log.Printf("[WebSocketOutput] ⚡ Streamed %d chunks (%d bytes) immediately (buffer_remainder=%d bytes)",
			numChunks, numChunks*chunkSize, len(p.audioBuffer))
	}

	return nil
}
