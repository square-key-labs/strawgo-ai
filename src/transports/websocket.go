package transports

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
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
	port               int
	path               string
	log                *logger.Logger
	serializer         serializers.FrameSerializer
	playbackAckTimeout time.Duration
	inputProc          *WebSocketInputProcessor
	outputProc         *WebSocketOutputProcessor
	server             *http.Server
	upgrader           websocket.Upgrader
	conns              map[string]*wsConnection
	connMu             sync.RWMutex

	// playbackKind: transport-declared playback classification. Defaults to
	// PlaybackNetworkBlind; set via SetPlaybackKind for local audio sinks.
	playbackKind atomic.Int32
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
	Port               int                         // Port to listen on (e.g., 8080)
	Path               string                      // WebSocket path (e.g., "/ws")
	Serializer         serializers.FrameSerializer // Protocol serializer (Twilio, Asterisk, etc.)
	PlaybackAckTimeout time.Duration               // Fallback timeout when playout ack is expected but never arrives
}

// NewWebSocketTransport creates a new generic WebSocket transport
func NewWebSocketTransport(config WebSocketConfig) *WebSocketTransport {
	if config.Path == "" {
		config.Path = "/ws"
	}
	if config.Serializer == nil {
		panic("WebSocketTransport requires a serializer")
	}
	if config.PlaybackAckTimeout <= 0 {
		config.PlaybackAckTimeout = 3 * time.Second
	}

	t := &WebSocketTransport{
		port:               config.Port,
		path:               config.Path,
		log:                logger.WithPrefix("WebSocketTransport"),
		serializer:         config.Serializer,
		playbackAckTimeout: config.PlaybackAckTimeout,
		conns:              make(map[string]*wsConnection),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins (configure based on security needs)
			},
		},
	}

	t.playbackKind.Store(int32(PlaybackNetworkBlind))
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

// PlaybackKind returns the transport's declared playback class. Implements
// PlaybackClassifier so the output processor can resolve the strategy.
func (t *WebSocketTransport) PlaybackKind() PlaybackKind {
	return PlaybackKind(t.playbackKind.Load())
}

// SetPlaybackKind declares how this transport relates server send-complete to
// client playback-complete. Use PlaybackLocal for in-process speaker sinks
// where send-complete is effectively play-complete. Defaults to
// PlaybackNetworkBlind.
func (t *WebSocketTransport) SetPlaybackKind(kind PlaybackKind) {
	t.playbackKind.Store(int32(kind))
}

// RegisterPlaybackAckHandler forwards to the output processor. See
// WebSocketOutputProcessor.RegisterPlaybackAckHandler.
func (t *WebSocketTransport) RegisterPlaybackAckHandler() {
	t.outputProc.RegisterPlaybackAckHandler()
}

// UnregisterPlaybackAckHandler forwards to the output processor.
func (t *WebSocketTransport) UnregisterPlaybackAckHandler() {
	t.outputProc.UnregisterPlaybackAckHandler()
}

// TriggerPlaybackComplete forwards to the output processor. Call when your
// custom integration has confirmed client-side playback drain.
func (t *WebSocketTransport) TriggerPlaybackComplete() {
	t.outputProc.TriggerPlaybackComplete()
}

// SetDrainPad forwards to the output processor.
func (t *WebSocketTransport) SetDrainPad(d time.Duration) {
	t.outputProc.SetDrainPad(d)
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
			t.log.Warn("WebSocket server shutdown error: %v", err)
		}
	}()

	t.log.Info("Listening on %s%s", t.server.Addr, t.path)
	if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("WebSocket server error: %w", err)
	}

	return nil
}

// handleWebSocket upgrades HTTP connections to WebSocket
func (t *WebSocketTransport) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := t.upgrader.Upgrade(w, r, nil)
	if err != nil {
		t.log.Warn("WebSocket upgrade error: %v", err)
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

	t.log.Info("Connection established: %s", connID)

	// Emit ClientConnectedFrame to notify downstream services
	if err := t.inputProc.pushFrame(frames.NewClientConnectedFrame()); err != nil {
		t.log.Error("Error pushing ClientConnectedFrame: %v", err)
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
					t.log.Warn("WebSocket read error: %v", readErr)
				}
				// Push EndFrame to notify downstream services to cleanup
				if err := t.inputProc.pushFrame(frames.NewEndFrame()); err != nil {
					t.log.Error("Error pushing end frame: %v", err)
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
				t.log.Warn("Deserialization error: %v", err)
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
					t.log.Error("Error pushing audio frame: %v", err)
				}

			case *frames.StartFrame:
				// Send start frame
				if err := t.inputProc.pushFrame(f); err != nil {
					t.log.Error("Error pushing start frame: %v", err)
				}

			case *frames.EndFrame:
				// Send end frame and close connection
				if err := t.inputProc.pushFrame(f); err != nil {
					t.log.Error("Error pushing end frame: %v", err)
				}
				return

			default:
				// Send other frames
				if err := t.inputProc.pushFrame(f); err != nil {
					t.log.Error("Error pushing frame: %v", err)
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
			t.log.Debug("Sending TEXT frame: '%s'", v)
			err = wsConn.conn.WriteMessage(websocket.TextMessage, []byte(v))
		default:
			wsConn.writeMu.Unlock()
			return fmt.Errorf("unsupported data type for WebSocket message: %T", data)
		}

		wsConn.writeMu.Unlock()

		if err != nil {
			t.log.Debug("Error sending to connection %s: %v", wsConn.id, err)
		}
	}

	return nil
}

// WebSocketInputProcessor handles incoming frames from WebSocket
type WebSocketInputProcessor struct {
	*processors.BaseProcessor
	transport *WebSocketTransport
	log       *logger.Logger
}

func newWebSocketInputProcessor(transport *WebSocketTransport) *WebSocketInputProcessor {
	p := &WebSocketInputProcessor{
		transport: transport,
		log:       logger.WithPrefix("WebSocketInputProcessor"),
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketInput", p)
	return p
}

func (p *WebSocketInputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - configure interruption settings
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
		p.log.Info("Interruptions configured: allowed=%v, strategies=%d",
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

// WebSocketOutputProcessor handles outgoing frames to WebSocket. It is
// a thin glue layer that wires together three submodules:
//   - Pacer: chunks audio and rate-limits the wire send
//   - BotSpeechTracker: emits BotStarted/Stopped frames upstream
//   - PlaybackAckTracker: resolves "playback complete" via the chosen
//     strategy (local / user-ack / serializer-ack / drain-pad)
//
// Frame routing logic, server-side flush command emission, and the
// userAck/drainPad configuration knobs live here. State machines live
// in the submodules. See ADR-0001 and the per-file headers.
type WebSocketOutputProcessor struct {
	*processors.BaseProcessor
	transport *WebSocketTransport
	log       *logger.Logger
	mu        sync.Mutex

	pacer            *Pacer
	botSpeechTracker *BotSpeechTracker
	ackTracker       *PlaybackAckTracker

	// Interruption flag — set on InterruptionFrame, cleared on TTSStartedFrame.
	// Gates inbound PlaybackCompleteFrame handling during the interrupt
	// window so a stale ack from a canceled utterance can't drive
	// BotStoppedSpeakingFrame in the next one. PlaybackAckTracker also
	// drains its own state on Cancel; this flag is the cheaper short-
	// circuit for HandleFrame.
	interrupted    bool
	interruptionMu sync.Mutex

	cleanupDone   bool
	cleanupLogged bool

	// userAckRegistered: when true, the application supplies playback-complete
	// signals via TriggerPlaybackComplete. Takes precedence over the transport
	// serializer's PlaybackAckSerializer capability.
	userAckRegistered atomic.Bool

	// drainPadNanos: delay (nanoseconds, atomic for lock-free read) applied after
	// send-complete for network-blind transports with no ack available.
	drainPadNanos atomic.Int64
}

// Sentinel correlation IDs used on playbackDoneChan for paths that do not
// generate a transport-level correlation (user-supplied acks, drain pad).
// The sender goroutine sets pendingPlaybackCorrelationID to the matching
// sentinel so the channel receiver accepts the signal.
const (
	correlationUserAck  = "user-ack"
	correlationDrainPad = "drain-pad"
)

func newWebSocketOutputProcessor(transport *WebSocketTransport) *WebSocketOutputProcessor {
	p := &WebSocketOutputProcessor{
		transport: transport,
		log:       logger.WithPrefix("WebSocketOutputProcessor"),
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketOutput", p)
	p.drainPadNanos.Store(int64(DefaultDrainPad))

	p.pacer = NewPacer(transport.sendMessage, p.log)
	p.ackTracker = NewPlaybackAckTracker(playbackTransportFns{
		send:             transport.sendMessage,
		serializer:       transport.serializer,
		resolveStrategy:  p.resolvePlaybackStrategy,
		drainPad:         func() time.Duration { return time.Duration(p.drainPadNanos.Load()) },
		fallbackDuration: transport.playbackAckTimeout,
		log:              p.log,
	})
	p.botSpeechTracker = NewBotSpeechTracker(
		p.pacer.Events(),
		p.ackTracker,
		func(f frames.Frame, d frames.FrameDirection) error { return p.PushFrame(f, d) },
		p.log,
	)

	return p
}

// RegisterPlaybackAckHandler marks that the application will supply its own
// playback-complete signal via TriggerPlaybackComplete. Takes precedence over
// the transport serializer's PlaybackAckSerializer. Use for custom
// integrations (data-channel acks, proprietary hardware) not covered by
// built-in serializers.
func (p *WebSocketOutputProcessor) RegisterPlaybackAckHandler() {
	p.userAckRegistered.Store(true)
}

// UnregisterPlaybackAckHandler reverts to transport-level playback resolution.
func (p *WebSocketOutputProcessor) UnregisterPlaybackAckHandler() {
	p.userAckRegistered.Store(false)
}

// TriggerPlaybackComplete signals client-side playback has finished. Safe to
// call from any goroutine. Only effective when RegisterPlaybackAckHandler has
// been called; otherwise the ack tracker is not waiting on a user trigger.
func (p *WebSocketOutputProcessor) TriggerPlaybackComplete() {
	p.ackTracker.UserTrigger()
}

// SetDrainPad configures the delay applied after send-complete for
// network-blind transports (no ack available). Defaults to DefaultDrainPad.
func (p *WebSocketOutputProcessor) SetDrainPad(d time.Duration) {
	if d < 0 {
		d = 0
	}
	p.drainPadNanos.Store(int64(d))
}

// playbackStrategy selects how the sender resolves playback completion.
type playbackStrategy int

const (
	stratUserAck playbackStrategy = iota
	stratSerializerAck
	stratLocal
	stratDrainPad
)

// resolvePlaybackStrategy picks the playback-completion strategy in order:
//  1. User-registered ack hook (RegisterPlaybackAckHandler)
//  2. Transport serializer PlaybackAckSerializer (Twilio mark, Asterisk event)
//  3. Transport-declared PlaybackLocal (speaker output)
//  4. Drain-pad fallback (conservative default for unknown network transports)
func (p *WebSocketOutputProcessor) resolvePlaybackStrategy() playbackStrategy {
	if p.userAckRegistered.Load() {
		return stratUserAck
	}
	if _, ok := p.transport.serializer.(serializers.PlaybackAckSerializer); ok {
		return stratSerializerAck
	}
	if p.transport.PlaybackKind() == PlaybackLocal {
		return stratLocal
	}
	return stratDrainPad
}

// Cleanup stops the trackers and the Pacer in dependency order. Safe to
// call multiple times.
func (p *WebSocketOutputProcessor) Cleanup() error {
	p.mu.Lock()
	if p.cleanupDone {
		p.mu.Unlock()
		return nil
	}
	p.cleanupDone = true
	p.mu.Unlock()

	p.log.Info("Cleaning up output processor")
	if p.botSpeechTracker != nil {
		p.botSpeechTracker.Stop()
	}
	if p.ackTracker != nil {
		p.ackTracker.Stop()
	}
	if p.pacer != nil {
		p.pacer.Cleanup()
	}
	p.log.Info("Cleanup complete")
	return nil
}

func (p *WebSocketOutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame - configure interruption settings
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
		p.log.Info("Interruptions configured: allowed=%v, strategies=%d",
			p.InterruptionsAllowed(), len(p.InterruptionStrategies()))
		// Pass frame downstream
		return p.PushFrame(frame, direction)
	}

	// Handle EndFrame - cleanup sender goroutine and stop processing
	if _, ok := frame.(*frames.EndFrame); ok {
		p.log.Info("Received EndFrame, cleaning up sender goroutine")
		if err := p.Cleanup(); err != nil {
			p.log.Warn("Error during cleanup: %v", err)
		}
		// Don't process any more frames after EndFrame
		return nil
	}

	// Handle LLMFullResponseEndFrame - propagate the LLM-end gate to the
	// bot-speech tracker so PacerIdle can resolve to a stop.
	if _, ok := frame.(*frames.LLMFullResponseEndFrame); ok {
		p.botSpeechTracker.LLMResponseEnded()
		p.log.Info("LLM response ended - bot will stop speaking after final audio")
		return p.PushFrame(frame, direction)
	}

	// Handle PlaybackCompleteFrame - forward to the ack tracker. Frame is
	// transport-internal and does not propagate.
	if playbackComplete, ok := frame.(*frames.PlaybackCompleteFrame); ok {
		p.interruptionMu.Lock()
		isInterrupted := p.interrupted
		p.interruptionMu.Unlock()
		if isInterrupted {
			p.log.Debug("Ignoring playback completion signal while interrupted")
			return nil
		}
		correlationID := ""
		if value, ok := playbackComplete.Metadata()["correlation_id"].(string); ok {
			correlationID = value
		}
		p.ackTracker.Notify(correlationID)
		return nil
	}

	// Handle TTSStartedFrame - reset LLM gate, clear interrupt latch, and
	// re-engage the Pacer if it was previously drained.
	if ttsFrame, ok := frame.(*frames.TTSStartedFrame); ok {
		p.botSpeechTracker.ResetLLMState()

		p.interruptionMu.Lock()
		p.interrupted = false
		p.interruptionMu.Unlock()

		p.pacer.Resume()

		p.log.Info("TTS started - generation %s", ttsFrame.ContextID)
		return p.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame - drain Pacer, signal trackers, and send
	// any server-side flush commands.
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		if !p.InterruptionsAllowed() {
			p.log.Debug("Interruptions not allowed, ignoring InterruptionFrame")
			return nil
		}

		p.log.Info("Interruption sequence started")

		p.interruptionMu.Lock()
		p.interrupted = true
		p.interruptionMu.Unlock()

		p.botSpeechTracker.Interrupt()
		p.pacer.Drain()

		data, err := p.transport.serializer.Serialize(frame)
		if err != nil {
			return fmt.Errorf("serialization error: %w", err)
		}

		if data != nil {
			if commands, ok := data.([]string); ok {
				p.log.Debug("Sending %d server-side flush commands", len(commands))
				for _, cmd := range commands {
					if err := p.transport.sendMessage(cmd); err != nil {
						return fmt.Errorf("send error: %w", err)
					}
				}
			} else {
				if err := p.transport.sendMessage(data); err != nil {
					return fmt.Errorf("send error: %w", err)
				}
			}
		}

		p.log.Info("Interruption handling complete")
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
	p.mu.Lock()
	if p.cleanupDone {
		if !p.cleanupLogged {
			p.log.Debug("Ignoring audio frames - cleanup already done (suppressing further logs)")
			p.cleanupLogged = true
		}
		p.mu.Unlock()
		return nil
	}
	p.mu.Unlock()

	// Stale-audio filtering by Generation lives in the AudioGuard processor
	// (see ADR-0001). Chunking, pacing, and the rate-limited sender live in
	// the Pacer (see src/transports/pacer.go). The transport just delegates.
	return p.pacer.Submit(audioFrame, p.transport.serializer.Serialize)
}
