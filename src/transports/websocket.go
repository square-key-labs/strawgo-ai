package transports

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"

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
	id     string
	conn   *websocket.Conn
	ctx    context.Context
	cancel context.CancelFunc
}

// WebSocketConfig holds configuration for the WebSocket transport
type WebSocketConfig struct {
	Port       int                           // Port to listen on (e.g., 8080)
	Path       string                        // WebSocket path (e.g., "/ws")
	Serializer serializers.FrameSerializer  // Protocol serializer (Twilio, Asterisk, etc.)
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

			// Read based on serializer type
			if t.serializer.Type() == serializers.SerializerTypeBinary {
				_, msgBytes, readErr := conn.ReadMessage()
				if readErr != nil {
					if websocket.IsUnexpectedCloseError(readErr, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
						log.Printf("WebSocket read error: %v", readErr)
					}
					return
				}
				data = msgBytes
			} else {
				// Text/JSON mode
				_, msgBytes, readErr := conn.ReadMessage()
				if readErr != nil {
					if websocket.IsUnexpectedCloseError(readErr, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
						log.Printf("WebSocket read error: %v", readErr)
					}
					return
				}
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
		if t.serializer.Type() == serializers.SerializerTypeBinary {
			// Send binary data
			if bytes, ok := data.([]byte); ok {
				err = wsConn.conn.WriteMessage(websocket.BinaryMessage, bytes)
			} else {
				return fmt.Errorf("expected []byte for binary serializer, got %T", data)
			}
		} else {
			// Send text data
			if str, ok := data.(string); ok {
				err = wsConn.conn.WriteMessage(websocket.TextMessage, []byte(str))
			} else {
				return fmt.Errorf("expected string for text serializer, got %T", data)
			}
		}

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
	// Input processor just passes frames through
	return p.PushFrame(frame, direction)
}

func (p *WebSocketInputProcessor) pushFrame(frame frames.Frame) error {
	return p.BaseProcessor.PushFrame(frame, frames.Downstream)
}

func (p *WebSocketInputProcessor) pushAudioFrame(frame *frames.AudioFrame) error {
	return p.BaseProcessor.PushFrame(frame, frames.Downstream)
}

// WebSocketOutputProcessor handles outgoing frames to WebSocket
type WebSocketOutputProcessor struct {
	*processors.BaseProcessor
	transport *WebSocketTransport
}

func newWebSocketOutputProcessor(transport *WebSocketTransport) *WebSocketOutputProcessor {
	p := &WebSocketOutputProcessor{
		transport: transport,
	}
	p.BaseProcessor = processors.NewBaseProcessor("WebSocketOutput", p)
	return p
}

func (p *WebSocketOutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Serialize the frame using the protocol-specific serializer
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
