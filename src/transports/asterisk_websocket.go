package transports

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/gorilla/websocket"
)

// AsteriskWebSocketTransport handles WebSocket connections from Asterisk
type AsteriskWebSocketTransport struct {
	port         int
	inputProc    *AsteriskInputProcessor
	outputProc   *AsteriskOutputProcessor
	server       *http.Server
	upgrader     websocket.Upgrader
	connections  map[string]*asteriskConnection
	connectionMu sync.RWMutex
}

type asteriskConnection struct {
	id     string
	conn   *websocket.Conn
	ctx    context.Context
	cancel context.CancelFunc
}

// AsteriskWebSocketConfig holds configuration for Asterisk WebSocket
type AsteriskWebSocketConfig struct {
	Port int // Port to listen on (e.g., 8080)
}

// NewAsteriskWebSocketTransport creates a new Asterisk WebSocket transport
func NewAsteriskWebSocketTransport(config AsteriskWebSocketConfig) *AsteriskWebSocketTransport {
	t := &AsteriskWebSocketTransport{
		port:        config.Port,
		connections: make(map[string]*asteriskConnection),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins (configure based on security needs)
			},
		},
	}

	t.inputProc = newAsteriskInputProcessor(t)
	t.outputProc = newAsteriskOutputProcessor(t)

	return t
}

// Input returns the input processor that receives audio from Asterisk
func (t *AsteriskWebSocketTransport) Input() processors.FrameProcessor {
	return t.inputProc
}

// Output returns the output processor that sends audio to Asterisk
func (t *AsteriskWebSocketTransport) Output() processors.FrameProcessor {
	return t.outputProc
}

// Start begins listening for WebSocket connections
func (t *AsteriskWebSocketTransport) Start(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/media/", t.handleWebSocket)

	t.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", t.port),
		Handler: mux,
	}

	go func() {
		log.Printf("[AsteriskWS] Listening on port %d", t.port)
		if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[AsteriskWS] Server error: %v", err)
		}
	}()

	return nil
}

// Stop gracefully stops the transport
func (t *AsteriskWebSocketTransport) Stop() error {
	// Close all connections
	t.connectionMu.Lock()
	for _, conn := range t.connections {
		conn.cancel()
		conn.conn.Close()
	}
	t.connections = make(map[string]*asteriskConnection)
	t.connectionMu.Unlock()

	// Shutdown server
	if t.server != nil {
		return t.server.Shutdown(context.Background())
	}
	return nil
}

func (t *AsteriskWebSocketTransport) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	// Extract connection ID from URL path
	connectionID := r.URL.Path[len("/media/"):]

	log.Printf("[AsteriskWS] New connection: %s from %s", connectionID, r.RemoteAddr)

	conn, err := t.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[AsteriskWS] Failed to upgrade connection: %v", err)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	astConn := &asteriskConnection{
		id:     connectionID,
		conn:   conn,
		ctx:    ctx,
		cancel: cancel,
	}

	t.connectionMu.Lock()
	t.connections[connectionID] = astConn
	t.connectionMu.Unlock()

	// Start receiving audio
	go t.receiveAudio(astConn)
}

func (t *AsteriskWebSocketTransport) receiveAudio(conn *asteriskConnection) {
	defer func() {
		conn.cancel()
		conn.conn.Close()
		t.connectionMu.Lock()
		delete(t.connections, conn.id)
		t.connectionMu.Unlock()
		log.Printf("[AsteriskWS] Connection closed: %s", conn.id)
	}()

	for {
		select {
		case <-conn.ctx.Done():
			return
		default:
			messageType, message, err := conn.conn.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("[AsteriskWS] Read error: %v", err)
				}
				return
			}

			if messageType == websocket.BinaryMessage {
				// Binary frame = audio data (ulaw/alaw)
				// Asterisk sends 160-byte packets for ulaw/alaw at 8kHz

				// Pass mulaw directly (no conversion)
				// Services like Deepgram can accept mulaw natively
				audioFrame := frames.NewAudioFrame(message, 8000, 1)
				audioFrame.SetMetadata("connection_id", conn.id)
				audioFrame.SetMetadata("codec", "mulaw") // Mark as mulaw for services

				// Push to input processor
				if err := t.inputProc.QueueFrame(audioFrame, frames.Downstream); err != nil {
					log.Printf("[AsteriskWS] Error queuing audio frame: %v", err)
				}
			} else if messageType == websocket.TextMessage {
				// Text frame = control event
				log.Printf("[AsteriskWS] Control message: %s", string(message))
				// Handle control messages like START_MEDIA_BUFFERING, STOP_MEDIA_BUFFERING, etc.
			}
		}
	}
}

func (t *AsteriskWebSocketTransport) sendAudio(connectionID string, audioData []byte, codec string) error {
	t.connectionMu.RLock()
	conn, exists := t.connections[connectionID]
	t.connectionMu.RUnlock()

	if !exists {
		return fmt.Errorf("connection not found: %s", connectionID)
	}

	var mulawData []byte

	// Convert to mulaw if needed
	if codec == "mulaw" || codec == "ulaw" {
		// Already mulaw, use directly
		mulawData = audioData
	} else {
		// Assume PCM, convert to mulaw
		pcmSamples, err := audio.BytesToPCM(audioData)
		if err != nil {
			return fmt.Errorf("failed to convert PCM bytes: %w", err)
		}
		mulawData = audio.PCMToMulaw(pcmSamples)
	}

	return conn.conn.WriteMessage(websocket.BinaryMessage, mulawData)
}

// AsteriskInputProcessor receives audio from Asterisk
type AsteriskInputProcessor struct {
	*processors.BaseProcessor
	transport *AsteriskWebSocketTransport
}

func newAsteriskInputProcessor(transport *AsteriskWebSocketTransport) *AsteriskInputProcessor {
	aip := &AsteriskInputProcessor{
		transport: transport,
	}
	aip.BaseProcessor = processors.NewBaseProcessor("AsteriskInput", aip)
	return aip
}

func (p *AsteriskInputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Just pass frames through
	return p.PushFrame(frame, direction)
}

// AsteriskOutputProcessor sends audio to Asterisk
type AsteriskOutputProcessor struct {
	*processors.BaseProcessor
	transport *AsteriskWebSocketTransport
}

func newAsteriskOutputProcessor(transport *AsteriskWebSocketTransport) *AsteriskOutputProcessor {
	aop := &AsteriskOutputProcessor{
		transport: transport,
	}
	aop.BaseProcessor = processors.NewBaseProcessor("AsteriskOutput", aop)
	return aop
}

func (p *AsteriskOutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Send audio frames to Asterisk
	if audioFrame, ok := frame.(*frames.TTSAudioFrame); ok {
		// Get codec from metadata (default to "linear16" for PCM)
		codec := "linear16"
		if codecRaw, exists := audioFrame.Metadata()["codec"]; exists {
			if codecStr, ok := codecRaw.(string); ok {
				codec = codecStr
			}
		}

		// Get connection ID from metadata (set by input processor or call context)
		connectionIDRaw := audioFrame.Metadata()["connection_id"]
		if connectionID, ok := connectionIDRaw.(string); ok {
			if err := p.transport.sendAudio(connectionID, audioFrame.Data, codec); err != nil {
				log.Printf("[AsteriskOutput] Error sending audio: %v", err)
				return p.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		} else {
			log.Printf("[AsteriskOutput] No connection_id in frame metadata, broadcasting to all")
			// Broadcast to all connections if no specific connection ID
			p.transport.connectionMu.RLock()
			for connID := range p.transport.connections {
				p.transport.sendAudio(connID, audioFrame.Data, codec)
			}
			p.transport.connectionMu.RUnlock()
		}
		return nil
	}

	// Pass other frames through
	return p.PushFrame(frame, direction)
}
