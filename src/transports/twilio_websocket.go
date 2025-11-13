package transports

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/gorilla/websocket"
)

// TwilioWebSocketTransport handles WebSocket connections from Twilio Media Streams
type TwilioWebSocketTransport struct {
	port         int
	inputProc    *TwilioInputProcessor
	outputProc   *TwilioOutputProcessor
	server       *http.Server
	upgrader     websocket.Upgrader
	streams      map[string]*twilioStream
	streamMu     sync.RWMutex
}

type twilioStream struct {
	streamSid string
	callSid   string
	conn      *websocket.Conn
	ctx       context.Context
	cancel    context.CancelFunc
}

type twilioMessage struct {
	Event     string                 `json:"event"`
	StreamSid string                 `json:"streamSid,omitempty"`
	Media     *twilioMedia           `json:"media,omitempty"`
	Start     *twilioStart           `json:"start,omitempty"`
	Mark      *twilioMark            `json:"mark,omitempty"`
	Stop      map[string]interface{} `json:"stop,omitempty"`
}

type twilioMedia struct {
	Track     string `json:"track"`
	Chunk     string `json:"chunk"`
	Timestamp string `json:"timestamp"`
	Payload   string `json:"payload"` // base64-encoded mulaw audio
}

type twilioStart struct {
	StreamSid  string                 `json:"streamSid"`
	CallSid    string                 `json:"callSid"`
	AccountSid string                 `json:"accountSid"`
	Tracks     []string               `json:"tracks"`
	MediaFormat map[string]interface{} `json:"mediaFormat"`
	CustomParameters map[string]string `json:"customParameters,omitempty"`
}

type twilioMark struct {
	Name string `json:"name"`
}

// TwilioWebSocketConfig holds configuration for Twilio WebSocket
type TwilioWebSocketConfig struct {
	Port int // Port to listen on (e.g., 8080)
	Path string // WebSocket path (e.g., "/media")
}

// NewTwilioWebSocketTransport creates a new Twilio WebSocket transport
func NewTwilioWebSocketTransport(config TwilioWebSocketConfig) *TwilioWebSocketTransport {
	if config.Path == "" {
		config.Path = "/media"
	}

	t := &TwilioWebSocketTransport{
		port:    config.Port,
		streams: make(map[string]*twilioStream),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins (configure based on security needs)
			},
		},
	}

	t.inputProc = newTwilioInputProcessor(t)
	t.outputProc = newTwilioOutputProcessor(t)

	return t
}

// Input returns the input processor that receives audio from Twilio
func (t *TwilioWebSocketTransport) Input() processors.FrameProcessor {
	return t.inputProc
}

// Output returns the output processor that sends audio to Twilio
func (t *TwilioWebSocketTransport) Output() processors.FrameProcessor {
	return t.outputProc
}

// Start begins listening for WebSocket connections
func (t *TwilioWebSocketTransport) Start(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/media", t.handleWebSocket)

	t.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", t.port),
		Handler: mux,
	}

	go func() {
		log.Printf("[TwilioWS] Listening on port %d", t.port)
		if err := t.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[TwilioWS] Server error: %v", err)
		}
	}()

	return nil
}

// Stop gracefully stops the transport
func (t *TwilioWebSocketTransport) Stop() error {
	// Close all streams
	t.streamMu.Lock()
	for _, stream := range t.streams {
		stream.cancel()
		stream.conn.Close()
	}
	t.streams = make(map[string]*twilioStream)
	t.streamMu.Unlock()

	// Shutdown server
	if t.server != nil {
		return t.server.Shutdown(context.Background())
	}
	return nil
}

func (t *TwilioWebSocketTransport) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	log.Printf("[TwilioWS] New connection from %s", r.RemoteAddr)

	conn, err := t.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("[TwilioWS] Failed to upgrade connection: %v", err)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	stream := &twilioStream{
		conn:   conn,
		ctx:    ctx,
		cancel: cancel,
	}

	// Start receiving messages
	go t.receiveMessages(stream)
}

func (t *TwilioWebSocketTransport) receiveMessages(stream *twilioStream) {
	defer func() {
		stream.cancel()
		stream.conn.Close()
		if stream.streamSid != "" {
			t.streamMu.Lock()
			delete(t.streams, stream.streamSid)
			t.streamMu.Unlock()
		}
		log.Printf("[TwilioWS] Stream closed: %s", stream.streamSid)
	}()

	for {
		select {
		case <-stream.ctx.Done():
			return
		default:
			_, message, err := stream.conn.ReadMessage()
			if err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
					log.Printf("[TwilioWS] Read error: %v", err)
				}
				return
			}

			var msg twilioMessage
			if err := json.Unmarshal(message, &msg); err != nil {
				log.Printf("[TwilioWS] Error parsing message: %v", err)
				continue
			}

			t.handleMessage(stream, &msg)
		}
	}
}

func (t *TwilioWebSocketTransport) handleMessage(stream *twilioStream, msg *twilioMessage) {
	switch msg.Event {
	case "start":
		stream.streamSid = msg.Start.StreamSid
		stream.callSid = msg.Start.CallSid
		t.streamMu.Lock()
		t.streams[stream.streamSid] = stream
		t.streamMu.Unlock()
		log.Printf("[TwilioWS] Stream started: %s (Call: %s)", stream.streamSid, stream.callSid)

	case "media":
		if msg.Media != nil {
			// Decode base64 mulaw audio
			mulawData, err := base64.StdEncoding.DecodeString(msg.Media.Payload)
			if err != nil {
				log.Printf("[TwilioWS] Error decoding audio: %v", err)
				return
			}

			// Pass mulaw directly (no conversion)
			// Services like Deepgram can accept mulaw natively
			audioFrame := frames.NewAudioFrame(mulawData, 8000, 1)
			audioFrame.SetMetadata("stream_sid", stream.streamSid)
			audioFrame.SetMetadata("call_sid", stream.callSid)
			audioFrame.SetMetadata("codec", "mulaw") // Mark as mulaw for services

			// Push to input processor
			if err := t.inputProc.QueueFrame(audioFrame, frames.Downstream); err != nil {
				log.Printf("[TwilioWS] Error queuing audio frame: %v", err)
			}
		}

	case "mark":
		if msg.Mark != nil {
			log.Printf("[TwilioWS] Mark received: %s", msg.Mark.Name)
		}

	case "stop":
		log.Printf("[TwilioWS] Stream stop event received")

	default:
		log.Printf("[TwilioWS] Unknown event: %s", msg.Event)
	}
}

func (t *TwilioWebSocketTransport) sendAudio(streamSid string, audioData []byte, codec string) error {
	t.streamMu.RLock()
	stream, exists := t.streams[streamSid]
	t.streamMu.RUnlock()

	if !exists {
		return fmt.Errorf("stream not found: %s", streamSid)
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

	// Encode mulaw to base64
	payload := base64.StdEncoding.EncodeToString(mulawData)

	// Create media message
	msg := twilioMessage{
		Event:     "media",
		StreamSid: streamSid,
		Media: &twilioMedia{
			Payload: payload,
		},
	}

	return stream.conn.WriteJSON(msg)
}

// TwilioInputProcessor receives audio from Twilio
type TwilioInputProcessor struct {
	*processors.BaseProcessor
	transport *TwilioWebSocketTransport
}

func newTwilioInputProcessor(transport *TwilioWebSocketTransport) *TwilioInputProcessor {
	tip := &TwilioInputProcessor{
		transport: transport,
	}
	tip.BaseProcessor = processors.NewBaseProcessor("TwilioInput", tip)
	return tip
}

func (p *TwilioInputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Just pass frames through
	return p.PushFrame(frame, direction)
}

// TwilioOutputProcessor sends audio to Twilio
type TwilioOutputProcessor struct {
	*processors.BaseProcessor
	transport *TwilioWebSocketTransport
}

func newTwilioOutputProcessor(transport *TwilioWebSocketTransport) *TwilioOutputProcessor {
	top := &TwilioOutputProcessor{
		transport: transport,
	}
	top.BaseProcessor = processors.NewBaseProcessor("TwilioOutput", top)
	return top
}

func (p *TwilioOutputProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Send audio frames to Twilio
	if audioFrame, ok := frame.(*frames.TTSAudioFrame); ok {
		// Get codec from metadata (default to "linear16" for PCM)
		codec := "linear16"
		if codecRaw, exists := audioFrame.Metadata()["codec"]; exists {
			if codecStr, ok := codecRaw.(string); ok {
				codec = codecStr
			}
		}

		// Get stream SID from metadata
		streamSidRaw := audioFrame.Metadata()["stream_sid"]
		if streamSid, ok := streamSidRaw.(string); ok {
			if err := p.transport.sendAudio(streamSid, audioFrame.Data, codec); err != nil {
				log.Printf("[TwilioOutput] Error sending audio: %v", err)
				return p.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
			}
		} else {
			log.Printf("[TwilioOutput] No stream_sid in frame metadata, broadcasting to all")
			// Broadcast to all streams if no specific stream SID
			p.transport.streamMu.RLock()
			for streamSid := range p.transport.streams {
				p.transport.sendAudio(streamSid, audioFrame.Data, codec)
			}
			p.transport.streamMu.RUnlock()
		}
		return nil
	}

	// Pass other frames through
	return p.PushFrame(frame, direction)
}
