package transports

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// audioChunk is a pre-serialized chunk ready for the wire.
type audioChunk struct {
	data         interface{}
	chunkSize    int
	sampleRate   int
	sendInterval time.Duration
}

// PacerEvent is a notification emitted by Pacer to its subscriber.
type PacerEvent int

const (
	// PacerStarted fires when the first chunk in a new burst is sent.
	// Subscribers translate this into a BotStartedSpeakingFrame.
	PacerStarted PacerEvent = iota
	// PacerIdle fires once the inactivity timer expires after the most
	// recent sent chunk — i.e. server-side TTS has finished streaming.
	// Subscribers translate this into a playback-completion signal.
	PacerIdle
)

// pacerIdleAfter is the time-after-last-chunk before Pacer emits PacerIdle.
// Matches the previous BOT_VAD_STOP_SECS = 0.35 inline behavior.
const pacerIdleAfter = 350 * time.Millisecond

// Pacer owns the rate-limited audio sender for a WebSocket-style transport.
// Incoming TTSAudioFrames are chunked, paced at the codec's real-time
// playback rate, and shipped via the injected sendFn. Pacer surfaces two
// coarse lifecycle events (started/idle) so subscribers can drive
// bot-speech and playback-ack handling without owning the goroutine.
//
// Pacer is the seam BotSpeechTracker and PlaybackAckTracker subscribe to
// in #44. Today, WebSocketOutputProcessor owns those tracker concerns
// inline; once they extract, the Pacer surface stays unchanged.
type Pacer struct {
	sendFn func(data interface{}) error
	log    *logger.Logger

	chunkQueue chan *audioChunk
	events     chan PacerEvent

	senderCtx    context.Context
	senderCancel context.CancelFunc
	senderWg     sync.WaitGroup
	cleanupOnce  sync.Once

	// drained, when true, causes Submit to no-op and the sender
	// goroutine to discard chunks. Cleared by Resume().
	drained atomic.Bool

	// burstReset, when true, tells the sender goroutine to treat the
	// next dequeued chunk as the first chunk of a new burst (resets
	// pacing and re-emits PacerStarted). Set by Drain so the next
	// utterance after an interrupt registers as a fresh burst.
	burstReset atomic.Bool

	mu          sync.Mutex
	audioBuffer []byte
}

// NewPacer constructs a Pacer that uses sendFn to push chunks on the
// wire. The sender goroutine starts immediately.
func NewPacer(sendFn func(data interface{}) error, log *logger.Logger) *Pacer {
	if log == nil {
		log = logger.WithPrefix("Pacer")
	}
	p := &Pacer{
		sendFn:      sendFn,
		log:         log,
		chunkQueue:  make(chan *audioChunk, 1000),
		events:      make(chan PacerEvent, 8),
		audioBuffer: make([]byte, 0),
	}
	p.senderCtx, p.senderCancel = context.WithCancel(context.Background())
	p.senderWg.Add(1)
	go p.run()
	return p
}

// Events returns the read-only event channel. Subscribers must keep up
// with this channel; capacity is 8 and PacerIdle emissions are dropped
// when the channel is full.
func (p *Pacer) Events() <-chan PacerEvent { return p.events }

// Submit chunks audioFrame using its codec metadata, serializes each
// chunk via serialize, and queues the results. No-ops while drained.
func (p *Pacer) Submit(
	audioFrame *frames.TTSAudioFrame,
	serialize func(frames.Frame) (interface{}, error),
) error {
	if p.drained.Load() {
		return nil
	}

	codec := "linear16"
	if codecRaw, exists := audioFrame.Metadata()["codec"]; exists {
		if codecStr, ok := codecRaw.(string); ok {
			codec = codecStr
		}
	}

	// Chunk size choice mirrors the prior inline transport logic:
	// telephony codecs ship 20ms frames at 8kHz (160 bytes), PCM ships
	// 10ms frames at 16kHz (320 bytes).
	chunkSize := 320
	if codec == "mulaw" || codec == "alaw" {
		chunkSize = 160
	}

	sendInterval := calculateSendInterval(chunkSize, audioFrame.SampleRate, codec)

	p.mu.Lock()
	currentData := append(p.audioBuffer, audioFrame.Data...)
	p.audioBuffer = make([]byte, 0)
	p.mu.Unlock()

	numChunks := 0
	for len(currentData) >= chunkSize {
		if p.drained.Load() {
			return nil
		}
		chunk := currentData[:chunkSize]
		currentData = currentData[chunkSize:]
		numChunks++

		chunkFrame := frames.NewTTSAudioFrame(chunk, audioFrame.SampleRate, audioFrame.Channels)
		for k, v := range audioFrame.Metadata() {
			chunkFrame.SetMetadata(k, v)
		}

		data, err := serialize(chunkFrame)
		if err != nil {
			p.log.Warn("Serialization error: %v", err)
			continue
		}
		if data == nil {
			continue
		}

		select {
		case p.chunkQueue <- &audioChunk{
			data:         data,
			chunkSize:    chunkSize,
			sampleRate:   audioFrame.SampleRate,
			sendInterval: sendInterval,
		}:
		case <-p.senderCtx.Done():
			return nil
		}
	}

	p.mu.Lock()
	p.audioBuffer = currentData
	p.mu.Unlock()

	if numChunks > 0 {
		p.log.Debug("Streamed %d chunks (%d bytes) immediately (buffer_remainder=%d bytes)",
			numChunks, numChunks*chunkSize, len(currentData))
	}
	return nil
}

// Drain discards the in-flight chunk queue, clears the local audio
// buffer, and marks the Pacer drained so subsequent Submit calls no-op
// until Resume. Also flags the next post-Resume chunk as the start of a
// fresh burst (resets pacing, re-emits PacerStarted).
func (p *Pacer) Drain() {
	p.drained.Store(true)
	p.burstReset.Store(true)

	dropped := 0
drainLoop:
	for {
		select {
		case <-p.chunkQueue:
			dropped++
		default:
			break drainLoop
		}
	}

	p.mu.Lock()
	bufferSize := len(p.audioBuffer)
	p.audioBuffer = make([]byte, 0)
	p.mu.Unlock()

	if dropped > 0 || bufferSize > 0 {
		p.log.Debug("Drained: %d chunks, %d buffered bytes", dropped, bufferSize)
	}
}

// Resume clears the drained flag so Submit re-engages.
func (p *Pacer) Resume() {
	p.drained.Store(false)
}

// Cleanup stops the sender goroutine. Safe to call multiple times.
func (p *Pacer) Cleanup() error {
	p.cleanupOnce.Do(func() {
		p.log.Info("Cleaning up Pacer")
		if p.senderCancel != nil {
			p.senderCancel()
		}
		p.senderWg.Wait()
		close(p.chunkQueue)
		p.log.Info("Pacer cleanup complete")
	})
	return nil
}

func (p *Pacer) run() {
	defer p.senderWg.Done()

	var nextSendTime time.Time
	firstChunk := true

	idleTimer := time.NewTimer(pacerIdleAfter)
	idleTimer.Stop()
	defer idleTimer.Stop()

	for {
		select {
		case <-p.senderCtx.Done():
			p.log.Info("Pacer goroutine stopped")
			return

		case chunk := <-p.chunkQueue:
			if p.drained.Load() {
				p.log.Debug("Pacer: discarding chunk (drained)")
				continue
			}

			if p.burstReset.CompareAndSwap(true, false) {
				firstChunk = true
			}

			now := time.Now()
			if firstChunk {
				nextSendTime = now
			}

			sleepDuration := nextSendTime.Sub(now)
			if sleepDuration > 0 {
				time.Sleep(sleepDuration)
			}

			if err := p.sendFn(chunk.data); err != nil {
				p.log.Warn("Error sending chunk: %v", err)
				if isFatalSendError(err) {
					p.log.Warn("Connection lost, stopping Pacer")
					return
				}
			}

			if sleepDuration <= 0 {
				nextSendTime = time.Now().Add(chunk.sendInterval)
			} else {
				nextSendTime = nextSendTime.Add(chunk.sendInterval)
			}

			if !idleTimer.Stop() {
				select {
				case <-idleTimer.C:
				default:
				}
			}
			idleTimer.Reset(pacerIdleAfter)

			if firstChunk {
				firstChunk = false
				p.emit(PacerStarted)
			}

		case <-idleTimer.C:
			p.emit(PacerIdle)
			// After idle, the next chunk starts a fresh burst.
			firstChunk = true
		}
	}
}

func (p *Pacer) emit(e PacerEvent) {
	select {
	case p.events <- e:
	default:
		p.log.Debug("Pacer events channel full; dropping event %d", e)
	}
}

func isFatalSendError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "broken pipe") ||
		strings.Contains(s, "connection reset") ||
		strings.Contains(s, "closed network connection") ||
		strings.Contains(s, "use of closed")
}

// calculateSendInterval computes the real-time pacing interval for audio
// chunks. chunk_duration = chunk_size / (sample_rate * bytes_per_sample).
// Example: 160 bytes / (8000 * 1) = 20ms (mulaw at 8kHz).
func calculateSendInterval(chunkSize int, sampleRate int, codec string) time.Duration {
	if sampleRate == 0 {
		sampleRate = 8000
	}

	var bytesPerSample int
	switch codec {
	case "float32", "f32":
		bytesPerSample = 4
	case "linear16", "pcm", "l16", "":
		bytesPerSample = 2
	default:
		// Telephony codecs (mulaw, alaw) and any unknown codec.
		bytesPerSample = 1
	}

	intervalSecs := float64(chunkSize) / float64(sampleRate*bytesPerSample)
	interval := time.Duration(intervalSecs * float64(time.Second))
	if interval < time.Millisecond {
		interval = time.Millisecond
	}
	return interval
}
