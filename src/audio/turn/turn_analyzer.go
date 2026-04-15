package turn

import (
	"time"
)

// EndOfTurnState represents the result of turn analysis
type EndOfTurnState int

const (
	// TurnIncomplete indicates the user is still speaking or may continue
	TurnIncomplete EndOfTurnState = iota
	// TurnComplete indicates the user has finished their turn
	TurnComplete
)

func (s EndOfTurnState) String() string {
	switch s {
	case TurnIncomplete:
		return "INCOMPLETE"
	case TurnComplete:
		return "COMPLETE"
	default:
		return "UNKNOWN"
	}
}

// TurnAnalyzerParams holds base configuration for turn analyzers
type TurnAnalyzerParams struct {
	// StopSecs is the duration of silence in seconds before triggering silence-based end of turn
	StopSecs float64
}

// DefaultTurnAnalyzerParams returns default parameters
func DefaultTurnAnalyzerParams() *TurnAnalyzerParams {
	return &TurnAnalyzerParams{
		StopSecs: 3.0,
	}
}

// TurnMetrics holds metrics from turn analysis
type TurnMetrics struct {
	// IsComplete indicates if the turn was classified as complete
	IsComplete bool
	// Probability is the model's confidence (0.0-1.0)
	Probability float64
	// InferenceTimeMs is the time taken for ML inference
	InferenceTimeMs float64
	// TotalTimeMs is the total end-to-end processing time
	TotalTimeMs float64
}

// TurnAnalyzer is the interface for analyzing user end of turn
// This is used alongside VAD to determine when a user has finished speaking
// using ML-based analysis rather than just silence detection
type TurnAnalyzer interface {
	// AppendAudio appends audio data for analysis
	// buffer: raw audio data (16kHz mono PCM int16)
	// isSpeech: whether VAD detected speech in this buffer
	// Returns the current end-of-turn state
	AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState

	// AnalyzeEndOfTurn runs ML inference to determine if the turn has ended
	// This is called when VAD detects silence after speech
	// Returns the state and optional metrics
	AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error)

	// SpeechTriggered returns true if speech has been detected and analysis is active
	SpeechTriggered() bool

	// SetSampleRate sets the sample rate for audio processing
	SetSampleRate(sampleRate int)

	// Clear resets the turn analyzer to its initial state
	Clear()
}

// audioChunk represents a timestamped audio segment
type audioChunk struct {
	timestamp time.Time
	audio     []float32
}

// BaseTurnAnalyzer provides common functionality for turn analyzers
type BaseTurnAnalyzer struct {
	sampleRate      int
	params          *TurnAnalyzerParams
	audioBuffer     []audioChunk
	speechTriggered bool
	silenceMs       float64
	speechStartTime time.Time
	stopMs          float64
}

// NewBaseTurnAnalyzer creates a new base turn analyzer
func NewBaseTurnAnalyzer(sampleRate int, params *TurnAnalyzerParams) *BaseTurnAnalyzer {
	if params == nil {
		params = DefaultTurnAnalyzerParams()
	}
	return &BaseTurnAnalyzer{
		sampleRate:  sampleRate,
		params:      params,
		audioBuffer: make([]audioChunk, 0),
		stopMs:      params.StopSecs * 1000,
	}
}

// SpeechTriggered returns true if speech has been detected
func (b *BaseTurnAnalyzer) SpeechTriggered() bool {
	return b.speechTriggered
}

// SetSampleRate sets the sample rate
func (b *BaseTurnAnalyzer) SetSampleRate(sampleRate int) {
	b.sampleRate = sampleRate
}

// Clear resets the analyzer state
func (b *BaseTurnAnalyzer) Clear() {
	b.clearState(TurnComplete)
}

func (b *BaseTurnAnalyzer) clearState(state EndOfTurnState) {
	// If incomplete, keep speech_triggered as true
	b.speechTriggered = state == TurnIncomplete
	b.audioBuffer = make([]audioChunk, 0)
	b.speechStartTime = time.Time{}
	b.silenceMs = 0
}

// AppendAudio appends audio data and tracks speech/silence state
func (b *BaseTurnAnalyzer) AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState {
	// Convert raw audio to float32 format
	audioFloat32 := bytesToFloat32(buffer)
	b.audioBuffer = append(b.audioBuffer, audioChunk{
		timestamp: time.Now(),
		audio:     audioFloat32,
	})

	state := TurnIncomplete

	if isSpeech {
		// Reset silence tracking on speech
		b.silenceMs = 0
		b.speechTriggered = true
		if b.speechStartTime.IsZero() {
			b.speechStartTime = time.Now()
		}
	} else {
		if b.speechTriggered {
			// Calculate chunk duration in ms
			numSamples := len(buffer) / 2 // int16 = 2 bytes per sample
			chunkDurationMs := float64(numSamples) / (float64(b.sampleRate) / 1000)
			b.silenceMs += chunkDurationMs

			// If silence exceeds threshold, mark end of turn
			if b.silenceMs >= b.stopMs {
				state = TurnComplete
				b.clearState(state)
			}
		} else {
			// Trim buffer to prevent unbounded growth before speech
			// Keep up to stopSecs + 8 seconds (max audio for model)
			maxBufferTime := b.params.StopSecs + 8.0
			cutoffTime := time.Now().Add(-time.Duration(maxBufferTime * float64(time.Second)))
			for len(b.audioBuffer) > 0 && b.audioBuffer[0].timestamp.Before(cutoffTime) {
				b.audioBuffer = b.audioBuffer[1:]
			}
		}
	}

	return state
}

// GetAudioSegment extracts the audio segment for ML analysis
// Returns up to maxDurationSecs of audio, keeping the most recent audio
func (b *BaseTurnAnalyzer) GetAudioSegment(maxDurationSecs float64, preSpeechMs float64) []float32 {
	if len(b.audioBuffer) == 0 {
		return nil
	}

	// Find start index based on speech start time minus pre-speech buffer
	startTime := b.speechStartTime.Add(-time.Duration(preSpeechMs * float64(time.Millisecond)))
	startIndex := 0
	for i, chunk := range b.audioBuffer {
		if !chunk.timestamp.Before(startTime) {
			startIndex = i
			break
		}
	}

	// Concatenate audio chunks
	var totalSamples int
	for i := startIndex; i < len(b.audioBuffer); i++ {
		totalSamples += len(b.audioBuffer[i].audio)
	}

	segment := make([]float32, 0, totalSamples)
	for i := startIndex; i < len(b.audioBuffer); i++ {
		segment = append(segment, b.audioBuffer[i].audio...)
	}

	// Limit to max duration (keeping the end)
	maxSamples := int(maxDurationSecs * float64(b.sampleRate))
	if len(segment) > maxSamples {
		segment = segment[len(segment)-maxSamples:]
	}

	return segment
}

// bytesToFloat32 converts raw int16 PCM audio to float32 normalized [-1, 1]
func bytesToFloat32(buffer []byte) []float32 {
	numSamples := len(buffer) / 2
	result := make([]float32, numSamples)
	for i := range numSamples {
		// Little-endian int16
		sample := int16(buffer[i*2]) | int16(buffer[i*2+1])<<8
		result[i] = float32(sample) / 32768.0
	}
	return result
}
