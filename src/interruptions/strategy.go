package interruptions

import "sync"

// InterruptionStrategy determines when users can interrupt bot speech
type InterruptionStrategy interface {
	// AppendAudio adds audio data for analysis
	// Not all strategies need to handle audio
	AppendAudio(audio []byte, sampleRate int) error

	// AppendText adds text data for analysis
	// Not all strategies need to handle text
	AppendText(text string) error

	// ShouldInterrupt determines if the user should interrupt the bot
	// This is called when the user stops speaking to decide whether
	// the user should interrupt the bot based on aggregated audio/text
	ShouldInterrupt() (bool, error)

	// Reset clears the current accumulated text and/or audio
	Reset() error
}

// BaseInterruptionStrategy provides a default implementation
// that does nothing for audio/text methods
type BaseInterruptionStrategy struct {
	mu sync.Mutex
}

func (b *BaseInterruptionStrategy) AppendAudio(audio []byte, sampleRate int) error {
	// Default implementation does nothing
	return nil
}

func (b *BaseInterruptionStrategy) AppendText(text string) error {
	// Default implementation does nothing
	return nil
}

func (b *BaseInterruptionStrategy) Reset() error {
	// Default implementation does nothing
	return nil
}
