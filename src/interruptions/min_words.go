package interruptions

import (
	"log"
	"strings"
)

// MinWordsInterruptionStrategy is an interruption strategy based on
// a minimum number of words spoken by the user. The strategy will
// return true if the user has said at least that amount of words.
type MinWordsInterruptionStrategy struct {
	BaseInterruptionStrategy
	minWords int
	text     string
}

// NewMinWordsInterruptionStrategy creates a new minimum words strategy
func NewMinWordsInterruptionStrategy(minWords int) *MinWordsInterruptionStrategy {
	return &MinWordsInterruptionStrategy{
		minWords: minWords,
		text:     "",
	}
}

// AppendText appends text for word count analysis
func (m *MinWordsInterruptionStrategy) AppendText(text string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.text += text
	return nil
}

// ShouldInterrupt checks if the minimum word count has been reached
func (m *MinWordsInterruptionStrategy) ShouldInterrupt() (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	words := strings.Fields(m.text)
	wordCount := len(words)
	interrupt := wordCount >= m.minWords

	log.Printf("[MinWordsStrategy] should_interrupt=%v num_spoken_words=%d min_words=%d",
		interrupt, wordCount, m.minWords)

	return interrupt, nil
}

// Reset resets the accumulated text for the next analysis cycle
func (m *MinWordsInterruptionStrategy) Reset() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.text = ""
	return nil
}
