# Interruptions Package

This package provides interruption strategies for StrawGo, allowing users to interrupt the bot during speech based on various conditions.

## Overview

Interruption strategies determine when a user should be allowed to interrupt the bot. They analyze user input (audio, text, or both) to make this decision.

## Interface

```go
type InterruptionStrategy interface {
    AppendAudio(audio []byte, sampleRate int) error
    AppendText(text string) error
    ShouldInterrupt() (bool, error)
    Reset() error
}
```

## Built-in Strategies

### MinWordsInterruptionStrategy

Interrupts when the user has spoken at least a minimum number of words.

```go
strategy := interruptions.NewMinWordsInterruptionStrategy(3)
```

## Creating Custom Strategies

1. Embed `BaseInterruptionStrategy` for default implementations
2. Implement `ShouldInterrupt()` with your custom logic
3. Override `AppendAudio()` and/or `AppendText()` if needed
4. Implement `Reset()` to clear state

Example:

```go
type MyStrategy struct {
    interruptions.BaseInterruptionStrategy
    // your fields
}

func (s *MyStrategy) ShouldInterrupt() (bool, error) {
    s.mu.Lock()
    defer s.mu.Unlock()

    // your logic here
    return shouldInterrupt, nil
}

func (s *MyStrategy) Reset() error {
    s.mu.Lock()
    defer s.mu.Unlock()

    // clear your state
    return nil
}
```

## Usage

See `docs/INTERRUPTIONS.md` for complete documentation and examples.
