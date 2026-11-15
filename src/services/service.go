package services

import (
	"context"
	"sync"

	"github.com/google/uuid"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type AIService interface {
	processors.FrameProcessor
	Initialize(ctx context.Context) error
	Cleanup() error
}

type STTService interface {
	AIService
	SetLanguage(lang string)
	SetModel(model string)
}

type TTSService interface {
	AIService
	SetVoice(voiceID string)
	SetModel(model string)
}

func GenerateContextID() string {
	return uuid.New().String()
}

type AudioContextManager struct {
	contextID            string
	currentTurnContextID string
	mu                   sync.Mutex

	ReuseContextIDWithinTurn bool
}

func NewAudioContextManager() *AudioContextManager {
	return &AudioContextManager{ReuseContextIDWithinTurn: true}
}

func (m *AudioContextManager) HasActiveAudioContext() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.contextID != ""
}

func (m *AudioContextManager) GetActiveAudioContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.contextID
}

func (m *AudioContextManager) SetActiveAudioContextID(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.contextID = id
}

func (m *AudioContextManager) ResetActiveAudioContext() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.contextID = ""
	m.currentTurnContextID = ""
}

func (m *AudioContextManager) GetTurnContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.currentTurnContextID
}

func (m *AudioContextManager) GetOrCreateTurnContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.currentTurnContextID == "" {
		m.currentTurnContextID = GenerateContextID()
	}
	return m.currentTurnContextID
}

func (m *AudioContextManager) GetOrCreateContextID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.contextID == "" {
		if m.ReuseContextIDWithinTurn && m.currentTurnContextID != "" {
			m.contextID = m.currentTurnContextID
		} else {
			m.contextID = GenerateContextID()
		}
	}
	return m.contextID
}
