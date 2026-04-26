package openai

import (
	"context"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestUpdateSettingsAppliesKnownKeys(t *testing.T) {
	s := NewLLMService(LLMConfig{
		APIKey:      "k",
		Model:       "gpt-4o",
		Temperature: 0.7,
	})
	if err := s.UpdateSettings(map[string]interface{}{
		"model":              "gpt-4o-mini",
		"temperature":        0.2,
		"system_instruction": "Be brief.",
	}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if s.model != "gpt-4o-mini" {
		t.Fatalf("expected model=gpt-4o-mini, got %s", s.model)
	}
	if s.temperature != 0.2 {
		t.Fatalf("expected temperature=0.2, got %v", s.temperature)
	}
	if s.systemInstruction != "Be brief." {
		t.Fatalf("expected system_instruction set, got %q", s.systemInstruction)
	}
}

func TestUpdateSettingsTemperatureAcceptsInt(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k", Model: "gpt-4o"})
	if err := s.UpdateSettings(map[string]interface{}{"temperature": 1}); err != nil {
		t.Fatalf("UpdateSettings: %v", err)
	}
	if s.temperature != 1.0 {
		t.Fatalf("expected temperature coerced to 1.0, got %v", s.temperature)
	}
}

func TestLLMUpdateSettingsFrameTargetingSelf(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k", Model: "gpt-4o"})
	frame := frames.NewLLMUpdateSettingsFrameForService(
		map[string]interface{}{"model": "gpt-4o-mini"}, "OpenAI",
	)
	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.model != "gpt-4o-mini" {
		t.Fatalf("expected model=gpt-4o-mini, got %s", s.model)
	}
}

func TestLLMUpdateSettingsFrameTargetingOther(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k", Model: "gpt-4o"})
	frame := frames.NewLLMUpdateSettingsFrameForService(
		map[string]interface{}{"model": "gpt-4o-mini"}, "Anthropic",
	)
	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.model != "gpt-4o" {
		t.Fatalf("expected model unchanged (gpt-4o), got %s", s.model)
	}
}

func TestLLMUpdateSettingsFrameUntargeted(t *testing.T) {
	s := NewLLMService(LLMConfig{APIKey: "k", Model: "gpt-4o"})
	frame := frames.NewLLMUpdateSettingsFrame(map[string]interface{}{"model": "gpt-4o-mini"})
	if err := s.HandleFrame(context.TODO(), frame, frames.Downstream); err != nil {
		t.Fatalf("HandleFrame: %v", err)
	}
	if s.model != "gpt-4o-mini" {
		t.Fatalf("expected model=gpt-4o-mini (untargeted), got %s", s.model)
	}
}
