package services

import (
	"testing"
)

func TestGenerateContextID(t *testing.T) {
	id := GenerateContextID()
	if id == "" {
		t.Error("Expected non-empty context ID")
	}
}

func TestGenerateContextIDUniqueness(t *testing.T) {
	ids := make(map[string]bool)

	for i := 0; i < 1000; i++ {
		id := GenerateContextID()
		if ids[id] {
			t.Errorf("Duplicate context ID generated: %s", id)
		}
		ids[id] = true
	}
}

func TestGenerateContextIDFormat(t *testing.T) {
	id := GenerateContextID()

	if len(id) != 36 {
		t.Errorf("Expected UUID format (36 chars), got %d chars", len(id))
	}

	expectedHyphens := 0
	for i, c := range id {
		if c == '-' {
			expectedHyphens++
		}
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				t.Errorf("Expected hyphen at position %d, got %c", i, c)
			}
		}
	}

	if expectedHyphens != 4 {
		t.Errorf("Expected 4 hyphens in UUID format, got %d", expectedHyphens)
	}
}

func TestGenerateContextIDConcurrency(t *testing.T) {
	const goroutines = 100
	const iterations = 10

	results := make(chan string, goroutines*iterations)

	for g := 0; g < goroutines; g++ {
		go func() {
			for i := 0; i < iterations; i++ {
				results <- GenerateContextID()
			}
		}()
	}

	ids := make(map[string]bool)
	for i := 0; i < goroutines*iterations; i++ {
		id := <-results
		if ids[id] {
			t.Errorf("Duplicate context ID in concurrent generation: %s", id)
		}
		ids[id] = true
	}
}
