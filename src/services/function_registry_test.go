package services

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestFunctionRegistryRegisterLookup(t *testing.T) {
	r := NewInMemoryFunctionRegistry()

	timeout := 5 * time.Second
	fn := RegisteredFunction{
		Name:                 "get_weather",
		Schema:               Tool{Type: "function", Function: ToolFunction{Name: "get_weather"}},
		Callback:             func(_ context.Context, _ map[string]interface{}) (interface{}, error) { return "sunny", nil },
		TimeoutSecs:          &timeout,
		CancelOnInterruption: true,
	}
	if err := r.Register(fn); err != nil {
		t.Fatalf("Register: %v", err)
	}

	got, ok := r.Lookup("get_weather")
	if !ok {
		t.Fatalf("Lookup: not found")
	}
	if got.Name != "get_weather" {
		t.Fatalf("expected name=get_weather, got %q", got.Name)
	}
	if got.TimeoutSecs == nil || *got.TimeoutSecs != 5*time.Second {
		t.Fatalf("expected TimeoutSecs=5s, got %v", got.TimeoutSecs)
	}
	if !got.CancelOnInterruption {
		t.Fatalf("expected CancelOnInterruption=true")
	}
}

func TestFunctionRegistryEmptyName(t *testing.T) {
	r := NewInMemoryFunctionRegistry()
	if err := r.Register(RegisteredFunction{}); err == nil {
		t.Fatalf("expected error for empty Name")
	}
}

func TestFunctionRegistryLookupMissing(t *testing.T) {
	r := NewInMemoryFunctionRegistry()
	if _, ok := r.Lookup("nope"); ok {
		t.Fatalf("expected lookup miss")
	}
}

func TestFunctionRegistryListAndOverwrite(t *testing.T) {
	r := NewInMemoryFunctionRegistry()
	_ = r.Register(RegisteredFunction{Name: "a"})
	_ = r.Register(RegisteredFunction{Name: "b"})
	if got := len(r.List()); got != 2 {
		t.Fatalf("expected 2, got %d", got)
	}

	// Overwrite "a"
	_ = r.Register(RegisteredFunction{Name: "a", CancelOnInterruption: true})
	a, _ := r.Lookup("a")
	if !a.CancelOnInterruption {
		t.Fatalf("overwrite did not take effect")
	}
}

func TestFunctionRegistryRaceSafe(t *testing.T) {
	r := NewInMemoryFunctionRegistry()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_ = r.Register(RegisteredFunction{Name: "fn"})
			_, _ = r.Lookup("fn")
			_ = r.List()
		}(i)
	}
	wg.Wait()
}
