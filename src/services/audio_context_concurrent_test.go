package services

import (
	"sync"
	"testing"
)

func TestConcurrentManagerNewContextIDRegistersInSet(t *testing.T) {
	m := NewConcurrentAudioContextManager()

	id1 := m.NewContextID()
	id2 := m.NewContextID()

	if id1 == id2 {
		t.Fatalf("expected unique IDs, got %s == %s", id1, id2)
	}
	if !m.IsActiveContext(id1) || !m.IsActiveContext(id2) {
		t.Fatalf("expected both IDs registered: %v %v", m.IsActiveContext(id1), m.IsActiveContext(id2))
	}
	got := m.ActiveContextIDs()
	if len(got) != 2 {
		t.Fatalf("expected 2 active contexts, got %d", len(got))
	}
}

func TestConcurrentManagerGetOrCreateAlwaysFresh(t *testing.T) {
	m := NewConcurrentAudioContextManager()

	id1 := m.GetOrCreateContextID()
	id2 := m.GetOrCreateContextID()

	if id1 == id2 {
		t.Fatalf("concurrent mode: GetOrCreateContextID must return fresh IDs, got %s == %s", id1, id2)
	}
	if !m.IsActiveContext(id1) || !m.IsActiveContext(id2) {
		t.Fatalf("both IDs should be registered")
	}
}

func TestConcurrentManagerRemoveContextDropsFromSet(t *testing.T) {
	m := NewConcurrentAudioContextManager()
	id := m.NewContextID()

	if !m.IsActiveContext(id) {
		t.Fatalf("ID should be active immediately after NewContextID")
	}
	m.RemoveContext(id)
	if m.IsActiveContext(id) {
		t.Fatalf("ID should be inactive after RemoveContext")
	}
	if len(m.ActiveContextIDs()) != 0 {
		t.Fatalf("active set should be empty after removing only ID")
	}
}

func TestConcurrentManagerResetClearsSet(t *testing.T) {
	m := NewConcurrentAudioContextManager()
	for i := 0; i < 5; i++ {
		m.NewContextID()
	}
	if len(m.ActiveContextIDs()) != 5 {
		t.Fatalf("expected 5 active contexts before reset")
	}
	m.ResetActiveAudioContext()
	if len(m.ActiveContextIDs()) != 0 {
		t.Fatalf("Reset must drain set; got %d", len(m.ActiveContextIDs()))
	}
	if m.HasActiveAudioContext() {
		t.Fatalf("Reset must clear scalar slot in concurrent mode")
	}
}

func TestSingleContextManagerUnchanged(t *testing.T) {
	m := NewAudioContextManager() // legacy mode

	id1 := m.GetOrCreateContextID()
	id2 := m.GetOrCreateContextID()

	if id1 != id2 {
		t.Fatalf("single-context mode: GetOrCreateContextID must reuse the slot, got %s != %s", id1, id2)
	}
	got := m.ActiveContextIDs()
	if len(got) != 1 || got[0] != id1 {
		t.Fatalf("single-context ActiveContextIDs must equal [scalar], got %v", got)
	}
}

func TestConcurrentManagerRegisterDoesNotDuplicate(t *testing.T) {
	m := NewConcurrentAudioContextManager()
	m.RegisterContext("ctx-A")
	m.RegisterContext("ctx-A")
	m.RegisterContext("ctx-B")

	ids := m.ActiveContextIDs()
	if len(ids) != 2 {
		t.Fatalf("expected 2 unique contexts after duplicate Register, got %d (%v)", len(ids), ids)
	}
}

func TestConcurrentManagerRaceNewAndRemove(t *testing.T) {
	m := NewConcurrentAudioContextManager()
	const N = 200
	var wg sync.WaitGroup
	wg.Add(N)
	ids := make(chan string, N)

	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			ids <- m.NewContextID()
		}()
	}
	wg.Wait()
	close(ids)

	for id := range ids {
		m.RemoveContext(id)
	}
	if got := len(m.ActiveContextIDs()); got != 0 {
		t.Fatalf("expected empty set after all removed, got %d", got)
	}
}

func TestRemoveContextAndIsEmptyAtomic(t *testing.T) {
	m := NewConcurrentAudioContextManager()
	id1 := m.NewContextID()
	id2 := m.NewContextID()

	if last := m.RemoveContextAndIsEmpty(id1); last {
		t.Fatalf("removing id1 must not report empty when id2 still active")
	}
	if last := m.RemoveContextAndIsEmpty(id2); !last {
		t.Fatalf("removing last id must report empty=true")
	}
	// Single-context mode: empty if scalar matches input.
	sm := NewAudioContextManager()
	id := sm.GetOrCreateContextID()
	if last := sm.RemoveContextAndIsEmpty(id); !last {
		t.Fatalf("single-context: removing only ID must report empty=true")
	}
}

func TestRemoveContextOnSingleContextClearsScalar(t *testing.T) {
	m := NewAudioContextManager()
	id := m.GetOrCreateContextID()
	m.RemoveContext(id)
	if m.HasActiveAudioContext() {
		t.Fatalf("scalar slot must be cleared when RemoveContext matches it")
	}
}
