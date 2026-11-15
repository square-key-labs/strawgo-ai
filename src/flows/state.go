package flows

import (
	"encoding/json"
	"strings"
	"sync"
)

type State struct {
	mu     sync.RWMutex
	values map[string]any
}

func NewState() *State {
	return &State{values: make(map[string]any)}
}

func (s *State) Set(key string, value any) {
	s.mu.Lock()
	s.values[key] = value
	s.mu.Unlock()
}

func (s *State) Get(key string) (any, bool) {
	s.mu.RLock()
	v, ok := s.values[key]
	s.mu.RUnlock()
	return v, ok
}

func (s *State) Delete(key string) {
	s.mu.Lock()
	delete(s.values, key)
	s.mu.Unlock()
}

func (s *State) Update(fn func(values map[string]any)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fn(s.values)
}

func (s *State) Snapshot() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]any, len(s.values))
	for k, v := range s.values {
		out[k] = v
	}
	return out
}

func Get[T any](s *State, key string) (T, bool) {
	v, ok := s.Get(key)
	if !ok {
		var zero T
		return zero, false
	}
	t, ok := v.(T)
	return t, ok
}

type Key[T any] string

func (k Key[T]) Get(s *State) (T, bool) {
	return Get[T](s, string(k))
}

func (k Key[T]) Set(s *State, value T) {
	s.Set(string(k), value)
}

func (s *State) lookupPath(path string) (any, bool) {
	parts := strings.Split(path, ".")
	current, ok := s.Get(parts[0])
	if !ok {
		return nil, false
	}
	for _, part := range parts[1:] {
		m, ok := asMap(current)
		if !ok {
			return nil, false
		}
		if current, ok = m[part]; !ok {
			return nil, false
		}
	}
	return current, true
}

func asMap(v any) (map[string]any, bool) {
	switch m := v.(type) {
	case map[string]any:
		return m, true
	case map[string]string:
		out := make(map[string]any, len(m))
		for k, v := range m {
			out[k] = v
		}
		return out, true
	case nil:
		return nil, false
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, false
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, false
	}
	return out, true
}
