package llm

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

func texts(msgs []Message) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = string(m.Role) + ":" + m.Text()
	}
	return out
}

func TestSnapshotIsUnaffectedByMutations(t *testing.T) {
	base := func() *Context {
		return NewContext(
			System("sys"),
			User("hi"),
			ToolCalls(ToolCallPart{ID: "c1", Name: "f", Arguments: json.RawMessage(`{}`)}),
			ToolResult("c1", "f", "pending"),
		)
	}
	cases := []struct {
		name   string
		mutate func(c *Context)
	}{
		{"Add", func(c *Context) { c.Add(Assistant("added"), User("more")) }},
		{"Set", func(c *Context) { c.Set([]Message{User("replaced")}) }},
		{"TransformInPlace", func(c *Context) {
			c.Transform(func(in []Message) []Message {
				in[1].Parts[0] = TextPart{Text: "mutated"}
				in[0].Role = RoleDeveloper
				return in
			})
		}},
		{"TransformAppend", func(c *Context) {
			c.Transform(func(in []Message) []Message { return append(in, User("x")) })
		}},
		{"ReplaceToolResult", func(c *Context) {
			if !c.ReplaceToolResult("c1", "done", true) {
				t.Fatal("ReplaceToolResult returned false")
			}
		}},
		{"SetTools", func(c *Context) {
			c.SetTools(Tools{Functions: []FunctionSchema{{Name: "g"}}})
		}},
		{"SetToolChoice", func(c *Context) { c.SetToolChoice(ToolChoice{Mode: ToolChoiceRequired}) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := base()
			c.SetTools(Tools{Functions: []FunctionSchema{{Name: "f"}}})
			snap := c.Snapshot()
			wantTexts := texts(snap.Messages)
			wantResult, _ := snap.Messages[3].ToolResult()
			wantTools := len(snap.Tools.Functions)
			wantChoice := snap.ToolChoice

			tc.mutate(c)

			if got := texts(snap.Messages); !reflect.DeepEqual(got, wantTexts) {
				t.Fatalf("snapshot messages changed: %v -> %v", wantTexts, got)
			}
			if r, _ := snap.Messages[3].ToolResult(); r != wantResult {
				t.Fatalf("snapshot tool result changed: %+v -> %+v", wantResult, r)
			}
			if len(snap.Tools.Functions) != wantTools || snap.Tools.Functions[0].Name != "f" {
				t.Fatalf("snapshot tools changed: %+v", snap.Tools)
			}
			if snap.ToolChoice != wantChoice {
				t.Fatalf("snapshot tool choice changed")
			}
			if c.Snapshot().Version <= snap.Version {
				t.Fatalf("version not bumped: %d -> %d", snap.Version, c.Snapshot().Version)
			}
		})
	}
}

func TestSnapshotAppendDoesNotLeakIntoContext(t *testing.T) {
	c := NewContext(User("a"))
	snap := c.Snapshot()
	_ = append(snap.Messages, User("leak"))
	c.Add(User("b"))
	if got := texts(c.Messages()); !reflect.DeepEqual(got, []string{"user:a", "user:b"}) {
		t.Fatalf("context = %v", got)
	}
	if len(snap.Messages) != 1 {
		t.Fatalf("snapshot len = %d", len(snap.Messages))
	}
}

func TestSetCopiesCallerSlice(t *testing.T) {
	in := []Message{User("a")}
	c := NewContext()
	c.Set(in)
	in[0] = User("changed")
	if c.Messages()[0].Text() != "a" {
		t.Fatalf("Set aliased caller slice")
	}
}

func TestReplaceToolResult(t *testing.T) {
	c := NewContext(ToolResult("c1", "f", "old"), ToolResult("c2", "g", "x"))
	if c.ReplaceToolResult("missing", "x", false) {
		t.Fatal("expected false for unknown id")
	}
	v := c.Snapshot().Version
	if !c.ReplaceToolResult("c1", "new", true) {
		t.Fatal("expected true")
	}
	r, ok := c.Messages()[0].ToolResult()
	if !ok || r.Content != "new" || !r.IsError || r.Name != "f" {
		t.Fatalf("result = %+v", r)
	}
	if c.Snapshot().Version != v+1 {
		t.Fatal("version not bumped")
	}
}

func TestEmptyMutationsDoNotBumpVersion(t *testing.T) {
	c := NewContext(User("a"))
	v := c.Snapshot().Version
	c.Add()
	if c.Snapshot().Version != v {
		t.Fatal("empty Add bumped version")
	}
}

func TestContextConcurrentReadersWriters(t *testing.T) {
	c := NewContext(System("s"))
	var wg sync.WaitGroup
	const writers, readers, iters = 4, 4, 200
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				switch i % 4 {
				case 0:
					c.Add(User(fmt.Sprintf("w%d-%d", w, i)))
				case 1:
					c.Transform(func(in []Message) []Message { return in })
				case 2:
					c.SetTools(Tools{Functions: []FunctionSchema{{Name: "f"}}})
				case 3:
					c.ReplaceToolResult("none", "x", false)
				}
			}
		}(w)
	}
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var last uint64
			for i := 0; i < iters; i++ {
				s := c.Snapshot()
				if s.Version < last {
					t.Errorf("version went backwards %d -> %d", last, s.Version)
					return
				}
				last = s.Version
				for _, m := range s.Messages {
					_ = m.Text()
				}
				_ = c.Len()
				_ = c.Tools()
			}
		}()
	}
	wg.Wait()
	if got, want := c.Len(), 1+writers*iters/4; got != want {
		t.Fatalf("len = %d want %d", got, want)
	}
}

func TestMessageHelpers(t *testing.T) {
	m := Message{Role: RoleAssistant, Parts: []Part{
		TextPart{Text: "a"},
		ToolCallPart{ID: "1", Name: "f"},
		TextPart{Text: "b"},
		ToolCallPart{ID: "2", Name: "g"},
	}}
	if m.Text() != "ab" {
		t.Fatalf("Text = %q", m.Text())
	}
	if calls := m.ToolCalls(); len(calls) != 2 || calls[1].Name != "g" {
		t.Fatalf("ToolCalls = %+v", calls)
	}
	if _, ok := m.ToolResult(); ok {
		t.Fatal("unexpected tool result")
	}
}
