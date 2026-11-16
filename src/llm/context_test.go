package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
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

func fullContext() *Context {
	c := NewContext(
		Message{Role: RoleSystem, Parts: []Part{TextPart{Text: "sys"}}},
		Message{Role: RoleUser, Name: "caller", Parts: []Part{
			TextPart{Text: "look"},
			ImagePart{Data: []byte{0, 1, 2, 255}, MIME: "image/png", Detail: "high"},
			ImagePart{URL: "https://example.com/a.jpg", MIME: "image/jpeg"},
			AudioPart{Data: []byte{9, 8, 7}, MIME: "audio/pcm", SampleRate: 16000},
			FilePart{Data: []byte("%PDF"), MIME: "application/pdf", Name: "a.pdf"},
			FilePart{URL: "https://example.com/b.pdf", MIME: "application/pdf"},
		}},
		Message{Role: RoleAssistant, Parts: []Part{
			ThoughtPart{Text: "thinking", Signature: "sig-1"},
			ThoughtPart{Signature: "opaque", Redacted: true},
			ToolCallPart{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`), Signature: "tc-sig"},
			ProviderPart{Provider: "anthropic", Raw: json.RawMessage(`{"type":"server_tool_use","id":"s1"}`)},
		}},
		Message{Role: RoleTool, Parts: []Part{ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: "boom", IsError: true}}},
		Assistant("done"),
	)
	c.SetTools(Tools{
		Functions: []FunctionSchema{{
			Name:        "lookup",
			Description: "Look up",
			Properties:  map[string]any{"q": map[string]any{"type": "string"}},
			Required:    []string{"q"},
			Strict:      true,
		}},
		Custom: map[string][]json.RawMessage{"gemini": {json.RawMessage(`{"google_search":{}}`)}},
	})
	c.SetToolChoice(ToolChoice{Mode: ToolChoiceFunction, Name: "lookup"})
	return c
}

func TestMarshalContextRoundTrip(t *testing.T) {
	orig := fullContext()
	data, err := MarshalContext(orig)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalContext(data)
	if err != nil {
		t.Fatal(err)
	}
	want, got := orig.Snapshot(), restored.Snapshot()
	if !reflect.DeepEqual(got.Messages, want.Messages) {
		t.Fatalf("messages differ:\n got %#v\nwant %#v", got.Messages, want.Messages)
	}
	if !reflect.DeepEqual(got.Tools, want.Tools) {
		t.Fatalf("tools differ:\n got %#v\nwant %#v", got.Tools, want.Tools)
	}
	if got.ToolChoice != want.ToolChoice {
		t.Fatalf("tool choice = %+v", got.ToolChoice)
	}
	again, err := MarshalContext(restored)
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != string(data) {
		t.Fatalf("second encoding differs:\n%s\n%s", again, data)
	}
}

func TestMarshalContextPreservesIDs(t *testing.T) {
	data, err := MarshalContext(fullContext())
	if err != nil {
		t.Fatal(err)
	}
	c, err := UnmarshalContext(data)
	if err != nil {
		t.Fatal(err)
	}
	msgs := c.Messages()
	calls := msgs[2].ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Signature != "tc-sig" {
		t.Fatalf("calls = %+v", calls)
	}
	r, ok := msgs[3].ToolResult()
	if !ok || r.ToolCallID != "call_1" {
		t.Fatalf("result = %+v", r)
	}
	if !c.ReplaceToolResult("call_1", "ok", false) {
		t.Fatal("restored tool result not found by id")
	}
}

func TestMarshalContextOmitBinary(t *testing.T) {
	data, err := MarshalContext(fullContext(), OmitBinary())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"data"`) {
		t.Fatalf("binary data still present: %s", data)
	}
	c, err := UnmarshalContext(data)
	if err != nil {
		t.Fatal(err)
	}
	parts := c.Messages()[1].Parts
	want := []Part{
		TextPart{Text: "look"},
		TextPart{Text: "[image/png image omitted, 4 bytes]"},
		ImagePart{URL: "https://example.com/a.jpg", MIME: "image/jpeg"},
		TextPart{Text: "[audio/pcm audio omitted, 3 bytes]"},
		TextPart{Text: "[application/pdf file omitted, 4 bytes]"},
		FilePart{URL: "https://example.com/b.pdf", MIME: "application/pdf"},
	}
	if !reflect.DeepEqual(parts, want) {
		t.Fatalf("parts = %#v", parts)
	}
	if got := c.Messages()[2].Parts; !reflect.DeepEqual(got, fullContext().Messages()[2].Parts) {
		t.Fatalf("non-binary parts changed: %#v", got)
	}
}

func TestUnmarshalContextErrors(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want error
	}{
		{"unknown part", `{"version":1,"messages":[{"role":"user","parts":[{"type":"hologram"}]}]}`, ErrUnknownPartType},
		{"future version", `{"version":2,"messages":[]}`, ErrUnsupportedContextVersion},
		{"missing version", `{"messages":[]}`, ErrUnsupportedContextVersion},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := UnmarshalContext([]byte(tc.in))
			if !errors.Is(err, tc.want) || c != nil {
				t.Fatalf("got %v, %v; want %v", c, err, tc.want)
			}
		})
	}
	if _, err := UnmarshalContext([]byte(`not json`)); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestRestoredContextKeepsSnapshotSemantics(t *testing.T) {
	data, err := MarshalContext(fullContext())
	if err != nil {
		t.Fatal(err)
	}
	c, err := UnmarshalContext(data)
	if err != nil {
		t.Fatal(err)
	}
	snap := c.Snapshot()
	before := texts(snap.Messages)
	beforeResult, _ := snap.Messages[3].ToolResult()

	c.Transform(func(in []Message) []Message {
		in[0].Parts[0] = TextPart{Text: "changed"}
		return in
	})
	c.ReplaceToolResult("call_1", "ok", false)
	c.Add(User("more"))
	c.SetToolChoice(ToolChoice{Mode: ToolChoiceNone})

	if got := texts(snap.Messages); !reflect.DeepEqual(got, before) {
		t.Fatalf("snapshot changed: %v -> %v", before, got)
	}
	if r, _ := snap.Messages[3].ToolResult(); r != beforeResult {
		t.Fatalf("snapshot tool result changed: %+v", r)
	}
	if snap.ToolChoice.Mode != ToolChoiceFunction {
		t.Fatalf("snapshot tool choice changed: %+v", snap.ToolChoice)
	}
	if c.Snapshot().Version <= snap.Version {
		t.Fatal("version not bumped")
	}
}

func TestMarshalContextGolden(t *testing.T) {
	c := NewContext(
		System("Be brief."),
		User("What time is it?"),
		ToolCalls(ToolCallPart{ID: "c1", Name: "now", Arguments: json.RawMessage(`{}`)}),
		ToolResult("c1", "now", "12:00"),
	)
	c.SetTools(Tools{Functions: []FunctionSchema{{Name: "now", Description: "Current time"}}})
	c.SetToolChoice(ToolChoice{Mode: ToolChoiceAuto})
	data, err := MarshalContext(c)
	if err != nil {
		t.Fatal(err)
	}
	const golden = `{"version":1,"messages":[` +
		`{"role":"system","parts":[{"type":"text","text":"Be brief."}]},` +
		`{"role":"user","parts":[{"type":"text","text":"What time is it?"}]},` +
		`{"role":"assistant","parts":[{"type":"tool_call","name":"now","id":"c1","arguments":"{}"}]},` +
		`{"role":"tool","parts":[{"type":"tool_result","name":"now","tool_call_id":"c1","content":"12:00"}]}],` +
		`"tools":{"functions":[{"name":"now","description":"Current time"}]},` +
		`"tool_choice":{"mode":"auto"}}`
	if string(data) != golden {
		t.Fatalf("encoding changed:\n got %s\nwant %s", data, golden)
	}
	restored, err := UnmarshalContext([]byte(golden))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(restored.Messages(), c.Messages()) {
		t.Fatalf("golden decode = %#v", restored.Messages())
	}
}
