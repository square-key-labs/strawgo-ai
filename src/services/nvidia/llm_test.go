package nvidia

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func TestInlineThinkTagsBecomeThoughts(t *testing.T) {
	chunks := []string{"<thi", "nk>The user greets", " me.</th", "ink>\n\n", "Hi ", "there!"}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", c)
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	p := NewProvider(LLMConfig{APIKey: "k", BaseURL: srv.URL, HTTPClient: srv.Client()})
	var text, thought strings.Builder
	order := ""
	err := p.Stream(context.Background(), llm.Snapshot{}, llm.Settings{Model: DefaultModel}, func(e llm.Event) {
		switch ev := e.(type) {
		case llm.TextDelta:
			text.WriteString(ev.Text)
			order += "t"
		case llm.ThoughtDelta:
			thought.WriteString(ev.Text)
			order += "r"
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if text.String() != "Hi there!" || thought.String() != "The user greets me." {
		t.Fatalf("text = %q thought = %q", text.String(), thought.String())
	}
	if strings.Contains(order, "tr") {
		t.Fatalf("thought emitted after answer text: %s", order)
	}
}
