package vertex

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/square-key-labs/strawgo-ai/src/services"
)

// --- Config validation ---

func TestNewLLMService_MissingProjectID(t *testing.T) {
	_, err := NewLLMService(LLMConfig{
		Location: "us-central1",
	})
	if err == nil {
		t.Fatal("expected error when ProjectID is missing")
	}
	if !strings.Contains(err.Error(), "ProjectID") {
		t.Errorf("expected error to mention ProjectID, got: %v", err)
	}
}

func TestNewLLMService_MissingLocation(t *testing.T) {
	_, err := NewLLMService(LLMConfig{
		ProjectID: "test-project",
	})
	if err == nil {
		t.Fatal("expected error when Location is missing")
	}
	if !strings.Contains(err.Error(), "Location") {
		t.Errorf("expected error to mention Location, got: %v", err)
	}
}

func TestNewLLMService_InvalidCredentialsJSON(t *testing.T) {
	_, err := NewLLMService(LLMConfig{
		ProjectID:       "test-project",
		Location:        "us-central1",
		CredentialsJSON: []byte("not-json"),
	})
	if err == nil {
		t.Fatal("expected error with invalid CredentialsJSON")
	}
	if !strings.Contains(err.Error(), "credentials") {
		t.Errorf("expected error to mention credentials, got: %v", err)
	}
}

// --- Role mapping ---

func TestNormalizeRole(t *testing.T) {
	tests := []struct {
		input   string
		want    genai.Role
		wantOK  bool
	}{
		{"user", genai.RoleUser, true},
		{"developer", genai.RoleUser, true},
		{"assistant", genai.RoleModel, true},
		{"model", genai.RoleModel, true},
		{"system", "", false},
		{"tool", "", false},
		{"function", "", false},
		{"", "", false},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got, ok := normalizeRole(tc.input)
			if ok != tc.wantOK {
				t.Errorf("normalizeRole(%q) ok = %v, want %v", tc.input, ok, tc.wantOK)
			}
			if got != tc.want {
				t.Errorf("normalizeRole(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// --- buildContents ---

func TestBuildContents_MapsKnownRoles(t *testing.T) {
	msgs := []services.LLMMessage{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "developer", Content: "dev note"},
		{Role: "model", Content: "from model"},
	}

	out := buildContents(msgs, nil)

	if len(out) != 4 {
		t.Fatalf("expected 4 contents, got %d", len(out))
	}
	wantRoles := []genai.Role{genai.RoleUser, genai.RoleModel, genai.RoleUser, genai.RoleModel}
	for i, c := range out {
		if genai.Role(c.Role) != wantRoles[i] {
			t.Errorf("msg[%d] role = %q, want %q", i, c.Role, wantRoles[i])
		}
		if len(c.Parts) != 1 || c.Parts[0].Text != msgs[i].Content {
			t.Errorf("msg[%d] content mismatch: %+v", i, c.Parts)
		}
	}
}

func TestBuildContents_DropsUnsupportedRoles(t *testing.T) {
	msgs := []services.LLMMessage{
		{Role: "user", Content: "keep"},
		{Role: "system", Content: "drop-system"},
		{Role: "tool", Content: "drop-tool"},
		{Role: "function", Content: "drop-function"},
		{Role: "assistant", Content: "keep-asst"},
	}

	out := buildContents(msgs, nil)

	if len(out) != 2 {
		t.Fatalf("expected 2 contents after filter, got %d", len(out))
	}
	if out[0].Parts[0].Text != "keep" {
		t.Errorf("out[0] = %q, want %q", out[0].Parts[0].Text, "keep")
	}
	if out[1].Parts[0].Text != "keep-asst" {
		t.Errorf("out[1] = %q, want %q", out[1].Parts[0].Text, "keep-asst")
	}
}

func TestBuildContents_EmptyInput(t *testing.T) {
	out := buildContents(nil, nil)
	if len(out) != 0 {
		t.Errorf("expected empty output, got %d", len(out))
	}
}

// --- Setters & context management ---
//
// These exercise the LLMService methods without requiring a live Vertex client.
// We construct LLMService directly to bypass credential loading.

func newTestService(t *testing.T, systemPrompt string) *LLMService {
	t.Helper()
	return &LLMService{
		context: services.NewLLMContext(systemPrompt),
	}
}

func TestSetters(t *testing.T) {
	s := newTestService(t, "")
	s.SetModel("gemini-2.5-pro")
	if s.model != "gemini-2.5-pro" {
		t.Errorf("model = %q, want gemini-2.5-pro", s.model)
	}

	s.SetSystemPrompt("new prompt")
	if s.context.SystemPrompt != "new prompt" {
		t.Errorf("SystemPrompt = %q, want 'new prompt'", s.context.SystemPrompt)
	}

	s.SetTemperature(0.9)
	if s.temperature != 0.9 {
		t.Errorf("temperature = %v, want 0.9", s.temperature)
	}
}

func TestMessageManagement(t *testing.T) {
	s := newTestService(t, "system")

	s.AddMessage("user", "hi")
	s.AddMessage("assistant", "hello")

	if len(s.context.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(s.context.Messages))
	}
	if s.context.Messages[0].Role != "user" || s.context.Messages[0].Content != "hi" {
		t.Errorf("msg[0] = %+v", s.context.Messages[0])
	}

	s.ClearContext()
	if len(s.context.Messages) != 0 {
		t.Errorf("after ClearContext, expected 0 messages, got %d", len(s.context.Messages))
	}
}

func TestDefaultModel_Constant(t *testing.T) {
	if DefaultModel == "" {
		t.Error("DefaultModel must not be empty")
	}
	if !strings.HasPrefix(DefaultModel, "gemini-") {
		t.Errorf("DefaultModel = %q, expected gemini-* prefix", DefaultModel)
	}
}
