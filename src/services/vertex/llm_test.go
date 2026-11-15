package vertex

import (
	"strings"
	"testing"
)

func TestNewLLMService_MissingProjectID(t *testing.T) {
	_, err := NewLLMService(LLMConfig{Location: "us-central1"})
	if err == nil || !strings.Contains(err.Error(), "ProjectID") {
		t.Fatalf("expected ProjectID error, got %v", err)
	}
}

func TestNewLLMService_MissingLocation(t *testing.T) {
	_, err := NewLLMService(LLMConfig{ProjectID: "test-project"})
	if err == nil || !strings.Contains(err.Error(), "Location") {
		t.Fatalf("expected Location error, got %v", err)
	}
}

func TestNewLLMService_InvalidCredentialsJSON(t *testing.T) {
	_, err := NewLLMService(LLMConfig{ProjectID: "p", Location: "us-central1", CredentialsJSON: []byte("not-json")})
	if err == nil || !strings.Contains(err.Error(), "credentials") {
		t.Fatalf("expected credentials error, got %v", err)
	}
}

func TestNewLLMService_Defaults(t *testing.T) {
	creds := []byte(`{"type":"authorized_user","client_id":"id","client_secret":"secret","refresh_token":"token"}`)
	svc, err := NewLLMService(LLMConfig{ProjectID: "p", Location: "asia-south1", SystemPrompt: "sys", CredentialsJSON: creds})
	if err != nil {
		t.Fatal(err)
	}
	s := svc.Settings()
	if s.Model != DefaultModel || s.SystemInstruction != "sys" {
		t.Errorf("settings = %+v", s)
	}
	if svc.Name() != processorName {
		t.Errorf("name = %q", svc.Name())
	}
}

func TestEndpoint(t *testing.T) {
	if got := Endpoint("asia-south1"); got != "https://asia-south1-aiplatform.googleapis.com" {
		t.Errorf("regional endpoint = %q", got)
	}
	if got := Endpoint("global"); got != "https://aiplatform.googleapis.com" {
		t.Errorf("global endpoint = %q", got)
	}
}

func TestDefaultModel_Constant(t *testing.T) {
	if !strings.HasPrefix(DefaultModel, "gemini-") {
		t.Errorf("DefaultModel = %q, expected gemini-* prefix", DefaultModel)
	}
}
