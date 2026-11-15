package vertex

import (
	"context"
	"fmt"
	"net/http"

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"

	"github.com/square-key-labs/strawgo-ai/src/services/gemini"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const (
	DefaultModel = "gemini-2.5-flash"

	cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
	processorName      = "VertexGemini"
)

type LLMConfig struct {
	ProjectID       string
	Location        string
	Model           string
	SystemPrompt    string
	Temperature     float64
	MaxTokens       int
	CredentialsJSON []byte
	Thinking        *gemini.Thinking
	SafetySettings  []map[string]any
	BaseURL         string
	HTTPClient      *http.Client
	Retry           gemini.RetryPolicy
}

func NewProvider(config LLMConfig) (*gemini.Provider, error) {
	if config.ProjectID == "" {
		return nil, fmt.Errorf("vertex: ProjectID is required")
	}
	if config.Location == "" {
		return nil, fmt.Errorf("vertex: Location is required")
	}
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	creds, err := credentials.DetectDefault(&credentials.DetectOptions{
		Scopes:          []string{cloudPlatformScope},
		CredentialsJSON: config.CredentialsJSON,
	})
	if err != nil {
		return nil, fmt.Errorf("vertex: load credentials: %w", err)
	}
	base := config.BaseURL
	if base == "" {
		base = Endpoint(config.Location)
	}
	prefix := fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/google/", base, config.ProjectID, config.Location)
	return gemini.NewProvider(gemini.ProviderConfig{
		Name:  processorName,
		Model: model,
		URL: func(m string) string {
			return prefix + modelPath(m) + ":streamGenerateContent?alt=sse"
		},
		Authorize:      bearer(creds),
		Thinking:       config.Thinking,
		SafetySettings: config.SafetySettings,
		HTTPClient:     config.HTTPClient,
		Retry:          config.Retry,
	}), nil
}

func NewLLMService(config LLMConfig, opts ...llmservice.Option) (*llmservice.Service, error) {
	p, err := NewProvider(config)
	if err != nil {
		return nil, err
	}
	model := config.Model
	if model == "" {
		model = DefaultModel
	}
	settings := gemini.SettingsFrom(model, config.SystemPrompt, config.Temperature, config.MaxTokens)
	return llmservice.New(p, append([]llmservice.Option{llmservice.WithName(processorName), llmservice.WithSettings(settings)}, opts...)...), nil
}

func Endpoint(location string) string {
	if location == "global" {
		return "https://aiplatform.googleapis.com"
	}
	return "https://" + location + "-aiplatform.googleapis.com"
}

func bearer(creds *auth.Credentials) func(ctx context.Context, h http.Header) error {
	return func(ctx context.Context, h http.Header) error {
		tok, err := creds.Token(ctx)
		if err != nil {
			return fmt.Errorf("vertex: token: %w", err)
		}
		h.Set("Authorization", "Bearer "+tok.Value)
		return nil
	}
}

func modelPath(model string) string {
	if len(model) > len("models/") && model[:len("models/")] == "models/" {
		return model
	}
	return "models/" + model
}
