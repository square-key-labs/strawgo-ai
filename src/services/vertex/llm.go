// Package vertex provides a Gemini LLM service that targets Google Cloud
// Vertex AI (regional endpoints, OAuth2/ADC auth) rather than the public
// ai.google.dev endpoint.
//
// Use this service when you need:
//   - Regional co-location (e.g. asia-south1) to cut TTFT for voice agents.
//   - GCP service-account auth instead of an API key.
//   - Vertex-only features (implicit prompt caching, grounding, etc.).
//
// For the public Gemini API (API key, AI Studio), use the gemini package.
//
// Note: NewLLMService returns an error (unlike the gemini package) because
// credential loading and client construction can fail at init time.
package vertex

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"cloud.google.com/go/auth/credentials"
	"google.golang.org/genai"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

const (
	// DefaultModel is the recommended low-latency model for voice agents.
	DefaultModel = "gemini-2.5-flash"

	// cloudPlatformScope is the OAuth2 scope required by Vertex AI.
	cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// processorName is the name registered with the frame-graph runtime.
	processorName = "VertexGemini"

	// logPrefix prefixes all log lines from this service.
	logPrefix = "VertexGemini"

	// staleInterruptWindow suppresses InterruptionFrames that arrive immediately
	// after a fresh LLMContextFrame — those interruptions belong to the prior turn.
	staleInterruptWindow = 100 * time.Millisecond
)

// LLMService is a Gemini LLM service backed by Vertex AI.
type LLMService struct {
	*processors.BaseProcessor
	client      *genai.Client
	model       string
	temperature float64
	context     *services.LLMContext
	ctx         context.Context
	cancel      context.CancelFunc

	// Request-scoped context for cancellable streaming (protected by streamMu)
	requestCtx    context.Context
	requestCancel context.CancelFunc
	isGenerating  bool
	lastContextAt time.Time
	streamMu      sync.Mutex
	log           *logger.Logger
}

// LLMConfig configures a Vertex AI Gemini LLM service.
//
// Auth precedence:
//  1. CredentialsJSON (raw service-account JSON bytes) — preferred for k8s
//     secrets / env vars.
//  2. Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS file,
//     gcloud ADC, metadata server on GCE/GKE/Cloud Run).
type LLMConfig struct {
	// ProjectID is the GCP project. Required.
	ProjectID string

	// Location is the GCP region, e.g. "asia-south1", "us-central1". Required.
	Location string

	// Model is the Vertex model ID. Defaults to DefaultModel when empty.
	Model string

	// SystemPrompt seeds the conversation. Sent as systemInstruction per request.
	SystemPrompt string

	// Temperature in [0.0, 2.0].
	Temperature float64

	// CredentialsJSON is raw service-account JSON. Optional — falls back to ADC.
	CredentialsJSON []byte
}

// NewLLMService creates a Vertex AI Gemini LLM service.
func NewLLMService(config LLMConfig) (*LLMService, error) {
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

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend:     genai.BackendVertexAI,
		Project:     config.ProjectID,
		Location:    config.Location,
		Credentials: creds,
	})
	if err != nil {
		return nil, fmt.Errorf("vertex: create client: %w", err)
	}

	s := &LLMService{
		client:      client,
		model:       model,
		temperature: config.Temperature,
		context:     services.NewLLMContext(config.SystemPrompt),
		log:         logger.WithPrefix(logPrefix),
	}
	s.BaseProcessor = processors.NewBaseProcessor(processorName, s)
	return s, nil
}

func (s *LLMService) SetModel(model string)         { s.model = model }
func (s *LLMService) SetSystemPrompt(prompt string) { s.context.SystemPrompt = prompt }
func (s *LLMService) SetTemperature(temp float64)   { s.temperature = temp }

func (s *LLMService) AddMessage(role, content string) {
	s.context.Messages = append(s.context.Messages, services.LLMMessage{
		Role:    role,
		Content: content,
	})
}

func (s *LLMService) ClearContext() { s.context.Clear() }

func (s *LLMService) Initialize(ctx context.Context) error {
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.log.Info("Initialized Vertex Gemini with model %s", s.model)
	return nil
}

func (s *LLMService) Cleanup() error {
	if s.cancel != nil {
		s.cancel()
	}
	return nil
}

func (s *LLMService) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// InterruptionFrame: cancel in-flight stream unless a new context arrived
	// <staleInterruptWindow ago (interruption belongs to the prior turn).
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.streamMu.Lock()
		isGen := s.isGenerating
		hasCancel := s.requestCancel != nil
		timeSinceContext := time.Since(s.lastContextAt)
		isNewContext := timeSinceContext < staleInterruptWindow

		s.log.Debug("Interruption received (isGenerating=%v, hasCancel=%v, timeSinceContext=%v)", isGen, hasCancel, timeSinceContext)

		if isNewContext {
			s.log.Debug("Ignoring interruption - new context was just received (%v ago)", timeSinceContext)
			s.streamMu.Unlock()
			return s.PushFrame(frame, direction)
		}

		if isGen && hasCancel {
			s.log.Info("Cancelling ongoing stream")
			s.requestCancel()
			s.isGenerating = false
		}
		s.streamMu.Unlock()
		return s.PushFrame(frame, direction)
	}

	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			s.log.Info("Received LLMContextFrame with %d messages", len(llmContext.Messages))

			s.streamMu.Lock()
			s.lastContextAt = time.Now()
			s.streamMu.Unlock()

			s.context = llmContext

			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			if err := s.generateResponse(); err != nil {
				if s.requestCtx != nil && s.requestCtx.Err() == context.Canceled {
					s.log.Info("Stream cancelled by interruption")
				} else {
					s.log.Error("Error generating response: %v", err)
					s.PushFrame(frames.NewErrorFrame(err), frames.Upstream)
				}
			}

			s.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
		}
		return nil
	}

	return s.PushFrame(frame, direction)
}

func (s *LLMService) generateResponse() error {
	parentCtx := s.ctx
	if parentCtx == nil {
		parentCtx = context.Background()
	}

	s.streamMu.Lock()
	s.requestCtx, s.requestCancel = context.WithCancel(parentCtx)
	s.isGenerating = true
	s.streamMu.Unlock()

	s.log.Info("Starting Vertex stream generation")
	defer func() {
		s.streamMu.Lock()
		s.isGenerating = false
		if s.requestCancel != nil {
			s.requestCancel()
		}
		s.requestCancel = nil
		s.streamMu.Unlock()
	}()

	contents := buildContents(s.context.Messages, s.log)

	temp := float32(s.temperature)
	cfg := &genai.GenerateContentConfig{
		Temperature: &temp,
	}
	if s.context.SystemPrompt != "" {
		// Role is ignored by the API for system instructions; omit for clarity.
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: s.context.SystemPrompt}},
		}
	}

	var fullResponse strings.Builder
	stream := s.client.Models.GenerateContentStream(s.requestCtx, s.model, contents, cfg)

	for resp, err := range stream {
		if s.requestCtx.Err() == context.Canceled {
			s.log.Info("Stream interrupted mid-generation (tokens so far: %d chars)", fullResponse.Len())
			return nil
		}
		if err != nil {
			return fmt.Errorf("vertex stream: %w", err)
		}
		if resp == nil {
			continue
		}

		text := resp.Text()
		if text == "" {
			continue
		}

		fullResponse.WriteString(text)
		s.PushFrame(frames.NewLLMTextFrame(text), frames.Downstream)
	}

	response := fullResponse.String()
	s.context.AddAssistantMessage(response)
	s.log.Debug("Assistant response length: %d", len(response))

	return nil
}

// buildContents maps LLMContext messages to genai.Content.
// System messages are handled separately via GenerateContentConfig.SystemInstruction.
// Unknown roles are dropped with a warning to avoid API-side rejections.
func buildContents(messages []services.LLMMessage, log *logger.Logger) []*genai.Content {
	out := make([]*genai.Content, 0, len(messages))
	for _, msg := range messages {
		role, ok := normalizeRole(msg.Role)
		if !ok {
			if log != nil {
				log.Warn("Dropping message with unsupported role %q", msg.Role)
			}
			continue
		}
		out = append(out, genai.NewContentFromText(msg.Content, role))
	}
	return out
}

// normalizeRole maps LLMContext roles to genai roles. Returns false for
// unsupported roles (tool, function, system — the last is handled elsewhere).
func normalizeRole(role string) (genai.Role, bool) {
	switch role {
	case "user", "developer":
		return genai.RoleUser, true
	case "assistant", "model":
		return genai.RoleModel, true
	default:
		return "", false
	}
}
