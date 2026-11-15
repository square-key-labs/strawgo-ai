package tracing

import (
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

const (
	KindSTT = "stt"
	KindLLM = "llm"
	KindTTS = "tts"
)

var providerOverrides = map[string]string{
	"awsbedrock":   "aws.bedrock",
	"azure":        "azure.ai.openai",
	"google":       "gcp.gemini",
	"gemini":       "gcp.gemini",
	"googlevertex": "gcp.vertex_ai",
	"vertex":       "gcp.vertex_ai",
	"grok":         "x_ai",
	"xai":          "x_ai",
	"mistral":      "mistral_ai",
}

func ProviderName(processorName string) string {
	name := processorName
	if i := strings.IndexAny(name, "#:"); i > 0 {
		name = name[:i]
	}
	for _, suffix := range []string{"LLMService", "STTService", "TTSService", "Service", "LLM", "STT", "TTS"} {
		if len(name) > len(suffix) && strings.HasSuffix(name, suffix) {
			name = name[:len(name)-len(suffix)]
			break
		}
	}
	name = strings.ToLower(strings.Trim(name, "-_ "))
	if mapped, ok := providerOverrides[name]; ok {
		return mapped
	}
	return name
}

func UsageAttributes(u llm.Usage) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		attribute.Int("gen_ai.usage.input_tokens", u.PromptTokens),
		attribute.Int("gen_ai.usage.output_tokens", u.CompletionTokens),
	}
	if u.CacheReadTokens > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.cache_read.input_tokens", u.CacheReadTokens))
	}
	if u.CacheCreationTokens > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.cache_creation.input_tokens", u.CacheCreationTokens))
	}
	if u.ReasoningTokens > 0 {
		attrs = append(attrs, attribute.Int("gen_ai.usage.reasoning.output_tokens", u.ReasoningTokens))
	}
	return attrs
}

func serviceStartAttributes(kind, provider, model string) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, 6)
	attrs = append(attrs, attribute.String("gen_ai.provider.name", provider))
	if model != "" {
		attrs = append(attrs, attribute.String("gen_ai.request.model", model))
	}
	switch kind {
	case KindLLM:
		attrs = append(attrs,
			attribute.String("gen_ai.operation.name", "chat"),
			attribute.String("gen_ai.output.type", "text"),
			attribute.Bool("stream", true),
		)
	case KindTTS:
		attrs = append(attrs,
			attribute.String("gen_ai.operation.name", "tts"),
			attribute.String("gen_ai.output.type", "speech"),
		)
	case KindSTT:
		attrs = append(attrs, attribute.String("gen_ai.operation.name", "stt"))
	}
	return attrs
}

func setTTFB(span trace.Span, d time.Duration) {
	span.SetAttributes(attribute.Float64("metrics.ttfb", d.Seconds()))
}
