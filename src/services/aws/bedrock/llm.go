package bedrock

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/transport/http/protocol/restjson1"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/aws/awsauth"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const (
	DefaultModel     = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
	DefaultMaxTokens = 4096

	ExtraAdditionalModelRequestFields = "additional_model_request_fields"
	ExtraStopSequences                = "stop_sequences"
	ExtraLatency                      = "latency"
	ExtraPromptCaching                = "enable_prompt_caching"
)

var prefillSupported = []string{
	"claude-2",
	"claude-instant",
	"claude-3",
	"claude-opus-4-0",
	"claude-opus-4-1",
	"claude-sonnet-4-0",
	"claude-sonnet-4-5",
	"claude-haiku-4-5",
}

type LLMConfig struct {
	AWS                          awsauth.Config
	Model                        string
	SystemPrompt                 string
	Temperature                  *float64
	TopP                         *float64
	MaxTokens                    int
	StopSequences                []string
	Latency                      string
	EnablePromptCaching          bool
	AdditionalModelRequestFields map[string]any
}

type Provider struct {
	client  *bedrockruntime.Client
	model   string
	stops   []string
	latency string
	caching bool
	extra   map[string]any
}

func NewProvider(ctx context.Context, cfg LLMConfig) (*Provider, error) {
	awsCfg, err := awsauth.Load(ctx, cfg.AWS)
	if err != nil {
		return nil, fmt.Errorf("bedrock: load aws config: %w", err)
	}
	client := bedrockruntime.NewFromConfig(awsCfg, func(o *bedrockruntime.Options) {
		if ep := cfg.AWS.BaseEndpoint(); ep != nil {
			o.BaseEndpoint = ep
		}
	})
	return NewProviderWithClient(client, cfg), nil
}

func NewProviderWithClient(client *bedrockruntime.Client, cfg LLMConfig) *Provider {
	model := cfg.Model
	if model == "" {
		model = DefaultModel
	}
	return &Provider{
		client:  client,
		model:   model,
		stops:   cfg.StopSequences,
		latency: cfg.Latency,
		caching: cfg.EnablePromptCaching,
		extra:   cfg.AdditionalModelRequestFields,
	}
}

func NewLLMService(ctx context.Context, cfg LLMConfig, opts ...llmservice.Option) (*llmservice.Service, error) {
	p, err := NewProvider(ctx, cfg)
	if err != nil {
		return nil, err
	}
	return newService(p, cfg, opts), nil
}

func NewLLMServiceWithClient(client *bedrockruntime.Client, cfg LLMConfig, opts ...llmservice.Option) *llmservice.Service {
	return newService(NewProviderWithClient(client, cfg), cfg, opts)
}

func newService(p *Provider, cfg LLMConfig, opts []llmservice.Option) *llmservice.Service {
	settings := llm.Settings{Model: p.model, SystemInstruction: cfg.SystemPrompt, Temperature: cfg.Temperature, TopP: cfg.TopP}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	settings.MaxTokens = &maxTokens
	return llmservice.New(p, append([]llmservice.Option{llmservice.WithName("AWSBedrock"), llmservice.WithSettings(settings)}, opts...)...)
}

func (p *Provider) Name() string { return "AWSBedrock" }

type toolBlock struct {
	id   string
	name string
	args []byte
}

func (p *Provider) Stream(ctx context.Context, snapshot llm.Snapshot, settings llm.Settings, emit func(llm.Event)) error {
	input, err := p.buildInput(snapshot, settings)
	if err != nil {
		return err
	}
	out, err := p.client.ConverseStream(ctx, input, isolateCodec)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("bedrock: %w", err)
	}
	stream := out.GetStream()
	defer stream.Close()
	requestID, _ := awsmiddleware.GetRequestIDMetadata(out.ResultMetadata)
	if requestID != "" {
		emit(llm.ResponseInfo{RequestID: requestID})
	}

	tools := map[int32]*toolBlock{}
	var usage llm.Usage
	gotUsage := false
	stopReason := ""
	events := stream.Events()
	for {
		var ev types.ConverseStreamOutput
		var ok bool
		select {
		case <-ctx.Done():
			return ctx.Err()
		case ev, ok = <-events:
		}
		if !ok {
			break
		}
		switch e := ev.(type) {
		case *types.ConverseStreamOutputMemberContentBlockStart:
			if start, isTool := e.Value.Start.(*types.ContentBlockStartMemberToolUse); isTool {
				tools[aws.ToInt32(e.Value.ContentBlockIndex)] = &toolBlock{id: aws.ToString(start.Value.ToolUseId), name: aws.ToString(start.Value.Name)}
			}
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			switch d := e.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				if d.Value != "" {
					emit(llm.TextDelta{Text: d.Value})
				}
			case *types.ContentBlockDeltaMemberToolUse:
				if tb := tools[aws.ToInt32(e.Value.ContentBlockIndex)]; tb != nil && d.Value.Input != nil {
					tb.args = append(tb.args, *d.Value.Input...)
				}
			case *types.ContentBlockDeltaMemberReasoningContent:
				switch r := d.Value.(type) {
				case *types.ReasoningContentBlockDeltaMemberText:
					if r.Value != "" {
						emit(llm.ThoughtDelta{Text: r.Value})
					}
				case *types.ReasoningContentBlockDeltaMemberSignature:
					if r.Value != "" {
						emit(llm.ThoughtDelta{Signature: r.Value})
					}
				}
			}
		case *types.ConverseStreamOutputMemberContentBlockStop:
			idx := aws.ToInt32(e.Value.ContentBlockIndex)
			if tb := tools[idx]; tb != nil {
				delete(tools, idx)
				emit(tb.call())
			}
		case *types.ConverseStreamOutputMemberMessageStop:
			stopReason = string(e.Value.StopReason)
		case *types.ConverseStreamOutputMemberMetadata:
			if u := e.Value.Usage; u != nil {
				usage.PromptTokens = int(aws.ToInt32(u.InputTokens))
				usage.CompletionTokens = int(aws.ToInt32(u.OutputTokens))
				usage.CacheReadTokens = int(aws.ToInt32(u.CacheReadInputTokens))
				usage.CacheCreationTokens = int(aws.ToInt32(u.CacheWriteInputTokens))
				usage.TotalTokens = int(aws.ToInt32(u.TotalTokens))
				if usage.TotalTokens == 0 {
					usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens + usage.CacheReadTokens + usage.CacheCreationTokens
				}
				gotUsage = true
			}
		}
	}
	if err := stream.Err(); err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("bedrock: %w", err)
	}
	if stopReason == "" {
		return errors.New("bedrock: stream ended before messageStop")
	}
	if gotUsage {
		emit(llm.UsageEvent{Usage: usage})
	}
	if category, refused := refusalCategory(stopReason); refused {
		r := llm.Refusal{Category: category, Provider: p.Name(), Reason: stopReason}
		emit(r)
		emit(llm.Done{StopReason: llm.StopReasonContentFilter, RawStopReason: stopReason})
		return &llm.RefusalError{Refusal: r}
	}
	emit(llm.Done{StopReason: normalizeStopReason(stopReason), RawStopReason: stopReason})
	return nil
}

func refusalCategory(stopReason string) (llm.RefusalCategory, bool) {
	switch types.StopReason(stopReason) {
	case types.StopReasonGuardrailIntervened:
		return llm.RefusalGuardrail, true
	case types.StopReasonContentFiltered:
		return llm.RefusalContentFilter, true
	}
	return "", false
}

func normalizeStopReason(stopReason string) string {
	switch types.StopReason(stopReason) {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return llm.StopReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return llm.StopReasonLength
	case types.StopReasonToolUse:
		return llm.StopReasonToolCalls
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return llm.StopReasonContentFilter
	}
	return llm.StopReasonOther
}

func outputConfig(f *llm.ResponseFormat) (*types.OutputConfig, error) {
	if !f.Structured() {
		return nil, nil
	}
	if f.Type != llm.ResponseFormatJSONSchema || len(f.Schema) == 0 {
		return nil, fmt.Errorf("bedrock: %w without a JSON schema", llm.ErrResponseFormatUnsupported)
	}
	schema, err := json.Marshal(f.Schema)
	if err != nil {
		return nil, fmt.Errorf("bedrock: encode response schema: %w", err)
	}
	def := types.JsonSchemaDefinition{Schema: aws.String(string(schema)), Name: aws.String(f.SchemaName())}
	if f.Description != "" {
		def.Description = aws.String(f.Description)
	}
	return &types.OutputConfig{TextFormat: &types.OutputFormat{
		Type:      types.OutputFormatTypeJsonSchema,
		Structure: &types.OutputFormatStructureMemberJsonSchema{Value: def},
	}}, nil
}

func isolateCodec(o *bedrockruntime.Options) {
	o.Protocol = restjson1.New(nil)
}

func (tb *toolBlock) call() llm.ToolCall {
	args := tb.args
	if len(bytes.TrimSpace(args)) == 0 {
		args = []byte("{}")
	}
	return llm.ToolCall{ID: tb.id, Name: tb.name, Arguments: args}
}

func (p *Provider) buildInput(snapshot llm.Snapshot, settings llm.Settings) (*bedrockruntime.ConverseStreamInput, error) {
	model := settings.Model
	if model == "" {
		model = p.model
	}
	caching := p.caching
	if v, ok := settings.Extra[ExtraPromptCaching].(bool); ok {
		caching = v
	}
	latency := p.latency
	if v, ok := settings.Extra[ExtraLatency].(string); ok {
		latency = v
	}

	system, messages, hasToolContent := convertMessages(snapshot.Messages, settings.SystemInstruction)
	if needsTrailingUser(model) && len(messages) > 0 && messages[len(messages)-1].Role == types.ConversationRoleAssistant {
		messages = append(messages, types.Message{Role: types.ConversationRoleUser, Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "."}}})
	}
	if len(messages) == 0 {
		return nil, errors.New("bedrock: context has no messages")
	}
	input := &bedrockruntime.ConverseStreamInput{
		ModelId:         aws.String(model),
		Messages:        messages,
		InferenceConfig: p.inferenceConfig(settings),
	}
	oc, err := outputConfig(settings.ResponseFormat)
	if err != nil {
		return nil, err
	}
	input.OutputConfig = oc
	if len(system) > 0 {
		if caching {
			system = append(system, &types.SystemContentBlockMemberCachePoint{Value: types.CachePointBlock{Type: types.CachePointTypeDefault}})
		}
		input.System = system
	}
	if tc := toolConfig(snapshot, hasToolContent, caching); tc != nil {
		input.ToolConfig = tc
	}
	if latency == string(types.PerformanceConfigLatencyStandard) || latency == string(types.PerformanceConfigLatencyOptimized) {
		input.PerformanceConfig = &types.PerformanceConfiguration{Latency: types.PerformanceConfigLatency(latency)}
	}
	if fields := p.additionalFields(settings.Extra); len(fields) > 0 {
		input.AdditionalModelRequestFields = document.NewLazyDocument(fields)
	}
	return input, nil
}

func (p *Provider) inferenceConfig(settings llm.Settings) *types.InferenceConfiguration {
	ic := &types.InferenceConfiguration{MaxTokens: aws.Int32(DefaultMaxTokens)}
	if settings.MaxTokens != nil && *settings.MaxTokens > 0 {
		ic.MaxTokens = aws.Int32(int32(*settings.MaxTokens))
	}
	if settings.Temperature != nil {
		ic.Temperature = aws.Float32(float32(*settings.Temperature))
	}
	if settings.TopP != nil {
		ic.TopP = aws.Float32(float32(*settings.TopP))
	}
	ic.StopSequences = p.stops
	switch v := settings.Extra[ExtraStopSequences].(type) {
	case []string:
		ic.StopSequences = v
	case []any:
		stops := make([]string, 0, len(v))
		for _, s := range v {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			}
		}
		ic.StopSequences = stops
	}
	return ic
}

func (p *Provider) additionalFields(extra map[string]any) map[string]any {
	out := make(map[string]any, len(p.extra)+len(extra))
	for k, v := range p.extra {
		out[k] = v
	}
	for k, v := range extra {
		switch k {
		case ExtraPromptCaching, ExtraLatency, ExtraStopSequences:
		case ExtraAdditionalModelRequestFields:
			if m, ok := v.(map[string]any); ok {
				for mk, mv := range m {
					out[mk] = mv
				}
			}
		default:
			out[k] = v
		}
	}
	return out
}

func needsTrailingUser(model string) bool {
	if !strings.Contains(model, "claude") {
		return false
	}
	for _, p := range prefillSupported {
		if strings.Contains(model, p) {
			return false
		}
	}
	return true
}

func toolConfig(snapshot llm.Snapshot, hasToolContent, caching bool) *types.ToolConfiguration {
	tools := convertTools(snapshot.Tools)
	choice := snapshot.ToolChoice
	if choice.Mode == llm.ToolChoiceNone {
		tools = nil
	}
	if len(tools) == 0 {
		if !hasToolContent {
			return nil
		}
		return &types.ToolConfiguration{Tools: []types.Tool{noOpTool()}}
	}
	if caching {
		tools = append(tools, &types.ToolMemberCachePoint{Value: types.CachePointBlock{Type: types.CachePointTypeDefault}})
	}
	tc := &types.ToolConfiguration{Tools: tools}
	switch choice.Mode {
	case llm.ToolChoiceAuto:
		tc.ToolChoice = &types.ToolChoiceMemberAuto{}
	case llm.ToolChoiceRequired:
		tc.ToolChoice = &types.ToolChoiceMemberAny{}
	case llm.ToolChoiceFunction:
		if choice.Name != "" {
			tc.ToolChoice = &types.ToolChoiceMemberTool{Value: types.SpecificToolChoice{Name: aws.String(choice.Name)}}
		}
	}
	return tc
}

func noOpTool() types.Tool {
	return &types.ToolMemberToolSpec{Value: types.ToolSpecification{
		Name:        aws.String("no_operation"),
		Description: aws.String("Internal placeholder function. Do not call this function."),
		InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(map[string]any{"type": "object", "properties": map[string]any{}, "required": []string{}})},
	}}
}

type rawToolSpec struct {
	ToolSpec    *rawToolSpec `json:"toolSpec"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	InputSchema struct {
		JSON map[string]any `json:"json"`
	} `json:"inputSchema"`
}

func convertTools(tools llm.Tools) []types.Tool {
	out := make([]types.Tool, 0, len(tools.Functions))
	for _, f := range tools.Functions {
		spec := types.ToolSpecification{
			Name:        aws.String(f.Name),
			InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(f.JSONSchema())},
		}
		if f.Description != "" {
			spec.Description = aws.String(f.Description)
		}
		if f.Strict {
			spec.Strict = aws.Bool(true)
		}
		out = append(out, &types.ToolMemberToolSpec{Value: spec})
	}
	for _, raw := range tools.Custom["bedrock"] {
		var t rawToolSpec
		if json.Unmarshal(raw, &t) != nil {
			continue
		}
		if t.ToolSpec != nil {
			t = *t.ToolSpec
		}
		if t.Name == "" {
			continue
		}
		schema := t.InputSchema.JSON
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		spec := types.ToolSpecification{Name: aws.String(t.Name), InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)}}
		if t.Description != "" {
			spec.Description = aws.String(t.Description)
		}
		out = append(out, &types.ToolMemberToolSpec{Value: spec})
	}
	return out
}

func convertMessages(msgs []llm.Message, instruction string) ([]types.SystemContentBlock, []types.Message, bool) {
	var system []types.SystemContentBlock
	if strings.TrimSpace(instruction) != "" {
		system = append(system, &types.SystemContentBlockMemberText{Value: instruction})
	}
	i := 0
	for ; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role != llm.RoleSystem && m.Role != llm.RoleDeveloper {
			break
		}
		if text := m.Text(); strings.TrimSpace(text) != "" {
			system = append(system, &types.SystemContentBlockMemberText{Value: text})
		}
	}

	hasToolContent := false
	var out []types.Message
	for _, m := range msgs[i:] {
		role := types.ConversationRoleUser
		if m.Role == llm.RoleAssistant {
			role = types.ConversationRoleAssistant
		}
		blocks, tool := convertParts(m)
		hasToolContent = hasToolContent || tool
		if len(blocks) == 0 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			continue
		}
		out = append(out, types.Message{Role: role, Content: blocks})
	}
	if len(out) > 0 && out[0].Role == types.ConversationRoleAssistant {
		out = append([]types.Message{{Role: types.ConversationRoleUser, Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "(empty)"}}}}, out...)
	}
	for _, m := range out {
		if m.Role == types.ConversationRoleUser {
			hoistToolResults(m.Content)
		}
	}
	return system, out, hasToolContent
}

func hoistToolResults(blocks []types.ContentBlock) {
	var results, rest []types.ContentBlock
	for _, b := range blocks {
		if _, ok := b.(*types.ContentBlockMemberToolResult); ok {
			results = append(results, b)
		} else {
			rest = append(rest, b)
		}
	}
	if len(results) == 0 || len(rest) == 0 {
		return
	}
	n := copy(blocks, results)
	copy(blocks[n:], rest)
}

func convertParts(m llm.Message) ([]types.ContentBlock, bool) {
	blocks := make([]types.ContentBlock, 0, len(m.Parts))
	tool := false
	for _, part := range m.Parts {
		switch p := part.(type) {
		case llm.TextPart:
			if strings.TrimSpace(p.Text) != "" {
				blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
			}
		case llm.ImagePart:
			if b := mediaBlock(p.URL, p.Data, p.MIME, ""); b != nil {
				blocks = append(blocks, b)
			}
		case llm.FilePart:
			if b := mediaBlock(p.URL, p.Data, p.MIME, p.Name); b != nil {
				blocks = append(blocks, b)
			}
		case llm.ToolCallPart:
			tool = true
			var input any = map[string]any{}
			if len(bytes.TrimSpace(p.Arguments)) > 0 {
				var v any
				if json.Unmarshal(p.Arguments, &v) == nil && v != nil {
					input = v
				}
			}
			blocks = append(blocks, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(p.ID),
				Name:      aws.String(p.Name),
				Input:     document.NewLazyDocument(input),
			}})
		case llm.ToolResultPart:
			tool = true
			content := p.Content
			if strings.TrimSpace(content) == "" {
				content = "(empty)"
			}
			r := types.ToolResultBlock{
				ToolUseId: aws.String(p.ToolCallID),
				Content:   []types.ToolResultContentBlock{&types.ToolResultContentBlockMemberText{Value: content}},
			}
			if p.IsError {
				r.Status = types.ToolResultStatusError
			}
			blocks = append(blocks, &types.ContentBlockMemberToolResult{Value: r})
		case llm.ThoughtPart:
			if m.Role != llm.RoleAssistant {
				continue
			}
			if p.Redacted {
				data, err := base64.StdEncoding.DecodeString(p.Signature)
				if err == nil && len(data) > 0 {
					blocks = append(blocks, &types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberRedactedContent{Value: data}})
				}
				continue
			}
			if p.Signature != "" {
				blocks = append(blocks, &types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberReasoningText{Value: types.ReasoningTextBlock{Text: aws.String(p.Text), Signature: aws.String(p.Signature)}}})
			}
		}
	}
	return blocks, tool
}

var imageFormats = map[string]types.ImageFormat{
	"image/png":  types.ImageFormatPng,
	"image/jpeg": types.ImageFormatJpeg,
	"image/jpg":  types.ImageFormatJpeg,
	"image/gif":  types.ImageFormatGif,
	"image/webp": types.ImageFormatWebp,
}

var documentFormats = map[string]types.DocumentFormat{
	"application/pdf":    types.DocumentFormatPdf,
	"text/csv":           types.DocumentFormatCsv,
	"application/msword": types.DocumentFormatDoc,
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document": types.DocumentFormatDocx,
	"application/vnd.ms-excel": types.DocumentFormatXls,
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": types.DocumentFormatXlsx,
	"text/html":     types.DocumentFormatHtml,
	"text/plain":    types.DocumentFormatTxt,
	"text/markdown": types.DocumentFormatMd,
}

func mediaBlock(url string, data []byte, mime, name string) types.ContentBlock {
	var s3 string
	switch {
	case len(data) > 0:
	case strings.HasPrefix(url, "data:"):
		meta, payload, ok := strings.Cut(strings.TrimPrefix(url, "data:"), ",")
		if !ok {
			return nil
		}
		decoded, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return nil
		}
		data = decoded
		if mime == "" {
			mime = strings.TrimSuffix(meta, ";base64")
		}
	case strings.HasPrefix(url, "s3://"):
		s3 = url
	default:
		return nil
	}
	if mime == "" && len(data) > 0 {
		mime = http.DetectContentType(data)
	}
	mime, _, _ = strings.Cut(mime, ";")
	if f, ok := imageFormats[mime]; ok {
		img := types.ImageBlock{Format: f}
		if s3 != "" {
			img.Source = &types.ImageSourceMemberS3Location{Value: types.S3Location{Uri: aws.String(s3)}}
		} else {
			img.Source = &types.ImageSourceMemberBytes{Value: data}
		}
		return &types.ContentBlockMemberImage{Value: img}
	}
	f, ok := documentFormats[mime]
	if !ok {
		return nil
	}
	doc := types.DocumentBlock{Format: f, Name: aws.String(documentName(name))}
	if s3 != "" {
		doc.Source = &types.DocumentSourceMemberS3Location{Value: types.S3Location{Uri: aws.String(s3)}}
	} else {
		doc.Source = &types.DocumentSourceMemberBytes{Value: data}
	}
	return &types.ContentBlockMemberDocument{Value: doc}
}

func documentName(name string) string {
	if i := strings.LastIndexByte(name, '.'); i > 0 {
		name = name[:i]
	}
	var b strings.Builder
	space := false
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '(', r == ')', r == '[', r == ']':
			b.WriteRune(r)
			space = false
		default:
			if !space && b.Len() > 0 {
				b.WriteByte(' ')
				space = true
			}
		}
	}
	out := strings.TrimSpace(b.String())
	if out == "" {
		return "document"
	}
	return out
}
