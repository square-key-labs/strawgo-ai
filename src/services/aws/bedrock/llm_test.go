package bedrock

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

func TestStopReasonMapping(t *testing.T) {
	cases := []struct {
		raw      types.StopReason
		stop     string
		category llm.RefusalCategory
		refused  bool
	}{
		{types.StopReasonEndTurn, llm.StopReasonStop, "", false},
		{types.StopReasonStopSequence, llm.StopReasonStop, "", false},
		{types.StopReasonMaxTokens, llm.StopReasonLength, "", false},
		{types.StopReasonToolUse, llm.StopReasonToolCalls, "", false},
		{types.StopReasonContentFiltered, llm.StopReasonContentFilter, llm.RefusalContentFilter, true},
		{types.StopReasonGuardrailIntervened, llm.StopReasonContentFilter, llm.RefusalGuardrail, true},
		{types.StopReasonMalformedToolUse, llm.StopReasonOther, "", false},
	}
	for _, tc := range cases {
		if got := normalizeStopReason(string(tc.raw)); got != tc.stop {
			t.Errorf("%s normalised to %q, want %q", tc.raw, got, tc.stop)
		}
		category, refused := refusalCategory(string(tc.raw))
		if category != tc.category || refused != tc.refused {
			t.Errorf("%s refusal = %q %v", tc.raw, category, refused)
		}
	}
}

func TestStructuredOutputConfig(t *testing.T) {
	p := NewProviderWithClient(nil, LLMConfig{})
	snap := llm.NewContext(llm.User("hi")).Snapshot()
	schema := map[string]any{"type": "object", "properties": map[string]any{"n": map[string]any{"type": "integer"}}}
	format := llm.JSONSchemaFormat("number", schema)
	format.Description = "a number"

	input, err := p.buildInput(snap, llm.Settings{ResponseFormat: format})
	if err != nil {
		t.Fatal(err)
	}
	if input.OutputConfig == nil || input.OutputConfig.TextFormat == nil || input.OutputConfig.TextFormat.Type != types.OutputFormatTypeJsonSchema {
		t.Fatalf("output config = %+v", input.OutputConfig)
	}
	js, ok := input.OutputConfig.TextFormat.Structure.(*types.OutputFormatStructureMemberJsonSchema)
	if !ok || aws.ToString(js.Value.Name) != "number" || aws.ToString(js.Value.Description) != "a number" {
		t.Fatalf("structure = %+v", input.OutputConfig.TextFormat.Structure)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(aws.ToString(js.Value.Schema)), &decoded); err != nil || decoded["type"] != "object" {
		t.Fatalf("schema = %q err = %v", aws.ToString(js.Value.Schema), err)
	}

	input, err = p.buildInput(snap, llm.Settings{})
	if err != nil || input.OutputConfig != nil {
		t.Fatalf("output config without a format = %+v err = %v", input.OutputConfig, err)
	}
	_, err = p.buildInput(snap, llm.Settings{ResponseFormat: &llm.ResponseFormat{Type: llm.ResponseFormatJSONObject}})
	if !errors.Is(err, llm.ErrResponseFormatUnsupported) {
		t.Fatalf("schema-less JSON mode must be rejected, got %v", err)
	}
}
