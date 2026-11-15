package llm

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func ptr[T any](v T) *T { return &v }

func TestSettingsMerge(t *testing.T) {
	base := Settings{
		Model:             "m1",
		SystemInstruction: "sys",
		Temperature:       ptr(0.5),
		MaxTokens:         ptr(100),
		Extra:             map[string]any{"a": 1, "b": 2},
	}
	cases := []struct {
		name  string
		delta Settings
		check func(t *testing.T, got Settings)
	}{
		{"empty delta keeps everything", Settings{}, func(t *testing.T, got Settings) {
			if !reflect.DeepEqual(got, base) {
				t.Fatalf("got %+v", got)
			}
		}},
		{"scalars override", Settings{Model: "m2", SystemInstruction: "s2", TopP: ptr(0.9), Seed: ptr(7)}, func(t *testing.T, got Settings) {
			if got.Model != "m2" || got.SystemInstruction != "s2" || *got.TopP != 0.9 || *got.Seed != 7 {
				t.Fatalf("got %+v", got)
			}
			if *got.Temperature != 0.5 || *got.MaxTokens != 100 {
				t.Fatalf("unset fields lost: %+v", got)
			}
		}},
		{"pointer zero values override", Settings{Temperature: ptr(0.0), MaxTokens: ptr(0)}, func(t *testing.T, got Settings) {
			if *got.Temperature != 0 || *got.MaxTokens != 0 {
				t.Fatalf("got %+v", got)
			}
		}},
		{"extra merges key by key", Settings{Extra: map[string]any{"b": 3, "c": 4}}, func(t *testing.T, got Settings) {
			want := map[string]any{"a": 1, "b": 3, "c": 4}
			if !reflect.DeepEqual(got.Extra, want) {
				t.Fatalf("extra = %v", got.Extra)
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := map[string]any{}
			for k, v := range base.Extra {
				before[k] = v
			}
			tc.check(t, base.Merge(tc.delta))
			if !reflect.DeepEqual(base.Extra, before) {
				t.Fatalf("Merge mutated receiver Extra: %v", base.Extra)
			}
		})
	}
}

func TestSettingsMergeExtraIntoNil(t *testing.T) {
	got := Settings{}.Merge(Settings{Extra: map[string]any{"k": "v"}})
	if got.Extra["k"] != "v" {
		t.Fatalf("extra = %v", got.Extra)
	}
}

func TestFunctionSchemaJSONSchema(t *testing.T) {
	cases := []struct {
		name   string
		schema FunctionSchema
		want   string
	}{
		{"empty", FunctionSchema{Name: "f"}, `{"properties":{},"required":[],"type":"object"}`},
		{"populated", FunctionSchema{
			Name:       "weather",
			Properties: map[string]any{"city": map[string]any{"type": "string"}},
			Required:   []string{"city"},
		}, `{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.schema.JSONSchema())
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != tc.want {
				t.Fatalf("got %s want %s", b, tc.want)
			}
		})
	}
}

func TestToolsEmptyAndDefaults(t *testing.T) {
	if !(Tools{}).Empty() {
		t.Fatal("zero Tools not empty")
	}
	if (Tools{Custom: map[string][]json.RawMessage{"x": nil}}).Empty() {
		t.Fatal("custom tools reported empty")
	}
	if DefaultFunctionOptions().CancelOnInterruption {
		t.Fatal("default must not cancel a running function on interruption")
	}
}

func TestSettingsMergeResponseFormat(t *testing.T) {
	format := JSONSchemaFormat("answer", map[string]any{"type": "object"})
	base := Settings{Model: "m", ResponseFormat: format}
	if got := base.Merge(Settings{Model: "n"}); got.ResponseFormat != format {
		t.Fatalf("empty delta dropped the response format: %+v", got)
	}
	text := &ResponseFormat{Type: ResponseFormatText}
	if got := base.Merge(Settings{ResponseFormat: text}); got.ResponseFormat != text || got.ResponseFormat.Structured() {
		t.Fatalf("delta did not replace the response format: %+v", got.ResponseFormat)
	}
	if !format.Structured() || !format.Strict || format.SchemaName() != "answer" || (&ResponseFormat{}).SchemaName() != "response" {
		t.Fatalf("format helpers: %+v", format)
	}
	var none *ResponseFormat
	if none.Structured() {
		t.Fatal("nil format must not be structured")
	}
}

func TestRefusalErrorIsMatchable(t *testing.T) {
	cause := errors.New("provider said no")
	var err error = fmt.Errorf("generation failed: %w", &RefusalError{
		Refusal: Refusal{Category: RefusalPromptBlocked, Provider: "P", Reason: "SAFETY", Detail: "HARM"},
		Err:     cause,
	})
	r, ok := AsRefusal(err)
	if !ok || r.Category != RefusalPromptBlocked || !errors.Is(err, cause) {
		t.Fatalf("refusal = %+v ok=%v", r, ok)
	}
	if want := "generation failed: P: response refused (prompt_blocked, SAFETY) [HARM]: provider said no"; err.Error() != want {
		t.Fatalf("error text = %q", err.Error())
	}
	if _, ok := AsRefusal(cause); ok {
		t.Fatal("plain error matched as refusal")
	}
}
