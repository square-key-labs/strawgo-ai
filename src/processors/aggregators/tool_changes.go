package aggregators

import (
	"fmt"
	"slices"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

const (
	ToolActivationMessageTemplate   = "The following function(s) have just been added and may now be called: %s. Any previously available functions remain available."
	ToolDeactivationMessageTemplate = "The following function(s) have just been removed and should not be called: %s. Any previously available functions remain available. The removed function(s) may become available again later, in which case you will be informed."
)

func (a *LLMContextAggregator) SetAddToolChangeMessages(enabled bool) {
	a.toolChanges = enabled
}

func (a *LLMContextAggregator) announceToolChanges(next llm.Tools) {
	if !a.toolChanges {
		return
	}
	if message, ok := ToolChangeMessage(a.context.Tools(), next); ok {
		a.context.Add(llm.Developer(message))
	}
}

func ToolChangeMessage(prev, next llm.Tools) (string, bool) {
	names := func(t llm.Tools) []string {
		out := make([]string, 0, len(t.Functions))
		for _, f := range t.Functions {
			out = append(out, f.Name)
		}
		return out
	}
	before, after := names(prev), names(next)
	diff := func(from, minus []string) string {
		var out []string
		for _, n := range from {
			if !slices.Contains(minus, n) && !slices.Contains(out, "`"+n+"`") {
				out = append(out, "`"+n+"`")
			}
		}
		slices.Sort(out)
		return strings.Join(out, ", ")
	}
	var parts []string
	if added := diff(after, before); added != "" {
		parts = append(parts, fmt.Sprintf(ToolActivationMessageTemplate, added))
	}
	if removed := diff(before, after); removed != "" {
		parts = append(parts, fmt.Sprintf(ToolDeactivationMessageTemplate, removed))
	}
	return strings.Join(parts, " "), len(parts) > 0
}
