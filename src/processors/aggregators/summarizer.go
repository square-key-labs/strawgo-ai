package aggregators

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

const (
	DefaultSummarizationTimeout = 120 * time.Second
	toolResultInProgress        = llm.ToolResultInProgress
	defaultSummarizationPrompt  = "Summarize the following conversation history while preserving key facts, decisions, commitments, user preferences, and unresolved questions. Reply with the summary only."
)

type SummaryFunc func(ctx context.Context, snapshot llm.Snapshot) (string, error)

type LLMContextSummaryConfig struct {
	TargetContextTokens     int
	MinMessagesAfterSummary int
	SummarizationPrompt     string
	SummaryMessageTemplate  string
	Keep                    func(llm.Message) bool
}

type LLMAutoContextSummarizationConfig struct {
	MaxContextTokens        int
	MaxUnsummarizedMessages int
	SummaryConfig           LLMContextSummaryConfig
	EstimateTokens          func(llm.Message) int
	ShouldSummarize         func(llm.Snapshot) bool
}

type SummaryResult struct {
	Summary string
	Err     error
	cut     int
	prefix  []llm.Message
	kept    []bool
}

type LLMContextSummarizer struct {
	config    LLMAutoContextSummarizationConfig
	summarize SummaryFunc

	OnSummaryApplied func(beforeCount, afterCount int)

	mu       sync.Mutex
	timeout  time.Duration
	inFlight bool
	cancel   context.CancelFunc
	wg       sync.WaitGroup
}

func NewLLMContextSummarizer(config LLMAutoContextSummarizationConfig, summarize SummaryFunc) *LLMContextSummarizer {
	return &LLMContextSummarizer{config: config, summarize: summarize, timeout: DefaultSummarizationTimeout}
}

func (s *LLMContextSummarizer) SetTimeout(timeout time.Duration) {
	if timeout <= 0 {
		timeout = DefaultSummarizationTimeout
	}
	s.mu.Lock()
	s.timeout = timeout
	s.mu.Unlock()
}

func (s *LLMContextSummarizer) InFlight() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inFlight
}

func (s *LLMContextSummarizer) ShouldAutoSummarize(snapshot llm.Snapshot) bool {
	maxTokens, maxMessages := s.config.MaxContextTokens, s.config.MaxUnsummarizedMessages
	bounded := maxTokens > 0 || maxMessages > 0
	if bounded {
		keep := s.config.SummaryConfig.Keep
		tokens, count := 0, 0
		for _, m := range snapshot.Messages {
			if keep != nil && keep(m) {
				continue
			}
			count++
			if maxTokens > 0 {
				tokens += s.estimateTokens(m)
			}
		}
		if !(maxTokens > 0 && tokens > maxTokens) && !(maxMessages > 0 && count > maxMessages) {
			return false
		}
	}
	if s.config.ShouldSummarize == nil {
		return bounded
	}
	return s.config.ShouldSummarize(snapshot)
}

func (s *LLMContextSummarizer) estimateTokens(m llm.Message) int {
	if s.config.EstimateTokens != nil {
		return s.config.EstimateTokens(m)
	}
	return estimateMessageTokens(m)
}

func (s *LLMContextSummarizer) Start(parent context.Context, snapshot llm.Snapshot, deliver func(SummaryResult)) bool {
	if s.summarize == nil {
		return false
	}
	plan, ok := s.plan(snapshot.Messages)
	if !ok {
		return false
	}

	s.mu.Lock()
	if s.inFlight {
		s.mu.Unlock()
		return false
	}
	runCtx, cancel := context.WithTimeout(parent, s.timeout)
	s.inFlight = true
	s.cancel = cancel
	s.wg.Add(1)
	s.mu.Unlock()

	go func() {
		defer s.wg.Done()
		defer cancel()
		res := s.run(runCtx, plan)
		s.mu.Lock()
		s.inFlight = false
		s.cancel = nil
		s.mu.Unlock()
		if deliver != nil {
			deliver(res)
		}
	}()
	return true
}

func (s *LLMContextSummarizer) Summarize(ctx context.Context, c *llm.Context) (bool, error) {
	if s.summarize == nil || c == nil {
		return false, nil
	}
	plan, ok := s.plan(c.Snapshot().Messages)
	if !ok {
		return false, nil
	}
	s.mu.Lock()
	timeout := s.timeout
	s.mu.Unlock()
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	res := s.run(runCtx, plan)
	if res.Err != nil {
		return false, res.Err
	}
	return s.Apply(c, res), nil
}

func (s *LLMContextSummarizer) Cancel() {
	s.mu.Lock()
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
	s.wg.Wait()
}

func (s *LLMContextSummarizer) Apply(c *llm.Context, res SummaryResult) bool {
	summary := strings.TrimSpace(res.Summary)
	if c == nil || res.Err != nil || summary == "" || res.cut == 0 || len(res.kept) != res.cut {
		return false
	}
	message := llm.User(s.applySummaryTemplate(summary))
	applied := false
	before, after := 0, 0
	c.Transform(func(msgs []llm.Message) []llm.Message {
		before = len(msgs)
		if len(msgs) < res.cut || !reflect.DeepEqual(msgs[:res.cut], res.prefix) {
			after = before
			return msgs
		}
		out := make([]llm.Message, 0, len(msgs))
		for i, m := range msgs[:res.cut] {
			switch {
			case res.kept[i]:
				out = append(out, m)
			case !applied:
				out = append(out, message)
				applied = true
			}
		}
		if !applied {
			after = before
			return msgs
		}
		out = append(out, msgs[res.cut:]...)
		after = len(out)
		return out
	})
	if applied && s.OnSummaryApplied != nil {
		s.OnSummaryApplied(before, after)
	}
	return applied
}

func (s *LLMContextSummarizer) run(ctx context.Context, plan SummaryResult) SummaryResult {
	prompt := s.config.SummaryConfig.SummarizationPrompt
	if strings.TrimSpace(prompt) == "" {
		prompt = defaultSummarizationPrompt
	}
	request := llm.Snapshot{Messages: []llm.Message{
		llm.System(prompt),
		llm.User(formatTranscript(plan.prefix, plan.kept)),
	}}
	summary, err := s.summarize(ctx, request)
	if err == nil && ctx.Err() != nil {
		err = ctx.Err()
	}
	if err == nil && strings.TrimSpace(summary) == "" {
		err = errors.New("empty summary")
	}
	plan.Summary = summary
	plan.Err = err
	return plan
}

func (s *LLMContextSummarizer) plan(messages []llm.Message) (SummaryResult, bool) {
	head := 0
	for head < len(messages) && (messages[head].Role == llm.RoleSystem || messages[head].Role == llm.RoleDeveloper) {
		head++
	}

	end := firstUnresolvedToolCall(messages)
	if end <= head {
		return SummaryResult{}, false
	}

	keep := s.computeRecentKeep(messages[head:end])
	cut := end - keep
	for cut > head && cut < len(messages) && messages[cut].Role == llm.RoleTool {
		cut--
	}
	for cut > head && len(messages[cut-1].ToolCalls()) > 0 {
		cut--
	}
	if cut <= head {
		return SummaryResult{}, false
	}

	kept := make([]bool, cut)
	for i := 0; i < head; i++ {
		kept[i] = true
	}
	if keep := s.config.SummaryConfig.Keep; keep != nil {
		for i := head; i < cut; i++ {
			kept[i] = keep(messages[i])
		}
		keepToolPairs(messages[:cut], kept)
	}
	summarizable := false
	for _, k := range kept[head:] {
		if !k {
			summarizable = true
			break
		}
	}
	if !summarizable {
		return SummaryResult{}, false
	}

	prefix := make([]llm.Message, cut)
	copy(prefix, messages[:cut])
	return SummaryResult{cut: cut, prefix: prefix, kept: kept}, true
}

func keepToolPairs(messages []llm.Message, kept []bool) {
	ids := make(map[string]struct{})
	for changed := true; changed; {
		changed = false
		for i, m := range messages {
			if !kept[i] {
				continue
			}
			for _, call := range m.ToolCalls() {
				ids[call.ID] = struct{}{}
			}
			if r, ok := m.ToolResult(); ok {
				ids[r.ToolCallID] = struct{}{}
			}
		}
		if len(ids) == 0 {
			return
		}
		for i, m := range messages {
			if kept[i] {
				continue
			}
			paired := false
			for _, call := range m.ToolCalls() {
				if _, ok := ids[call.ID]; ok {
					paired = true
				}
			}
			if r, ok := m.ToolResult(); ok {
				if _, ok := ids[r.ToolCallID]; ok {
					paired = true
				}
			}
			if paired {
				kept[i] = true
				changed = true
			}
		}
	}
}

func (s *LLMContextSummarizer) computeRecentKeep(messages []llm.Message) int {
	minKeep := s.config.SummaryConfig.MinMessagesAfterSummary
	if minKeep < 0 {
		minKeep = 0
	}

	keep := minKeep
	if s.config.MaxContextTokens > 0 {
		target := s.config.SummaryConfig.TargetContextTokens
		if target <= 0 {
			target = s.config.MaxContextTokens * 8 / 10
		}
		if target < 1 {
			target = 1
		}
		keep = 0
		kept := 0
		for i := len(messages) - 1; i >= 0; i-- {
			tokens := s.estimateTokens(messages[i])
			if keep >= minKeep && kept+tokens > target {
				break
			}
			kept += tokens
			keep++
		}
	}

	if keep >= len(messages) {
		keep = len(messages) - 1
	}
	if keep < 0 {
		keep = 0
	}
	return keep
}

func (s *LLMContextSummarizer) applySummaryTemplate(summary string) string {
	template := s.config.SummaryConfig.SummaryMessageTemplate
	if strings.TrimSpace(template) == "" {
		return "Conversation summary: " + summary
	}
	if strings.Contains(template, "%s") {
		return fmt.Sprintf(template, summary)
	}
	return template + "\n" + summary
}

func firstUnresolvedToolCall(messages []llm.Message) int {
	resolved := make(map[string]bool)
	for _, m := range messages {
		if r, ok := m.ToolResult(); ok {
			resolved[r.ToolCallID] = strings.TrimSpace(r.Content) != "" && r.Content != toolResultInProgress
		}
	}
	for i, m := range messages {
		for _, call := range m.ToolCalls() {
			if !resolved[call.ID] {
				return i
			}
		}
	}
	return len(messages)
}

func formatTranscript(messages []llm.Message, kept []bool) string {
	var b strings.Builder
	for i, m := range messages {
		if kept[i] {
			continue
		}
		b.WriteString(string(m.Role))
		b.WriteString(": ")
		b.WriteString(messageSummaryText(m))
		b.WriteByte('\n')
	}
	return b.String()
}

func messageSummaryText(m llm.Message) string {
	var b strings.Builder
	for _, p := range m.Parts {
		if b.Len() > 0 {
			b.WriteByte(' ')
		}
		switch v := p.(type) {
		case llm.TextPart:
			b.WriteString(v.Text)
		case llm.ToolCallPart:
			b.WriteString("[call ")
			b.WriteString(v.Name)
			b.WriteByte(' ')
			b.Write(v.Arguments)
			b.WriteByte(']')
		case llm.ToolResultPart:
			b.WriteString("[result ")
			b.WriteString(v.Name)
			b.WriteByte(' ')
			b.WriteString(v.Content)
			b.WriteByte(']')
		case llm.ImagePart:
			b.WriteString("[image]")
		case llm.AudioPart:
			b.WriteString("[audio]")
		case llm.FilePart:
			b.WriteString("[file]")
		}
	}
	return b.String()
}

func estimateMessageTokens(m llm.Message) int {
	n := 0
	for _, p := range m.Parts {
		switch v := p.(type) {
		case llm.TextPart:
			n += len(v.Text)
		case llm.ToolCallPart:
			n += len(v.Name) + len(v.Arguments)
		case llm.ToolResultPart:
			n += len(v.Content)
		case llm.ThoughtPart:
			n += len(v.Text)
		}
	}
	return n/4 + 10
}
