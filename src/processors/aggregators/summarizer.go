package aggregators

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

const defaultSummarizationTimeout = 120 * time.Second

type LLMContextSummaryConfig struct {
	TargetContextTokens     int
	MinMessagesAfterSummary int
	SummarizationPrompt     string
	SummaryMessageTemplate  string
}

type LLMAutoContextSummarizationConfig struct {
	MaxContextTokens        int
	MaxUnsummarizedMessages int
	SummaryConfig           LLMContextSummaryConfig
}

type contextSummaryLLM interface {
	SummarizeContext(ctx context.Context, prompt string, llmCtx *services.LLMContext) (string, error)
}

type LLMContextSummarizer struct {
	config     LLMAutoContextSummarizationConfig
	summaryLLM services.LLMService
	timeout    time.Duration
	log        *logger.Logger

	OnSummaryApplied func(beforeCount, afterCount int)

	summarizeWithLLM func(ctx context.Context, llm services.LLMService, prompt string, llmCtx *services.LLMContext) (string, error)
}

func NewLLMContextSummarizer(config LLMAutoContextSummarizationConfig, summaryLLM services.LLMService) *LLMContextSummarizer {
	s := &LLMContextSummarizer{
		config:     config,
		summaryLLM: summaryLLM,
		timeout:    defaultSummarizationTimeout,
		log:        logger.WithPrefix("Summarizer"),
	}
	s.summarizeWithLLM = s.defaultSummarizeWithLLM
	return s
}

func (s *LLMContextSummarizer) SetTimeout(timeout time.Duration) {
	if timeout <= 0 {
		s.timeout = defaultSummarizationTimeout
		return
	}
	s.timeout = timeout
}

func (s *LLMContextSummarizer) ShouldAutoSummarize(llmCtx *services.LLMContext) bool {
	if llmCtx == nil {
		return false
	}

	if s.config.MaxContextTokens > 0 && estimateContextTokens(llmCtx.Messages) > s.config.MaxContextTokens {
		return true
	}

	if s.config.MaxUnsummarizedMessages > 0 && len(llmCtx.Messages) > s.config.MaxUnsummarizedMessages {
		return true
	}

	return false
}

func (s *LLMContextSummarizer) SummarizeContext(ctx context.Context, llmCtx *services.LLMContext, mainLLM services.LLMService) bool {
	if llmCtx == nil || len(llmCtx.Messages) == 0 {
		return false
	}

	preserveStart := detectIncompleteFunctionTailStart(llmCtx.Messages)
	candidateMessages := cloneMessages(llmCtx.Messages[:preserveStart])
	preservedTail := cloneMessages(llmCtx.Messages[preserveStart:])
	if len(candidateMessages) == 0 {
		return false
	}

	recentKeep := s.computeRecentKeep(candidateMessages)
	if recentKeep > len(candidateMessages) {
		recentKeep = len(candidateMessages)
	}

	olderCount := len(candidateMessages) - recentKeep
	if olderCount <= 0 {
		return false
	}

	olderMessages := cloneMessages(candidateMessages[:olderCount])
	recentMessages := cloneMessages(candidateMessages[olderCount:])

	prompt := s.buildPrompt(olderMessages)
	summaryCtx := &services.LLMContext{Messages: olderMessages}
	summary, err := s.runSummarization(ctx, prompt, summaryCtx, mainLLM)
	if err != nil && errors.Is(err, context.DeadlineExceeded) {
		s.log.Warn("summary timeout reached after %s", s.timeout)
	} else if err != nil {
		s.log.Warn("summarization failed: %v", err)
	}
	if err != nil || strings.TrimSpace(summary) == "" {
		return false
	}

	summaryMessage := services.LLMMessage{
		Role:    "user",
		Content: s.applySummaryTemplate(summary),
	}

	beforeCount := len(llmCtx.Messages)
	newMessages := make([]services.LLMMessage, 0, 1+len(recentMessages)+len(preservedTail))
	newMessages = append(newMessages, summaryMessage)
	newMessages = append(newMessages, recentMessages...)
	newMessages = append(newMessages, preservedTail...)
	llmCtx.Messages = newMessages

	if s.OnSummaryApplied != nil {
		s.OnSummaryApplied(beforeCount, len(newMessages))
	}

	return true
}

func (s *LLMContextSummarizer) computeRecentKeep(messages []services.LLMMessage) int {
	minKeep := s.config.SummaryConfig.MinMessagesAfterSummary
	if minKeep < 0 {
		minKeep = 0
	}

	if s.config.MaxContextTokens <= 0 {
		return minKeep
	}

	targetTokens := int(float64(s.config.MaxContextTokens) * 0.8)
	if s.config.SummaryConfig.TargetContextTokens > 0 {
		targetTokens = s.config.SummaryConfig.TargetContextTokens
	}
	if targetTokens < 1 {
		targetTokens = 1
	}

	keep := 0
	keptTokens := 0
	for i := len(messages) - 1; i >= 0; i-- {
		tokens := estimateMessageTokens(messages[i])
		if keep >= minKeep && keptTokens+tokens > targetTokens {
			break
		}
		keptTokens += tokens
		keep++
	}

	if keep < minKeep {
		keep = minKeep
	}

	if keep >= len(messages) {
		keep = len(messages) - 1
		if keep < 0 {
			keep = 0
		}
	}

	return keep
}

func (s *LLMContextSummarizer) buildPrompt(messages []services.LLMMessage) string {
	prompt := s.config.SummaryConfig.SummarizationPrompt
	if strings.TrimSpace(prompt) == "" {
		prompt = "Summarize the following conversation history while preserving key facts, commitments, and unresolved questions."
	}

	var b strings.Builder
	b.WriteString(prompt)
	b.WriteString("\n\nConversation:\n")
	for _, msg := range messages {
		b.WriteString(msg.Role)
		b.WriteString(": ")
		b.WriteString(messageContentForSummary(msg))
		b.WriteString("\n")
	}

	return b.String()
}

func (s *LLMContextSummarizer) applySummaryTemplate(summary string) string {
	template := s.config.SummaryConfig.SummaryMessageTemplate
	if strings.TrimSpace(template) == "" {
		return summary
	}
	if strings.Contains(template, "%s") {
		return fmt.Sprintf(template, summary)
	}
	return template + "\n" + summary
}

func (s *LLMContextSummarizer) runSummarization(ctx context.Context, prompt string, summaryCtx *services.LLMContext, mainLLM services.LLMService) (string, error) {
	llm := s.summaryLLM
	if llm == nil {
		llm = mainLLM
	}
	if llm == nil {
		return heuristicSummary(summaryCtx.Messages), nil
	}

	timeout := s.timeout
	if timeout <= 0 {
		timeout = defaultSummarizationTimeout
	}

	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	type result struct {
		summary string
		err     error
	}

	resCh := make(chan result, 1)
	go func() {
		summary, err := s.summarizeWithLLM(runCtx, llm, prompt, summaryCtx)
		resCh <- result{summary: summary, err: err}
	}()

	select {
	case <-runCtx.Done():
		return "", runCtx.Err()
	case res := <-resCh:
		return res.summary, res.err
	}
}

func (s *LLMContextSummarizer) defaultSummarizeWithLLM(ctx context.Context, llm services.LLMService, prompt string, llmCtx *services.LLMContext) (string, error) {
	if summaryService, ok := llm.(contextSummaryLLM); ok {
		return summaryService.SummarizeContext(ctx, prompt, llmCtx)
	}
	return heuristicSummary(llmCtx.Messages), nil
}

func estimateContextTokens(messages []services.LLMMessage) int {
	tokens := 0
	for _, msg := range messages {
		tokens += estimateMessageTokens(msg)
	}
	return tokens
}

func estimateMessageTokens(msg services.LLMMessage) int {
	content := messageContentForSummary(msg)
	return len(content)/4 + 10
}

func messageContentForSummary(msg services.LLMMessage) string {
	content := msg.Content
	if len(msg.ToolCalls) > 0 {
		parts := make([]string, 0, len(msg.ToolCalls))
		for _, call := range msg.ToolCalls {
			parts = append(parts, call.Function.Name+":"+call.Function.Arguments)
		}
		if content == "" {
			content = strings.Join(parts, ";")
		} else {
			content = content + " " + strings.Join(parts, ";")
		}
	}
	if msg.ToolCallID != "" {
		if content == "" {
			content = msg.ToolCallID
		} else {
			content = content + " " + msg.ToolCallID
		}
	}
	return content
}

func heuristicSummary(messages []services.LLMMessage) string {
	if len(messages) == 0 {
		return ""
	}

	parts := make([]string, 0, 3)
	for i := len(messages) - 1; i >= 0 && len(parts) < 3; i-- {
		content := strings.TrimSpace(messageContentForSummary(messages[i]))
		if content == "" {
			continue
		}
		parts = append([]string{messages[i].Role + ": " + content}, parts...)
	}

	if len(parts) == 0 {
		return "Conversation history summarized."
	}

	return "Conversation summary: " + strings.Join(parts, " | ")
}

func detectIncompleteFunctionTailStart(messages []services.LLMMessage) int {
	if len(messages) == 0 {
		return 0
	}

	unresolved := make(map[string]struct{})
	for _, msg := range messages {
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				if tc.ID != "" {
					unresolved[tc.ID] = struct{}{}
				}
			}
		}
		if msg.Role == "tool" && msg.ToolCallID != "" && !isIncompleteToolResponse(msg) {
			delete(unresolved, msg.ToolCallID)
		}
	}

	if len(unresolved) == 0 {
		return len(messages)
	}

	start := len(messages)
	for i := 0; i < len(messages); i++ {
		if isMessageUsingUnresolvedID(messages[i], unresolved) {
			start = i
			break
		}
	}

	return start
}

func isMessageUsingUnresolvedID(msg services.LLMMessage, unresolved map[string]struct{}) bool {
	if msg.Role == "tool" && msg.ToolCallID != "" {
		_, ok := unresolved[msg.ToolCallID]
		return ok
	}
	if len(msg.ToolCalls) > 0 {
		for _, tc := range msg.ToolCalls {
			if _, ok := unresolved[tc.ID]; ok {
				return true
			}
		}
	}
	return false
}

func isIncompleteToolResponse(msg services.LLMMessage) bool {
	if msg.Role != "tool" {
		return false
	}
	content := strings.TrimSpace(msg.Content)
	return content == "" || content == "IN_PROGRESS"
}

func cloneMessages(messages []services.LLMMessage) []services.LLMMessage {
	if len(messages) == 0 {
		return nil
	}
	cloned := make([]services.LLMMessage, len(messages))
	copy(cloned, messages)
	return cloned
}
