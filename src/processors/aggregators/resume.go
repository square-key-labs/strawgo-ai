package aggregators

import (
	"strings"
	"unicode"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const queuedContextsLimit = 32

type queuedChunk struct {
	text     string
	textType frames.TextType
}

type queuedContext struct {
	chunks    []queuedChunk
	untracked bool
	complete  bool
}

type interruptedSpeech struct {
	message   string
	base      string
	remainder string
}

func (a *LLMAssistantAggregator) queueText(f *frames.TTSQueuedTextFrame) {
	q := a.queued[f.ContextID]
	if f.Complete {
		if q != nil {
			q.complete = true
		}
		return
	}
	if q == nil {
		if a.queued == nil {
			a.queued = make(map[string]*queuedContext)
		}
		if len(a.queued) >= queuedContextsLimit {
			clear(a.queued)
		}
		q = &queuedContext{}
		a.queued[f.ContextID] = q
	}
	q.chunks = append(q.chunks, queuedChunk{text: f.Text, textType: f.TextType})
	q.untracked = q.untracked || !f.PlaybackTracked
}

func (a *LLMAssistantAggregator) takeQueued(contexts []string) ([]queuedChunk, bool) {
	var chunks []queuedChunk
	resumable := len(contexts) > 0
	for _, id := range contexts {
		q := a.queued[id]
		delete(a.queued, id)
		if q == nil || q.untracked || !q.complete {
			resumable = false
			continue
		}
		chunks = append(chunks, q.chunks...)
	}
	return chunks, resumable
}

func (a *LLMAssistantAggregator) addAssistantText(text string, resume *interruptedSpeech) (message, base string) {
	if resume == nil {
		a.context.Add(llm.Assistant(text))
		return text, ""
	}
	full := joinSpeech(resume.base, text)
	if resume.message == "" {
		a.context.Add(llm.Assistant(full))
		return full, resume.base
	}
	if !a.replaceAssistantText(resume.message, full) {
		a.context.Add(llm.Assistant(text))
		return text, ""
	}
	return full, resume.base
}

func (a *LLMAssistantAggregator) replaceAssistantText(old, text string) bool {
	found := false
	a.context.Transform(func(msgs []llm.Message) []llm.Message {
		for i := len(msgs) - 1; i >= 0; i-- {
			if msgs[i].Role == llm.RoleAssistant && len(msgs[i].Parts) == 1 && msgs[i].Text() == old {
				msgs[i] = llm.Assistant(text)
				found = true
				break
			}
		}
		return msgs
	})
	return found
}

func (a *LLMAssistantAggregator) rememberInterrupted(chunks []queuedChunk, spoken, message, base string) {
	if len(a.calls) > 0 || len(chunks) == 0 {
		return
	}
	prefix, remainder, ok := splitUnspoken(chunks, spoken)
	if !ok || remainder == "" {
		return
	}
	a.resumeMu.Lock()
	a.interrupted = &interruptedSpeech{message: message, base: joinSpeech(base, prefix), remainder: remainder}
	a.resumeMu.Unlock()
}

func (a *LLMAssistantAggregator) beginTurnResume(fromTTS bool) {
	a.resumeMu.Lock()
	defer a.resumeMu.Unlock()
	a.resume = nil
	if fromTTS {
		a.resume = a.pending
	}
	a.pending = nil
	a.interrupted = nil
}

func (a *LLMAssistantAggregator) restorePending() {
	a.resumeMu.Lock()
	defer a.resumeMu.Unlock()
	if a.pending != nil {
		a.interrupted = a.pending
		a.pending = nil
	}
}

func (a *LLMAssistantAggregator) takeInterruptedSpeech() (string, bool) {
	a.resumeMu.Lock()
	defer a.resumeMu.Unlock()
	rec := a.interrupted
	a.interrupted = nil
	if rec == nil {
		return "", false
	}
	a.pending = rec
	return rec.remainder, true
}

func (a *LLMAssistantAggregator) discardInterruptedSpeech() {
	a.resumeMu.Lock()
	a.interrupted = nil
	a.resumeMu.Unlock()
}

func joinSpeech(base, text string) string {
	base, text = strings.TrimSpace(base), strings.TrimSpace(text)
	if base == "" || text == "" {
		return base + text
	}
	return base + " " + text
}

func joinQueued(chunks []queuedChunk) (string, []int) {
	var b strings.Builder
	bounds := []int{0}
	for _, c := range chunks {
		if c.textType != frames.TextTypeToken {
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			bounds = append(bounds, b.Len())
		}
		b.WriteString(c.text)
	}
	text := b.String()
	from := 0
	for _, sentence := range tts.SplitSentences(text) {
		i := strings.Index(text[from:], sentence)
		if i < 0 {
			break
		}
		bounds = append(bounds, from+i)
		from += i + len(sentence)
	}
	return text, bounds
}

func splitUnspoken(chunks []queuedChunk, spoken string) (prefix, remainder string, ok bool) {
	text, bounds := joinQueued(chunks)
	cut := 0
	for _, w := range strings.Fields(spoken) {
		want := normalizeWord(w)
		if want == "" {
			continue
		}
		for {
			start, end := nextField(text, cut)
			if start < 0 {
				return "", "", false
			}
			cut = end
			got := normalizeWord(text[start:end])
			if got == "" {
				continue
			}
			if got != want {
				return "", "", false
			}
			break
		}
	}
	cut = len(text) - len(strings.TrimLeftFunc(text[cut:], unicode.IsSpace))
	start := 0
	for _, b := range bounds {
		if b <= cut && b > start {
			start = b
		}
	}
	return strings.TrimSpace(text[:start]), strings.TrimSpace(text[start:]), true
}

func nextField(text string, from int) (int, int) {
	rest := strings.TrimLeftFunc(text[from:], unicode.IsSpace)
	if rest == "" {
		return -1, -1
	}
	start := len(text) - len(rest)
	end := strings.IndexFunc(rest, unicode.IsSpace)
	if end < 0 {
		return start, len(text)
	}
	return start, start + end
}

func normalizeWord(w string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return unicode.ToLower(r)
		}
		return -1
	}, w)
}
