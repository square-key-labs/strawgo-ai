package filters

import (
	"context"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

const (
	DefaultWakeKeepalive  = 3 * time.Second
	maxWakeAccumulatorLen = 1024
)

type WakeCheckOption func(*WakeCheckFilter)

func WithWakeKeepalive(d time.Duration) WakeCheckOption {
	return func(f *WakeCheckFilter) { f.keepalive = d }
}

func WithWakeClock(now func() time.Time) WakeCheckOption {
	return func(f *WakeCheckFilter) { f.now = now }
}

type WakeCheckFilter struct {
	*processors.BaseProcessor
	patterns    []*regexp.Regexp
	keepalive   time.Duration
	now         func() time.Time
	awake       bool
	lastWake    time.Time
	accumulator string
}

func NewWakeCheckFilter(phrases []string, opts ...WakeCheckOption) *WakeCheckFilter {
	f := &WakeCheckFilter{keepalive: DefaultWakeKeepalive, now: time.Now}
	for _, phrase := range phrases {
		if re := CompilePhrasePattern(phrase); re != nil {
			f.patterns = append(f.patterns, re)
		}
	}
	for _, opt := range opts {
		opt(f)
	}
	f.BaseProcessor = processors.NewBaseProcessor("WakeCheckFilter", f)
	return f
}

func CompilePhrasePattern(phrase string) *regexp.Regexp {
	words := strings.Fields(phrase)
	if len(words) == 0 {
		return nil
	}
	quoted := make([]string, len(words))
	for i, w := range words {
		quoted[i] = regexp.QuoteMeta(w)
	}
	expr := strings.Join(quoted, `\s*`)
	first, _ := utf8.DecodeRuneInString(words[0])
	last, _ := utf8.DecodeLastRuneInString(words[len(words)-1])
	if !unspacedScript(first) {
		expr = `\b` + expr
	}
	if !unspacedScript(last) {
		expr += `\b`
	}
	return regexp.MustCompile(`(?i)` + expr)
}

func unspacedScript(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai, unicode.Lao, unicode.Khmer, unicode.Myanmar)
}

func (f *WakeCheckFilter) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	t, ok := frame.(*frames.TranscriptionFrame)
	if !ok {
		return f.PushFrame(frame, direction)
	}
	if len(f.patterns) == 0 {
		return nil
	}

	now := f.now()
	if f.awake {
		if now.Sub(f.lastWake) < f.keepalive {
			f.lastWake = now
			return f.PushFrame(frame, direction)
		}
		f.awake = false
	}

	f.accumulator += t.Text
	if over := len(f.accumulator) - maxWakeAccumulatorLen; over > 0 {
		for over < len(f.accumulator) && !utf8.RuneStart(f.accumulator[over]) {
			over++
		}
		f.accumulator = f.accumulator[over:]
	}
	for _, re := range f.patterns {
		loc := re.FindStringIndex(f.accumulator)
		if loc == nil {
			continue
		}
		f.awake = true
		f.lastWake = now
		t.Text = f.accumulator[loc[0]:]
		f.accumulator = ""
		return f.PushFrame(t, direction)
	}
	return nil
}
