package transforms

import (
	"regexp"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

type Transform interface {
	Transform(text string) string
}

type Func func(text string) string

func (f Func) Transform(text string) string { return f(text) }

type Chain struct {
	transforms []Transform
	carry      []string

	mu   sync.Mutex
	held string
}

func NewChain(transforms ...Transform) *Chain {
	return &Chain{transforms: transforms}
}

func (c *Chain) Transform(text string) string {
	for _, t := range c.transforms {
		text = t.Transform(text)
	}
	return text
}

func (c *Chain) Filter(text string) string {
	c.mu.Lock()
	text, c.held = splitCarry(c.held+text, c.carry)
	c.mu.Unlock()
	return c.Transform(text)
}

func (c *Chain) Reset() {
	c.mu.Lock()
	c.held = ""
	c.mu.Unlock()
}

func splitCarry(text string, prefixes []string) (string, string) {
	trimmed := strings.TrimRightFunc(text, unicode.IsSpace)
	for _, p := range prefixes {
		start := len(trimmed) - len(p)
		if start < 0 || !strings.EqualFold(trimmed[start:], p) || isWordRune(runeBefore(trimmed, start)) {
			continue
		}
		before := strings.TrimRightFunc(trimmed[:start], unicode.IsSpace)
		if isDigit(runeBefore(before, len(before))) {
			continue
		}
		return trimmed[:start], trimmed[start:] + " "
	}
	return text, ""
}

type matchFunc func(s string, m []int) (replacement string, end int, ok bool)

func rewrite(re *regexp.Regexp, s string, fn matchFunc) string {
	matches := re.FindAllStringSubmatchIndex(s, -1)
	if len(matches) == 0 {
		return s
	}
	var b strings.Builder
	pos := 0
	changed := false
	for _, m := range matches {
		if m[0] < pos {
			continue
		}
		repl, end, ok := fn(s, m)
		if !ok {
			continue
		}
		b.WriteString(s[pos:m[0]])
		b.WriteString(repl)
		pos = end
		changed = true
	}
	if !changed {
		return s
	}
	b.WriteString(s[pos:])
	return b.String()
}

func group(s string, m []int, i int) string {
	if 2*i+1 >= len(m) || m[2*i] < 0 {
		return ""
	}
	return s[m[2*i]:m[2*i+1]]
}

func runeBefore(s string, i int) rune {
	if i <= 0 {
		return -1
	}
	r, _ := utf8.DecodeLastRuneInString(s[:i])
	return r
}

func runeAt(s string, i int) rune {
	if i >= len(s) {
		return -1
	}
	r, _ := utf8.DecodeRuneInString(s[i:])
	return r
}

func isDigit(r rune) bool { return r >= '0' && r <= '9' }

func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || unicode.IsMark(r)
}

func onlyDigits(s string) string {
	var b strings.Builder
	for _, r := range s {
		if isDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}
