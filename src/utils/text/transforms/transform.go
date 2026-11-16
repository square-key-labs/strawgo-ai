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

type carryPrefix struct {
	text   string
	spoken string
}

type Chain struct {
	transforms []Transform
	carry      []carryPrefix
	protect    *regexp.Regexp

	mu         sync.Mutex
	held       string
	heldSpoken string
}

func NewChain(transforms ...Transform) *Chain {
	return &Chain{transforms: transforms}
}

func (c *Chain) Fork() *Chain {
	return &Chain{transforms: c.transforms, carry: c.carry, protect: c.protect}
}

func (c *Chain) ForkFilter() interface {
	Filter(text string) string
	Reset()
} {
	return c.Fork()
}

func (c *Chain) Transform(text string) string {
	if c.protect == nil {
		return c.apply(text)
	}
	locs := c.protect.FindAllStringIndex(text, -1)
	if len(locs) == 0 {
		return c.apply(text)
	}
	var b strings.Builder
	pos := 0
	for _, l := range locs {
		end := l[0] + len(strings.TrimRight(text[l[0]:l[1]], ".,;:!?)'\""))
		b.WriteString(c.apply(text[pos:l[0]]))
		b.WriteString(text[l[0]:end])
		pos = end
	}
	b.WriteString(c.apply(text[pos:]))
	return b.String()
}

func (c *Chain) apply(text string) string {
	for _, t := range c.transforms {
		text = t.Transform(text)
	}
	return text
}

func (c *Chain) Filter(text string) string {
	c.mu.Lock()
	if c.held != "" {
		if startsWithAmount(text) {
			text = c.held + text
		} else {
			text = c.heldSpoken + " " + strings.TrimLeftFunc(text, unicode.IsSpace)
		}
	}
	var p *carryPrefix
	text, p = splitCarry(text, c.carry)
	c.held, c.heldSpoken = "", ""
	if p != nil {
		c.held, c.heldSpoken = p.text+" ", p.spoken
	}
	c.mu.Unlock()
	return c.Transform(text)
}

func (c *Chain) Reset() {
	c.mu.Lock()
	c.held, c.heldSpoken = "", ""
	c.mu.Unlock()
}

func startsWithAmount(text string) bool {
	return isDigit(runeAt(strings.TrimLeftFunc(text, unicode.IsSpace), 0))
}

func splitCarry(text string, prefixes []carryPrefix) (string, *carryPrefix) {
	trimmed := strings.TrimRightFunc(text, unicode.IsSpace)
	for i := range prefixes {
		p := &prefixes[i]
		start := len(trimmed) - len(p.text)
		if start < 0 || !strings.EqualFold(trimmed[start:], p.text) || isWordRune(runeBefore(trimmed, start)) {
			continue
		}
		before := strings.TrimRightFunc(trimmed[:start], unicode.IsSpace)
		if isDigit(runeBefore(before, len(before))) {
			continue
		}
		return trimmed[:start], p
	}
	return text, nil
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
