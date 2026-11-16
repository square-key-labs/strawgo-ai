package tts

import (
	"regexp"
	"strings"
)

type TextFilter interface {
	Filter(text string) string
	Reset()
}

type MarkdownFilter struct{}

var (
	mdLink    = regexp.MustCompile(`\[([^\]]*)\]\([^)]*\)`)
	mdHeading = regexp.MustCompile(`(?m)^\s{0,3}#{1,6}\s+`)
	mdBullet  = regexp.MustCompile(`(?m)^\s*[-*+]\s+`)
	mdEmph    = strings.NewReplacer("**", "", "__", "", "~~", "", "`", "", "*", "")
)

func (MarkdownFilter) Filter(text string) string {
	if !strings.ContainsAny(text, "*_`#[~-+") {
		return text
	}
	text = mdLink.ReplaceAllString(text, "$1")
	text = mdHeading.ReplaceAllString(text, "")
	text = mdBullet.ReplaceAllString(text, "")
	return mdEmph.Replace(text)
}

func (MarkdownFilter) Reset() {}

type forkableFilter interface {
	ForkFilter() interface {
		Filter(text string) string
		Reset()
	}
}

func (b *Base) AddFilters(filters ...TextFilter) {
	b.settingsMu.Lock()
	defer b.settingsMu.Unlock()
	next := make([]TextFilter, 0, len(b.opts.Filters)+len(filters))
	next = append(next, b.opts.Filters...)
	b.opts.Filters = append(next, forkFilters(filters)...)
}

func forkFilters(filters []TextFilter) []TextFilter {
	out := make([]TextFilter, len(filters))
	for i, f := range filters {
		if fk, ok := f.(forkableFilter); ok {
			out[i] = fk.ForkFilter()
		} else {
			out[i] = f
		}
	}
	return out
}

func (b *Base) filters() []TextFilter {
	b.settingsMu.RLock()
	defer b.settingsMu.RUnlock()
	return b.opts.Filters
}

func (b *Base) resetFilters() {
	for _, f := range b.filters() {
		f.Reset()
	}
}
