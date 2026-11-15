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
