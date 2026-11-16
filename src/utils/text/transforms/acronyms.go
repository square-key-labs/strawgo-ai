package transforms

import (
	"regexp"
	"strings"
)

var acronymRe = regexp.MustCompile(`\b[A-Z]{2,}\b`)

var NormalizeAcronyms Transform = Func(func(text string) string {
	return acronymRe.ReplaceAllStringFunc(text, func(a string) string {
		return strings.Join(strings.Split(a, ""), " ")
	})
})
