package transforms

import (
	"regexp"
	"strings"
)

const DefaultDigitCutoff = 2025

const amountPattern = `\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+`

var numberRe = regexp.MustCompile(`\b(` + amountPattern + `)(?:\.(\d+))?\b`)

func ExpandNumbers(loc *Locale, digitCutoff int) Transform {
	l := orDefault(loc)
	return Func(func(text string) string {
		return rewrite(numberRe, text, func(s string, m []int) (string, int, bool) {
			whole := group(s, m, 1)
			frac := group(s, m, 2)
			if frac == "" && len(whole) == 1 && l.inSpelledDigits(s, m[0], m[1]) {
				return "", 0, false
			}
			n, ok := parseWhole(whole)
			if !ok || (digitCutoff > 0 && n > uint64(digitCutoff)) {
				out := l.spellDigits(whole)
				if frac != "" {
					out += " " + l.Point + " " + l.spellDigits(frac)
				}
				return out, m[1], true
			}
			return l.decimal(n, frac), m[1], true
		})
	})
}

func (l *Locale) inSpelledDigits(s string, start, end int) bool {
	lone := func(i int) bool {
		return i >= 0 && isDigit(runeAt(s, i)) && !isWordRune(runeBefore(s, i)) && !isWordRune(runeAt(s, i+1))
	}
	if start >= 2 && s[start-1] == ' ' && lone(start-2) {
		return true
	}
	if end+1 < len(s) && s[end] == ' ' && lone(end+1) {
		return true
	}
	point := " " + l.Point + " "
	return strings.HasSuffix(s[:start], point) && lone(start-len(point)-1)
}
