package transforms

import (
	"regexp"
	"strings"
)

const DefaultDigitCutoff = 2025

const amountPattern = `\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3}|\d+`

var numberRe = regexp.MustCompile(`(-?)\b(` + amountPattern + `)(?:\.(\d+))?\b`)

func ExpandNumbers(loc *Locale, digitCutoff int) Transform {
	l := orDefault(loc)
	return Func(func(text string) string {
		return rewrite(numberRe, text, func(s string, m []int) (string, int, bool) {
			whole := group(s, m, 2)
			frac := group(s, m, 3)
			start := m[4]
			if structured(s, start, m[1]) {
				return "", 0, false
			}
			if frac == "" && len(whole) == 1 && l.inSpelledDigits(s, start, m[1]) {
				return "", 0, false
			}
			sign := l.sign(s, m[0], start)
			n, ok := parseWhole(whole)
			if !ok || (digitCutoff > 0 && n > uint64(digitCutoff)) || (len(whole) > 1 && whole[0] == '0') {
				out := l.spellDigits(whole)
				if frac != "" {
					out += " " + l.Point + " " + l.spellDigits(frac)
				}
				return sign + out, m[1], true
			}
			return sign + l.decimal(n, frac), m[1], true
		})
	})
}

func (l *Locale) sign(s string, start, digits int) string {
	if start == digits {
		return ""
	}
	if r := runeBefore(s, start); isWordRune(r) || r == '.' || r == ',' {
		return "-"
	}
	return l.Minus + " "
}

func structured(s string, start, end int) bool {
	before, after := runeBefore(s, start), runeAt(s, end)
	switch {
	case before == ':' || before == '/' || after == ':' || after == '/':
		return true
	case before == '.' && isDigit(runeBefore(s, start-1)):
		return true
	case after == '.' && isDigit(runeAt(s, end+1)):
		return true
	}
	return false
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
