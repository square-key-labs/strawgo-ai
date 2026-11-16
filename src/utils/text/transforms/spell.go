package transforms

import (
	"strconv"
	"strings"
	"unicode"
)

var nuktaForms = strings.NewReplacer(
	"\u0958", "\u0915\u093c", "\u0959", "\u0916\u093c", "\u095a", "\u0917\u093c", "\u095b", "\u091c\u093c", "\u095c", "\u0921\u093c", "\u095d", "\u0922\u093c", "\u095e", "\u092b\u093c", "\u095f", "\u092f\u093c",
)

func (l *Locale) Cardinal(n uint64) string {
	if n < 100 {
		return l.Below100[n]
	}
	type part struct {
		text  string
		units bool
		value uint64
	}
	var parts []part
	if l.IndianGrouping {
		if c := n / 10000000; c > 0 {
			parts = append(parts, part{text: l.Cardinal(c) + " " + l.Scales[2]})
		}
		if v := n / 100000 % 100; v > 0 {
			parts = append(parts, part{text: l.Below100[v] + " " + l.Scales[1]})
		}
		if v := n / 1000 % 100; v > 0 {
			parts = append(parts, part{text: l.Below100[v] + " " + l.Scales[0]})
		}
		if v := n % 1000; v > 0 {
			parts = append(parts, part{text: l.below1000(v), units: true, value: v})
		}
	} else {
		var groups []uint64
		for v := n; v > 0; v /= 1000 {
			groups = append(groups, v%1000)
		}
		for i := len(groups) - 1; i >= 0; i-- {
			g := groups[i]
			if g == 0 {
				continue
			}
			if i == 0 {
				parts = append(parts, part{text: l.below1000(g), units: true, value: g})
				continue
			}
			parts = append(parts, part{text: l.below1000(g) + " " + l.Scales[i-1]})
		}
	}
	var b strings.Builder
	for i, p := range parts {
		if i > 0 {
			if p.units && p.value < 100 {
				b.WriteString(l.FinalJoin)
			} else {
				b.WriteString(l.GroupJoin)
			}
		}
		b.WriteString(p.text)
	}
	return b.String()
}

func (l *Locale) below1000(n uint64) string {
	if n < 100 {
		return l.Below100[n]
	}
	s := l.Below100[n/100] + " " + l.Hundred
	if r := n % 100; r > 0 {
		s += l.HundredJoin + l.Below100[r]
	}
	return s
}

func (l *Locale) digitWords(digits string) string {
	words := make([]string, 0, len(digits))
	for _, r := range digits {
		if isDigit(r) {
			words = append(words, l.Below100[r-'0'])
		}
	}
	return strings.Join(words, " ")
}

func (l *Locale) spellDigits(digits string) string {
	if l.DigitWords {
		return l.digitWords(digits)
	}
	out := make([]string, 0, len(digits))
	for _, r := range digits {
		if isDigit(r) {
			out = append(out, string(r))
		}
	}
	return strings.Join(out, " ")
}

func (l *Locale) spellCode(code string) string {
	var out []string
	for _, r := range code {
		switch {
		case isDigit(r):
			if l.DigitWords {
				out = append(out, l.Below100[r-'0'])
			} else {
				out = append(out, string(r))
			}
		case unicode.IsLetter(r):
			out = append(out, string(unicode.ToUpper(r)))
		}
	}
	return strings.Join(out, " ")
}

func (l *Locale) decimal(whole uint64, frac string) string {
	s := l.Cardinal(whole)
	if frac != "" {
		s += " " + l.Point + " " + l.digitWords(frac)
	}
	return s
}

func (l *Locale) englishYear(y int) string {
	high, low := y/100, y%100
	if high == 0 || (high%10 == 0 && low < 10) || high >= 100 {
		return l.Cardinal(uint64(y))
	}
	h := l.Cardinal(uint64(high))
	switch {
	case low == 0:
		return h + " " + l.Hundred
	case low < 10:
		return h + " oh-" + l.Below100[low]
	}
	return h + " " + l.Below100[low]
}

func parseWhole(s string) (uint64, bool) {
	n, err := strconv.ParseUint(strings.ReplaceAll(s, ",", ""), 10, 64)
	return n, err == nil
}
