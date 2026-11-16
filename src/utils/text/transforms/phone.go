package transforms

import (
	"regexp"
	"sort"
	"strings"
	"unicode"
)

func ExpandPhoneNumbers(loc *Locale) Transform {
	l := orDefault(loc)
	words := map[string]bool{}
	var symbols []string
	for _, cl := range []*Locale{l, EnUS} {
		for _, c := range cl.Currencies {
			words[strings.ToLower(c.Singular)] = true
			words[strings.ToLower(c.Plural)] = true
			for _, sym := range c.SuffixSymbols {
				words[strings.ToLower(sym)] = true
			}
			symbols = append(symbols, c.Symbols...)
		}
		for w := range cl.ScaleWords {
			words[w] = true
		}
	}
	moneyAround := func(s string, start, end int) bool {
		before := strings.TrimRightFunc(s[:start], unicode.IsSpace)
		for _, sym := range symbols {
			if len(before) >= len(sym) && strings.EqualFold(before[len(before)-len(sym):], sym) {
				return true
			}
		}
		word, _ := letterRun(s, len(s)-len(strings.TrimLeftFunc(s[end:], unicode.IsSpace)))
		return words[nuktaForms.Replace(strings.ToLower(word))]
	}
	return Func(func(text string) string {
		for _, p := range l.PhonePatterns {
			text = rewrite(p.Pattern, text, func(s string, m []int) (string, int, bool) {
				if isDigit(runeBefore(s, m[0])) || isDigit(runeAt(s, m[1])) || moneyAround(s, m[0], m[1]) {
					return "", 0, false
				}
				digits := onlyDigits(s[m[0]:m[1]])
				if len(digits) < p.MinDigits || (p.MaxDigits > 0 && len(digits) > p.MaxDigits) {
					return "", 0, false
				}
				return l.spellDigits(digits), m[1], true
			})
		}
		return text
	})
}

const codeToken = `[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*(?: \d{2,}(?:[-/][A-Za-z0-9]+)*)*`

func SpellCodes(loc *Locale) Transform {
	l := orDefault(loc)
	if len(l.CodeKeywords) == 0 {
		return Func(func(text string) string { return text })
	}
	keywords := append([]string(nil), l.CodeKeywords...)
	sort.SliceStable(keywords, func(i, j int) bool { return len(keywords[i]) > len(keywords[j]) })
	fillers := `\s*[:#=\-]+`
	if len(l.CodeFillers) > 0 {
		fillers = `\s*(?:` + strings.Join(l.CodeFillers, "|") + `)\b\.?|` + fillers
	}
	re := regexp.MustCompile(`(?i)(` + strings.Join(keywords, "|") + `)((?:` + fillers + `)*\s*)(` + codeToken + `)`)
	return Func(func(text string) string {
		return rewrite(re, text, func(s string, m []int) (string, int, bool) {
			if isWordRune(runeBefore(s, m[0])) {
				return "", 0, false
			}
			if m[5] == m[4] && isWordRune(runeBefore(s, m[3])) {
				return "", 0, false
			}
			start, end := m[6], m[7]
			for isWordRune(runeAt(s, end)) {
				cut := strings.LastIndexByte(s[start:end], ' ')
				if cut < 0 {
					return "", 0, false
				}
				end = start + cut
			}
			token := s[start:end]
			if !codeLike(token) {
				return "", 0, false
			}
			return s[m[0]:start] + l.spellCode(token), end, true
		})
	})
}

func SpellMatches(loc *Locale, patterns ...*regexp.Regexp) Transform {
	l := orDefault(loc)
	return Func(func(text string) string {
		for _, re := range patterns {
			text = rewrite(re, text, func(s string, m []int) (string, int, bool) {
				start, end := m[0], m[1]
				if len(m) > 3 && m[2] >= 0 {
					start, end = m[2], m[3]
				}
				return s[m[0]:start] + l.spellCode(s[start:end]) + s[end:m[1]], m[1], true
			})
		}
		return text
	})
}

func codeLike(token string) bool {
	alnum, digits := 0, 0
	for _, r := range token {
		if unicode.IsLetter(r) || isDigit(r) {
			alnum++
		}
		if isDigit(r) {
			digits++
		}
	}
	return alnum >= 3 && digits > 0
}
