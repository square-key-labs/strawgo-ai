package transforms

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
)

func ExpandCurrency(loc *Locale) Transform {
	l := orDefault(loc)
	bySymbol := map[string]*Currency{}
	var prefixes, suffixes []string
	for i := range l.Currencies {
		c := &l.Currencies[i]
		for _, s := range c.Symbols {
			bySymbol[strings.ToLower(s)] = c
			prefixes = append(prefixes, s)
		}
		for _, s := range c.SuffixSymbols {
			bySymbol[strings.ToLower(s)] = c
			suffixes = append(suffixes, s)
		}
	}
	if len(prefixes) == 0 {
		return Func(func(text string) string { return text })
	}
	prefixRe := regexp.MustCompile(`(?i)(` + alternation(prefixes) + `)\s*(` + amountPattern + `)(?:\.(\d+))?(?:(?:\s*[-–]\s*|\s+to\s+)(` + amountPattern + `)(?:\.(\d+))?)?`)
	var suffixRe *regexp.Regexp
	if len(suffixes) > 0 {
		suffixRe = regexp.MustCompile(`(?i)\b(` + amountPattern + `)(?:\.(\d+))?\s*(` + alternation(suffixes) + `)\b`)
	}
	return Func(func(text string) string {
		text = rewrite(prefixRe, text, func(s string, m []int) (string, int, bool) {
			sym := group(s, m, 1)
			c := bySymbol[strings.ToLower(sym)]
			before := runeBefore(s, m[0])
			if c == nil || (unicode.IsLetter(runeAt(s, m[2])) && isWordRune(before)) {
				return "", 0, false
			}
			end := m[1]
			if isDigit(runeAt(s, end)) || (runeAt(s, end) == '.' && isDigit(runeAt(s, end+1))) {
				return "", 0, false
			}
			whole, ok := parseWhole(group(s, m, 2))
			if !ok {
				return "", 0, false
			}
			frac := group(s, m, 3)
			upper, upperOK := parseWhole(group(s, m, 4))
			ranged := group(s, m, 4) != ""
			if ranged && !upperOK {
				return "", 0, false
			}
			upperFrac := group(s, m, 5)
			spoken := ""
			if scale, scaleEnd := l.scaleAfter(s, end); scale != "" {
				end = scaleEnd
				spoken = l.decimal(whole, strings.TrimRight(frac, "0"))
				if ranged {
					spoken += l.RangeJoin + l.decimal(upper, strings.TrimRight(upperFrac, "0"))
				}
				spoken += " " + scale + " " + c.Plural
			} else {
				for _, suffix := range l.AmountSuffixes {
					if strings.HasPrefix(s[end:], suffix) {
						end += len(suffix)
						break
					}
				}
				switch {
				case !ranged:
					spoken = l.money(c, whole, frac)
				case frac == "" && upperFrac == "":
					spoken = l.Cardinal(whole) + l.RangeJoin + l.money(c, upper, "")
				default:
					spoken = l.money(c, whole, frac) + l.RangeJoin + l.money(c, upper, upperFrac)
				}
			}
			if isWordRune(runeAt(s, end)) {
				return "", 0, false
			}
			if unicode.IsLetter(before) {
				spoken = " " + spoken
			}
			return spoken, end, true
		})
		if suffixRe == nil {
			return text
		}
		return rewrite(suffixRe, text, func(s string, m []int) (string, int, bool) {
			c := bySymbol[strings.ToLower(group(s, m, 3))]
			whole, ok := parseWhole(group(s, m, 1))
			if c == nil || !ok {
				return "", 0, false
			}
			return l.money(c, whole, group(s, m, 2)), m[1], true
		})
	})
}

func (l *Locale) money(c *Currency, whole uint64, frac string) string {
	spoken := l.Cardinal(whole) + " " + plural(whole, c.Singular, c.Plural)
	if frac != "" && c.MinorSingular != "" {
		if len(frac) > 2 {
			frac = frac[:2]
		}
		minor, _ := strconv.ParseUint((frac + "0")[:2], 10, 64)
		if minor > 0 {
			spoken += l.MinorJoin + l.Cardinal(minor) + " " + plural(minor, c.MinorSingular, c.MinorPlural)
		}
	}
	return spoken
}

func (l *Locale) scaleAfter(s string, end int) (string, int) {
	if word, wordEnd := letterRun(s, end); word != "" {
		if scale, ok := l.ScaleAbbreviations[strings.ToLower(word)]; ok {
			return scale, wordEnd
		}
		return "", 0
	}
	i := end
	for i < len(s) && unicode.IsSpace(runeAt(s, i)) {
		i += len(string(runeAt(s, i)))
	}
	if i == end {
		return "", 0
	}
	word, wordEnd := letterRun(s, i)
	if word == "" {
		return "", 0
	}
	key := nuktaForms.Replace(strings.ToLower(word))
	if scale, ok := l.ScaleWords[key]; ok {
		return scale, wordEnd
	}
	if scale, ok := l.ScaleAbbreviations[key]; ok && len(key) > 1 {
		return scale, wordEnd
	}
	return "", 0
}

func letterRun(s string, i int) (string, int) {
	j := i
	for j < len(s) {
		r := runeAt(s, j)
		if !unicode.IsLetter(r) && !unicode.IsMark(r) {
			break
		}
		j += len(string(r))
	}
	return s[i:j], j
}

func plural(n uint64, singular, pluralForm string) string {
	if n == 1 {
		return singular
	}
	return pluralForm
}
