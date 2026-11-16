package transforms

import (
	"regexp"
	"sort"
	"strings"
)

var ambiguousUnits = []string{"in", "m", "g", "l", "L"}

func ExpandUnits(loc *Locale) Transform {
	l := orDefault(loc)
	if len(l.Units) == 0 {
		return Func(func(text string) string { return text })
	}
	var plain []string
	for u := range l.Units {
		if !isAmbiguousUnit(u) {
			plain = append(plain, u)
		}
	}
	plainRe := regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(` + alternation(plain) + `)\b`)
	ambiguousRe := regexp.MustCompile(`(\d+(?:\.\d+)?)(` + alternation(ambiguousUnits) + `)\b`)
	replace := func(s string, m []int) (string, int, bool) {
		u, ok := l.Units[strings.ToLower(group(s, m, 2))]
		if !ok {
			return "", 0, false
		}
		number := group(s, m, 1)
		if number == "1" {
			return number + " " + u.Singular, m[1], true
		}
		return number + " " + u.Plural, m[1], true
	}
	return Func(func(text string) string {
		return rewrite(ambiguousRe, rewrite(plainRe, text, replace), replace)
	})
}

func isAmbiguousUnit(u string) bool {
	for _, a := range ambiguousUnits {
		if a == u {
			return true
		}
	}
	return false
}

func alternation(words []string) string {
	sorted := append([]string(nil), words...)
	sort.SliceStable(sorted, func(i, j int) bool {
		if len(sorted[i]) != len(sorted[j]) {
			return len(sorted[i]) > len(sorted[j])
		}
		return sorted[i] < sorted[j]
	})
	quoted := make([]string, len(sorted))
	for i, w := range sorted {
		quoted[i] = regexp.QuoteMeta(w)
	}
	return strings.Join(quoted, "|")
}
