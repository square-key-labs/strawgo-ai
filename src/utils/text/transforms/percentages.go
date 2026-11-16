package transforms

import (
	"regexp"
	"strings"
)

var percentRe = regexp.MustCompile(`(` + amountPattern + `)(?:\.(\d+))?\s*%`)

func ExpandPercentages(loc *Locale) Transform {
	l := orDefault(loc)
	return Func(func(text string) string {
		return rewrite(percentRe, text, func(s string, m []int) (string, int, bool) {
			n, ok := parseWhole(group(s, m, 1))
			if !ok {
				return "", 0, false
			}
			return l.decimal(n, strings.TrimRight(group(s, m, 2), "0")) + " " + l.Percent, m[1], true
		})
	})
}
