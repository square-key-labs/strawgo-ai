package transforms

import (
	"regexp"
	"strconv"
	"strings"
)

var timeRe = regexp.MustCompile(`\b(\d{1,2})(?::(\d{2}))?(?:\s?([AaPp])\.?\s?([Mm])\.?)?`)

func NormalizeTimes(loc *Locale) Transform {
	l := orDefault(loc)
	if l.TimeLayout == "" {
		return Func(func(text string) string { return text })
	}
	return Func(func(text string) string {
		return rewrite(timeRe, text, func(s string, m []int) (string, int, bool) {
			minutes := group(s, m, 2)
			meridiem := strings.ToUpper(group(s, m, 3))
			if minutes == "" && meridiem == "" {
				return "", 0, false
			}
			end := m[1]
			if meridiem != "" && s[end-1] != '.' && isWordRune(runeAt(s, end)) {
				return "", 0, false
			}
			before, after := runeBefore(s, m[0]), runeAt(s, end)
			if before == ':' || before == '.' || before == '/' || (meridiem == "" && (after == ':' || isWordRune(after))) {
				return "", 0, false
			}
			h, _ := strconv.Atoi(group(s, m, 1))
			mins, _ := strconv.Atoi(minutes)
			if mins > 59 || h > 23 || (meridiem != "" && (h < 1 || h > 12)) {
				return "", 0, false
			}
			hour := l.Cardinal(uint64(h))
			spoken := strings.ReplaceAll(l.TimeOnHour, "{h}", hour)
			if meridiem != "" && mins == 0 {
				spoken = hour
			}
			if mins > 0 {
				minuteText := l.Cardinal(uint64(mins))
				if mins < 10 {
					minuteText = l.TimeMinuteOh + minuteText
				}
				spoken = strings.NewReplacer("{h}", hour, "{m}", minuteText).Replace(l.TimeLayout)
			}
			switch meridiem {
			case "A":
				spoken += " " + l.AM
			case "P":
				spoken += " " + l.PM
			}
			if meridiem != "" && s[end-1] == '.' && (end == len(s) || runeAt(s, end) == ' ' && end+1 < len(s) && isUpperStart(s[end+1:])) {
				end--
			}
			return spoken, end, true
		})
	})
}

func isUpperStart(s string) bool {
	r := runeAt(s, 0)
	return r >= 'A' && r <= 'Z'
}
