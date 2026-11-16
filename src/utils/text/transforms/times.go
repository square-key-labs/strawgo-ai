package transforms

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"
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
			hourText := group(s, m, 1)
			if meridiem == "" && !(len(hourText) == 2 && hourText[0] == '0') && !l.timeCue(s, m[0], end) {
				return "", 0, false
			}
			h, _ := strconv.Atoi(hourText)
			mins, _ := strconv.Atoi(minutes)
			if mins > 59 || h > 23 || (meridiem != "" && (h < 1 || h > 12)) {
				return "", 0, false
			}
			if meridiem == "" && (h == 0 || h > 12) {
				meridiem = "P"
				if h == 0 {
					meridiem = "A"
				}
				if h != 12 {
					h = (h+11)%12 + 1
				}
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
			rest := strings.TrimLeft(s[end:], " ")
			for _, w := range l.TimeWords {
				if strings.HasPrefix(strings.ToLower(rest), w) && !isWordRune(runeAt(rest, len(w))) {
					end = len(s) - len(rest) + len(w)
					if mins == 0 {
						spoken = hour + " " + w
					}
					break
				}
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

func (l *Locale) timeCue(s string, start, end int) bool {
	before := strings.TrimRightFunc(s[:start], unicode.IsSpace)
	i := len(before)
	for i > 0 && isWordRune(runeBefore(before, i)) {
		i -= len(string(runeBefore(before, i)))
	}
	prev := strings.ToLower(before[i:])
	rest := strings.ToLower(strings.TrimLeft(s[end:], " "))
	for _, cue := range l.TimeCues {
		if prev == cue {
			return true
		}
	}
	for _, w := range l.TimeWords {
		if strings.HasPrefix(rest, w) {
			return true
		}
	}
	return false
}
