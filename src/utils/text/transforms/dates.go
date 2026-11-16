package transforms

import (
	"regexp"
	"strconv"
	"strings"
	"time"
)

var isoDateRe = regexp.MustCompile(`\b(\d{4})-(\d{2})-(\d{2})\b`)

var ordinalSuffixes = map[int]string{1: "st", 2: "nd", 3: "rd"}

func NormalizeDates(loc *Locale) Transform {
	l := orDefault(loc)
	var class strings.Builder
	for _, r := range l.DateSeparators {
		class.WriteString(`\` + string(r))
	}
	sep := "[" + class.String() + "]"
	localRe := regexp.MustCompile(`\b(\d{1,2})` + sep + `(\d{1,2})` + sep + `(\d{4})\b`)
	day, month := 2, 1
	if l.DateOrder == DayMonthYear {
		day, month = 1, 2
	}
	return Func(func(text string) string {
		text = rewrite(isoDateRe, text, func(s string, m []int) (string, int, bool) {
			return l.spokenDate(group(s, m, 1), group(s, m, 2), group(s, m, 3), m[1])
		})
		return rewrite(localRe, text, func(s string, m []int) (string, int, bool) {
			return l.spokenDate(group(s, m, 3), group(s, m, month), group(s, m, day), m[1])
		})
	})
}

func (l *Locale) spokenDate(ys, ms, ds string, end int) (string, int, bool) {
	y, _ := strconv.Atoi(ys)
	mo, _ := strconv.Atoi(ms)
	d, _ := strconv.Atoi(ds)
	t := time.Date(y, time.Month(mo), d, 0, 0, 0, 0, time.UTC)
	if y < 1 || t.Year() != y || int(t.Month()) != mo || t.Day() != d {
		return "", 0, false
	}
	dayText := l.Cardinal(uint64(d))
	if l.OrdinalDays {
		suffix := "th"
		if d%100 < 11 || d%100 > 13 {
			if s, ok := ordinalSuffixes[d%10]; ok {
				suffix = s
			}
		}
		dayText = strconv.Itoa(d) + suffix
	}
	yearText := l.Cardinal(uint64(y))
	if l.EnglishYears {
		yearText = l.englishYear(y)
	}
	return strings.NewReplacer("{month}", l.Months[mo-1], "{day}", dayText, "{year}", yearText).Replace(l.DateLayout), end, true
}
