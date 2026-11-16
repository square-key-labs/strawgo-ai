package transforms

import (
	"html"
	"regexp"
	"sort"
	"strings"
)

type PronunciationFormatter func(word, ipa string) (string, bool)

func SSMLPhoneme(word, ipa string) (string, bool) {
	if ipa == "" {
		return "", false
	}
	return `<phoneme alphabet="ipa" ph="` + html.EscapeString(ipa) + `">` + word + `</phoneme>`, true
}

func Pronunciations(words map[string]string, format PronunciationFormatter) (Transform, []string) {
	phonemes := map[string]string{}
	var unusable []string
	for word, ipa := range words {
		word = strings.TrimSpace(word)
		if word == "" {
			continue
		}
		if _, ok := format(word, ipa); !ok {
			unusable = append(unusable, word)
			continue
		}
		phonemes[strings.ToLower(word)] = ipa
	}
	sort.Strings(unusable)
	if len(phonemes) == 0 {
		return Func(func(text string) string { return text }), unusable
	}
	keys := make([]string, 0, len(phonemes))
	for w := range phonemes {
		keys = append(keys, w)
	}
	sort.Slice(keys, func(i, j int) bool {
		if len(keys[i]) != len(keys[j]) {
			return len(keys[i]) > len(keys[j])
		}
		return keys[i] < keys[j]
	})
	re := regexp.MustCompile(`(?i)` + alternation(keys))
	standalone := func(s string, start, end int) bool {
		before, after := runeBefore(s, start), runeAt(s, end)
		return !isWordRune(before) && before != '-' && !isWordRune(after) && after != '-'
	}
	return Func(func(text string) string {
		return rewrite(re, text, func(s string, m []int) (string, int, bool) {
			for _, k := range keys {
				end := m[0] + len(k)
				if end > len(s) || !strings.EqualFold(s[m[0]:end], k) || !standalone(s, m[0], end) {
					continue
				}
				word := s[m[0]:end]
				if out, ok := format(word, phonemes[k]); ok {
					return out, end, true
				}
				return word, end, true
			}
			return "", 0, false
		})
	}), unusable
}
