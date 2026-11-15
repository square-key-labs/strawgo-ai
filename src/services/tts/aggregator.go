package tts

import (
	"bytes"
	"strings"
	"unicode"
	"unicode/utf8"
)

type AggregationMode int

const (
	AggregateSentences AggregationMode = iota
	AggregateTokens
)

var sentenceEnders = map[rune]bool{
	'.': true, '!': true, '?': true, ';': true, '…': true,
	'。': true, '？': true, '！': true, '；': true, '．': true, '｡': true,
	'।': true, '॥': true,
	'؟': true, '؛': true, '۔': true,
	'။': true, '។': true,
	'։': true, '።': true, '፧': true,
}

var abbreviations = []string{
	"Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.", "vs.", "etc.",
	"Inc.", "Ltd.", "Co.", "Prof.", "Gen.", "Col.", "St.", "Ave.", "Blvd.",
	"Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.",
	"Oct.", "Nov.", "Dec.", "U.S.", "U.K.", "E.U.",
	"a.m.", "p.m.", "e.g.", "i.e.",
}

type sentenceAggregator struct {
	mode AggregationMode
	buf  []byte
	scan int
}

func (a *sentenceAggregator) add(text string, emit func(string)) {
	if text == "" {
		return
	}
	if a.mode == AggregateTokens {
		emit(text)
		return
	}
	a.buf = append(a.buf, text...)
	start := 0
	i := a.scan
	for i < len(a.buf) {
		r, size := utf8.DecodeRune(a.buf[i:])
		end := i + size
		if sentenceEnders[r] && end == len(a.buf) {
			break
		}
		if sentenceEnders[r] && a.isBoundary(start, end, r) {
			if s := strings.TrimSpace(string(a.buf[start:end])); s != "" {
				emit(s)
			}
			start = end
		}
		i = end
	}
	if start > 0 {
		n := copy(a.buf, a.buf[start:])
		a.buf = a.buf[:n]
		i -= start
	}
	a.scan = i
}

func (a *sentenceAggregator) isBoundary(start, end int, r rune) bool {
	next, _ := utf8.DecodeRune(a.buf[end:])
	if r == '.' && unicode.IsDigit(next) {
		return false
	}
	if !unicode.IsSpace(next) && r < utf8.RuneSelf {
		return false
	}
	sentence := a.buf[start:end]
	for _, abbr := range abbreviations {
		if bytes.HasSuffix(sentence, []byte(abbr)) {
			return false
		}
	}
	return true
}

func (a *sentenceAggregator) flush() string {
	s := strings.TrimSpace(string(a.buf))
	a.reset()
	return s
}

func (a *sentenceAggregator) reset() {
	a.buf = a.buf[:0]
	a.scan = 0
}
