package text

import (
	"bytes"
	"errors"
	"fmt"
	"strings"
)

type MatchAction int

const (
	MatchRemove MatchAction = iota
	MatchKeep
	MatchAggregate
)

func (a MatchAction) String() string {
	switch a {
	case MatchRemove:
		return "remove"
	case MatchKeep:
		return "keep"
	case MatchAggregate:
		return "aggregate"
	}
	return "unknown"
}

type Pattern struct {
	Type   string
	Start  string
	End    string
	Action MatchAction
}

type Segment struct {
	Text   string
	Match  bool
	Type   string
	Action MatchAction
	Full   string
}

const DefaultMaxPatternLength = 4096

type delimiters struct {
	start []byte
	end   []byte
}

type PatternPairAggregator struct {
	patterns []Pattern
	delims   []delimiters
	buf      []byte
	open     int
	scanFrom int
	maxLen   int
}

func NewPatternPairAggregator(patterns ...Pattern) (*PatternPairAggregator, error) {
	a := &PatternPairAggregator{open: -1, maxLen: DefaultMaxPatternLength}
	for _, p := range patterns {
		if err := a.AddPattern(p); err != nil {
			return nil, err
		}
	}
	return a, nil
}

func (a *PatternPairAggregator) AddPattern(p Pattern) error {
	if p.Type == "" {
		return errors.New("pattern type is required")
	}
	if p.Start == "" || p.End == "" {
		return fmt.Errorf("pattern %q needs both a start and an end delimiter", p.Type)
	}
	for _, existing := range a.patterns {
		if existing.Type == p.Type {
			return fmt.Errorf("pattern %q already registered", p.Type)
		}
	}
	a.patterns = append(a.patterns, p)
	a.delims = append(a.delims, delimiters{start: []byte(p.Start), end: []byte(p.End)})
	return nil
}

func (a *PatternPairAggregator) SetMaxPatternLength(n int) {
	a.maxLen = n
}

func (a *PatternPairAggregator) Buffered() bool {
	return len(a.buf) > 0 || a.open >= 0
}

func (a *PatternPairAggregator) Write(s string, emit func(Segment)) {
	if s == "" {
		return
	}
	a.buf = append(a.buf, s...)
	a.drain(emit)
}

func (a *PatternPairAggregator) Flush(emit func(Segment)) {
	if a.open >= 0 {
		p := a.patterns[a.open]
		if p.Action == MatchKeep {
			emitText(emit, p.Start+string(a.buf))
		}
	} else if n := len(a.buf) - a.trailingPartial(true); n > 0 {
		emitText(emit, string(a.buf[:n]))
	}
	a.Reset()
}

func (a *PatternPairAggregator) Reset() {
	a.buf = a.buf[:0]
	a.open = -1
	a.scanFrom = 0
}

func (a *PatternPairAggregator) drain(emit func(Segment)) {
	for {
		if a.open < 0 {
			idx, pi := a.earliestStart()
			if idx >= 0 {
				if idx > 0 {
					emitText(emit, string(a.buf[:idx]))
				}
				a.consume(idx + len(a.patterns[pi].Start))
				a.open = pi
				a.scanFrom = 0
				continue
			}
			if n := len(a.buf) - a.trailingPartial(false); n > 0 {
				emitText(emit, string(a.buf[:n]))
				a.consume(n)
			}
			return
		}

		p := a.patterns[a.open]
		j := bytes.Index(a.buf[a.scanFrom:], a.delims[a.open].end)
		if j < 0 {
			if a.maxLen > 0 && len(a.buf) > a.maxLen {
				if p.Action == MatchKeep {
					emitText(emit, p.Start+string(a.buf))
				}
				a.Reset()
				return
			}
			a.scanFrom = max(0, len(a.buf)-len(p.End)+1)
			return
		}
		j += a.scanFrom
		raw := string(a.buf[:j])
		full := p.Start + raw + p.End
		a.consume(j + len(p.End))
		a.open = -1
		a.scanFrom = 0
		emit(Segment{Text: strings.TrimSpace(raw), Match: true, Type: p.Type, Action: p.Action, Full: full})
		if p.Action == MatchKeep {
			emitText(emit, full)
		}
	}
}

func (a *PatternPairAggregator) earliestStart() (int, int) {
	best, bestPattern := -1, -1
	for i, d := range a.delims {
		idx := bytes.Index(a.buf, d.start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best || (idx == best && len(d.start) > len(a.delims[bestPattern].start)) {
			best, bestPattern = idx, i
		}
	}
	return best, bestPattern
}

func (a *PatternPairAggregator) trailingPartial(skipKeep bool) int {
	longest := 0
	for i, p := range a.patterns {
		if skipKeep && p.Action == MatchKeep {
			continue
		}
		start := a.delims[i].start
		for l := min(len(start)-1, len(a.buf)); l > longest; l-- {
			if bytes.HasSuffix(a.buf, start[:l]) {
				longest = l
				break
			}
		}
	}
	return longest
}

func (a *PatternPairAggregator) consume(n int) {
	rest := copy(a.buf, a.buf[n:])
	a.buf = a.buf[:rest]
}

func emitText(emit func(Segment), s string) {
	if s != "" {
		emit(Segment{Text: s, Full: s})
	}
}
