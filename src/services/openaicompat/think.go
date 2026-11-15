package openaicompat

import (
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type thoughtFilter struct {
	open      string
	close     string
	pending   string
	inThought bool
	trimLead  bool
}

func newThoughtFilter(open, close string) *thoughtFilter {
	return &thoughtFilter{open: open, close: close, trimLead: true}
}

func (f *thoughtFilter) write(s string, emit func(llm.Event)) {
	if f.pending != "" {
		s = f.pending + s
		f.pending = ""
	}
	for s != "" {
		tag := f.open
		if f.inThought {
			tag = f.close
		}
		i := strings.Index(s, tag)
		if i < 0 {
			keep := partialSuffix(s, tag)
			f.emit(s[:len(s)-keep], emit)
			f.pending = s[len(s)-keep:]
			return
		}
		f.emit(s[:i], emit)
		s = s[i+len(tag):]
		f.inThought = !f.inThought
		f.trimLead = !f.inThought
	}
}

func (f *thoughtFilter) flush(emit func(llm.Event)) {
	s := f.pending
	f.pending = ""
	f.emit(s, emit)
}

func (f *thoughtFilter) emit(s string, emit func(llm.Event)) {
	if f.inThought {
		if s != "" {
			emit(llm.ThoughtDelta{Text: s})
		}
		return
	}
	if f.trimLead {
		s = strings.TrimLeft(s, " \t\r\n")
		if s == "" {
			return
		}
		f.trimLead = false
	}
	if s != "" {
		emit(llm.TextDelta{Text: s})
	}
}

func partialSuffix(s, tag string) int {
	for n := min(len(tag)-1, len(s)); n > 0; n-- {
		if strings.HasSuffix(s, tag[:n]) {
			return n
		}
	}
	return 0
}
