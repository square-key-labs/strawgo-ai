package user_start

import (
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/square-key-labs/strawgo-ai/src/turns/internal/frameinfo"
)

const (
	DefaultBackchannelMaxWords          = 4
	DefaultBackchannelRepeatWindow      = time.Second
	DefaultBackchannelMaxSpeechDuration = 2 * time.Second
)

type BackchannelDetector interface {
	TakeBackchannel() (text string, ok bool)
}

type BackchannelParams struct {
	Words             []string
	MaxWords          int
	RepeatWindow      time.Duration
	MaxSpeechDuration time.Duration
}

type BackchannelUserTurnStartStrategy struct {
	phrases             [][]string
	maxWords            int
	repeatWindow        time.Duration
	maxSpeech           time.Duration
	enableInterruptions bool

	mu               sync.Mutex
	botSpeaking      bool
	userSpeaking     bool
	candidate        bool
	speechStart      time.Time
	lastBackchannel  time.Time
	backchannel      string
	hasBackchannel   bool
	resetAggregation bool
}

func NewBackchannelUserTurnStartStrategy(params BackchannelParams, enableInterruptions bool) *BackchannelUserTurnStartStrategy {
	words := params.Words
	if words == nil {
		words = DefaultBackchannelWords()
	}
	if params.MaxWords <= 0 {
		params.MaxWords = DefaultBackchannelMaxWords
	}
	if params.RepeatWindow == 0 {
		params.RepeatWindow = DefaultBackchannelRepeatWindow
	}
	if params.MaxSpeechDuration == 0 {
		params.MaxSpeechDuration = DefaultBackchannelMaxSpeechDuration
	}
	s := &BackchannelUserTurnStartStrategy{
		maxWords:            params.MaxWords,
		repeatWindow:        params.RepeatWindow,
		maxSpeech:           params.MaxSpeechDuration,
		enableInterruptions: enableInterruptions,
	}
	for _, w := range words {
		if tokens := backchannelTokens(w); len(tokens) > 0 {
			s.phrases = append(s.phrases, tokens)
		}
	}
	return s
}

func (s *BackchannelUserTurnStartStrategy) ShouldStart(frame any) bool {
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.decide(frame, now) || s.spokeTooLong(now)
}

func (s *BackchannelUserTurnStartStrategy) decide(frame any, now time.Time) bool {
	name := frameinfo.Name(frame)
	switch {
	case name == "BotStartedSpeakingFrame":
		s.botSpeaking = true
		return false
	case name == "BotStoppedSpeakingFrame":
		s.botSpeaking = false
		pending := s.candidate && s.userSpeaking
		s.candidate = false
		return pending
	case frameinfo.IsUserStarted(name):
		s.userSpeaking = true
		s.speechStart = now
		if !s.botSpeaking || s.repeated(now) {
			return true
		}
		s.candidate = true
		return false
	case frameinfo.IsUserStopped(name):
		s.userSpeaking = false
		return false
	case !frameinfo.IsTranscription(name):
		return false
	}

	text, _ := frameinfo.Text(frame)
	tokens := backchannelTokens(text)
	final := frameinfo.IsFinal(frame)
	if len(tokens) == 0 {
		if final && s.botSpeaking && strings.TrimSpace(text) != "" {
			s.resetAggregation = true
		}
		return false
	}
	if !s.botSpeaking || len(tokens) > s.maxWords {
		return true
	}
	if !final {
		return !s.matches(tokens, true)
	}
	if !s.matches(tokens, false) || s.repeated(now) {
		return true
	}
	s.lastBackchannel = now
	s.candidate = false
	s.backchannel = strings.TrimSpace(text)
	s.hasBackchannel = true
	s.resetAggregation = true
	return false
}

func (s *BackchannelUserTurnStartStrategy) repeated(now time.Time) bool {
	return s.repeatWindow > 0 && !s.lastBackchannel.IsZero() && now.Sub(s.lastBackchannel) <= s.repeatWindow
}

func (s *BackchannelUserTurnStartStrategy) spokeTooLong(now time.Time) bool {
	return s.maxSpeech > 0 && s.candidate && s.userSpeaking && s.botSpeaking && now.Sub(s.speechStart) >= s.maxSpeech
}

func (s *BackchannelUserTurnStartStrategy) NextDeadline() (time.Time, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.maxSpeech <= 0 || !s.candidate || !s.userSpeaking || !s.botSpeaking {
		return time.Time{}, false
	}
	return s.speechStart.Add(s.maxSpeech), true
}

func (s *BackchannelUserTurnStartStrategy) TakeBackchannel() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	text, ok := s.backchannel, s.hasBackchannel
	s.backchannel, s.hasBackchannel = "", false
	return text, ok
}

func (s *BackchannelUserTurnStartStrategy) ConsumeAggregationReset() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.resetAggregation
	s.resetAggregation = false
	return r
}

func (s *BackchannelUserTurnStartStrategy) EnableInterruptions() bool {
	return s.enableInterruptions
}

func (s *BackchannelUserTurnStartStrategy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.candidate = false
	s.lastBackchannel = time.Time{}
	s.backchannel, s.hasBackchannel = "", false
	s.resetAggregation = false
}

func (s *BackchannelUserTurnStartStrategy) matches(tokens []string, partial bool) bool {
	reachable := make([]bool, len(tokens)+1)
	reachable[0] = true
	for i := 0; i < len(tokens); i++ {
		if !reachable[i] {
			continue
		}
		for _, p := range s.phrases {
			if i+len(p) <= len(tokens) && equalTokens(tokens[i:i+len(p)], p) {
				reachable[i+len(p)] = true
			}
			if partial && i+len(p) >= len(tokens) && prefixOfPhrase(tokens[i:], p) {
				return true
			}
		}
	}
	return reachable[len(tokens)]
}

func equalTokens(a, b []string) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func prefixOfPhrase(rest, phrase []string) bool {
	last := len(rest) - 1
	for i := 0; i < last; i++ {
		if rest[i] != phrase[i] {
			return false
		}
	}
	return strings.HasPrefix(phrase[last], rest[last])
}

func backchannelTokens(text string) []string {
	var b strings.Builder
	var prev rune
	for _, r := range strings.ToLower(text) {
		switch {
		case unicode.Is(unicode.Cf, r) || r == '\'' || r == '’':
			continue
		case unicode.IsLetter(r) || unicode.IsMark(r) || unicode.IsDigit(r):
			if r == prev {
				continue
			}
			b.WriteRune(r)
		default:
			b.WriteRune(' ')
			r = ' '
		}
		prev = r
	}
	return strings.Fields(b.String())
}
