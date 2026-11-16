package transforms

import "regexp"

var urlRe = regexp.MustCompile(`(?i)\b(?:https?://|www\.)[^\s<>"]+`)

type VoiceFormatterOptions struct {
	Locale             *Locale
	EmailToSpeech      bool
	ExpandPhoneNumbers bool
	CodePatterns       []*regexp.Regexp
	SpellCodes         bool
	NormalizeDates     bool
	NormalizeTimes     bool
	ExpandCurrency     bool
	ExpandPercentages  bool
	ExpandUnits        bool
	NormalizeAcronyms  bool
	ExpandNumbers      bool
	NumberDigitCutoff  int
	CustomReplacements []Replacement
}

func DefaultVoiceFormatterOptions(loc *Locale) VoiceFormatterOptions {
	return VoiceFormatterOptions{
		Locale:             loc,
		EmailToSpeech:      true,
		ExpandPhoneNumbers: true,
		SpellCodes:         true,
		NormalizeDates:     true,
		NormalizeTimes:     true,
		ExpandCurrency:     true,
		ExpandPercentages:  true,
		ExpandUnits:        true,
		NormalizeAcronyms:  true,
	}
}

func NewVoiceFormatter(o VoiceFormatterOptions) (*Chain, error) {
	l := orDefault(o.Locale)
	var ts []Transform
	add := func(on bool, t Transform) {
		if on {
			ts = append(ts, t)
		}
	}
	add(o.EmailToSpeech, EmailToSpeech)
	add(o.ExpandPhoneNumbers, ExpandPhoneNumbers(l))
	add(len(o.CodePatterns) > 0, SpellMatches(l, o.CodePatterns...))
	add(o.SpellCodes, SpellCodes(l))
	add(o.NormalizeDates, NormalizeDates(l))
	add(o.NormalizeTimes, NormalizeTimes(l))
	add(o.ExpandCurrency, ExpandCurrency(l))
	add(o.ExpandPercentages, ExpandPercentages(l))
	add(o.ExpandUnits, ExpandUnits(l))
	add(o.NormalizeAcronyms, NormalizeAcronyms)
	add(o.ExpandNumbers, ExpandNumbers(l, o.NumberDigitCutoff))
	if len(o.CustomReplacements) > 0 {
		r, err := ReplaceText(o.CustomReplacements...)
		if err != nil {
			return nil, err
		}
		ts = append(ts, r)
	}
	c := NewChain(ts...)
	c.protect = urlRe
	if o.ExpandCurrency {
		for _, p := range l.SplitPrefixes {
			c.carry = append(c.carry, carryPrefix{text: p.Text, spoken: p.Spoken})
		}
	}
	return c, nil
}
