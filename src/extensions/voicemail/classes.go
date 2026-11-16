package voicemail

import (
	"strings"
	"unicode"
)

type Class int

const (
	ClassUnknown Class = iota
	ClassHuman
	ClassVoicemail
	ClassIVR
	ClassAnnouncement
)

func (c Class) String() string {
	switch c {
	case ClassHuman:
		return "human"
	case ClassVoicemail:
		return "voicemail"
	case ClassIVR:
		return "ivr"
	case ClassAnnouncement:
		return "announcement"
	}
	return "unknown"
}

func (c Class) Verdict() Verdict {
	switch c {
	case ClassVoicemail, ClassAnnouncement:
		return Voicemail
	}
	return Conversation
}

func (c Class) machine() bool {
	return c == ClassVoicemail || c == ClassIVR || c == ClassAnnouncement
}

func classOf(v Verdict) Class {
	switch v {
	case Conversation:
		return ClassHuman
	case Voicemail:
		return ClassVoicemail
	}
	return ClassUnknown
}

type Source int

const (
	SourceNone Source = iota
	SourceClassifier
	SourceTiming
	SourceBeep
	SourceCarrier
	SourceTimeout
)

func (s Source) String() string {
	switch s {
	case SourceClassifier:
		return "classifier"
	case SourceTiming:
		return "timing"
	case SourceBeep:
		return "beep"
	case SourceCarrier:
		return "carrier"
	case SourceTimeout:
		return "timeout"
	}
	return "none"
}

type Result struct {
	Class  Class
	Source Source
}

const DetailedPrompt = `You are an answering detection classifier for an outbound phone call. A bot placed the call, and you receive the transcription of what was heard after the call connected. The transcription may be in English or Hindi, in Devanagari or in Latin letters. Decide what answered.

CONVERSATION: a person answered. A greeting such as "hello?", "hi", "yeah?", "haan ji", "हाँ जी", "boliye", "बोलिए" or "John speaking"; a question to the caller such as "who is this?", "kaun bol raha hai?" or "कौन बोल रहा है?"; spontaneous speech that expects a reply.

VOICEMAIL: a personal or business answering machine that will record a message. "You've reached", "leave a message after the tone", "I'm not available right now", "call me back", "mailbox is full", "our office is currently closed".

IVR: an automated menu that expects key presses or spoken choices. "Press 1 for", "for billing press 2", "please say or enter", "your call is important to us, please hold".

ANNOUNCEMENT: a recorded message from the phone network saying the call cannot be completed. English examples: "the number you are calling is switched off", "the number you have dialled is not reachable", "the subscriber cannot be reached at the moment", "is out of coverage area", "the number you are calling is busy", "all lines are busy", "the number you have dialled does not exist", "please check the number you have dialled", "is temporarily out of service", "all circuits are busy". Hindi examples: "आप जिस नंबर पर कॉल कर रहे हैं वह अभी बंद है" ("aap jis number par call kar rahe hain woh abhi band hai"), "आप जिस नंबर पर कॉल कर रहे हैं वह अभी पहुँच से बाहर है" ("abhi pahunch se bahar hai"), "कवरेज क्षेत्र से बाहर है" ("coverage kshetra se bahar hai"), "आप जिस नंबर पर कॉल कर रहे हैं वह अभी व्यस्त है" ("abhi vyast hai"), "यह नंबर मौजूद नहीं है" ("yeh number maujood nahin hai"), "कृपया नंबर की जाँच करें" ("kripya number ki jaanch karein"), "कृपया थोड़ी देर बाद कॉल करें" ("kripya thodi der baad call karein"), "सभी लाइनें व्यस्त हैं" ("sabhi lines vyast hain").

UNKNOWN: too little was heard to tell.

Respond with exactly one word: CONVERSATION, VOICEMAIL, IVR, ANNOUNCEMENT or UNKNOWN.`

func ParseClass(reply string) Class {
	for _, word := range strings.FieldsFunc(strings.ToUpper(reply), func(r rune) bool { return !unicode.IsLetter(r) }) {
		switch word {
		case "CONVERSATION", "HUMAN":
			return ClassHuman
		case "VOICEMAIL":
			return ClassVoicemail
		case "IVR":
			return ClassIVR
		case "ANNOUNCEMENT":
			return ClassAnnouncement
		case "UNKNOWN":
			return ClassUnknown
		}
	}
	return ClassUnknown
}

func ClassFromAnsweredBy(answeredBy string) Class {
	switch v := strings.ToLower(strings.TrimSpace(answeredBy)); {
	case v == "human":
		return ClassHuman
	case v == "machine" || strings.HasPrefix(v, "machine_"):
		return ClassVoicemail
	}
	return ClassUnknown
}
