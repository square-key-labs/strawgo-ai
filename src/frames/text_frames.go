package frames

import "time"

type TextType string

const (
	TextTypeBackchannel     TextType = "backchannel"
	TextTypeExpressionSound TextType = "expression_sound"
	TextTypeExpressionPause TextType = "expression_pause"
	TextTypeSentence        TextType = "sentence"
	TextTypeToken           TextType = "token"
	TextTypeWord            TextType = "word"
)

type TranscriptionSpeaker struct {
	Label  string
	Number int
}

type PerceptionLabel struct {
	Label      string
	Confidence float64
}

type PerceptionSignal struct {
	Name   string
	Labels []PerceptionLabel
	Value  *float64
}

type AggregatedTextFrame struct {
	*DataFrame
	Text            string
	SkipTTS         bool
	AppendToContext bool
	TextType        TextType
	ContextID       string
	RawText         string
	WillBeSpoken    bool
}

func NewAggregatedTextFrame(text string, textType TextType) *AggregatedTextFrame {
	return &AggregatedTextFrame{
		DataFrame:       newDataFrame("AggregatedTextFrame"),
		Text:            text,
		AppendToContext: true,
		TextType:        textType,
	}
}

type TTSTextFrame struct {
	*DataFrame
	Text            string
	SkipTTS         bool
	AppendToContext bool
	TextType        TextType
	ContextID       string
	RawText         string
	WillBeSpoken    bool
	SegmentID       *int
	Delivery        []string
}

func NewTTSTextFrame(text string, textType TextType, contextID string) *TTSTextFrame {
	return &TTSTextFrame{
		DataFrame:       newDataFrame("TTSTextFrame"),
		Text:            text,
		AppendToContext: true,
		TextType:        textType,
		ContextID:       contextID,
	}
}

type AggregatedTextProgressFrame struct {
	*DataFrame
	SegmentID       int
	ContextID       string
	Text            string
	TextType        TextType
	AccumulatedText string
	RemainingText   string
	Delivery        []string
}

func NewAggregatedTextProgressFrame(segmentID int, contextID, text string, textType TextType, accumulated, remaining string) *AggregatedTextProgressFrame {
	return &AggregatedTextProgressFrame{
		DataFrame:       newDataFrame("AggregatedTextProgressFrame"),
		SegmentID:       segmentID,
		ContextID:       contextID,
		Text:            text,
		TextType:        textType,
		AccumulatedText: accumulated,
		RemainingText:   remaining,
	}
}

type VisionTextFrame struct {
	*DataFrame
	Text    string
	SkipTTS bool
}

func NewVisionTextFrame(text string) *VisionTextFrame {
	return &VisionTextFrame{
		DataFrame: newDataFrame("VisionTextFrame"),
		Text:      text,
	}
}

type InterimTranscriptionFrame struct {
	*DataFrame
	Text       string
	UserID     string
	Timestamp  time.Time
	Language   string
	Result     any
	Speaker    *TranscriptionSpeaker
	Perception []PerceptionSignal
}

func NewInterimTranscriptionFrame(text, userID string) *InterimTranscriptionFrame {
	return &InterimTranscriptionFrame{
		DataFrame: newDataFrame("InterimTranscriptionFrame"),
		Text:      text,
		UserID:    userID,
		Timestamp: time.Now(),
	}
}

type EagerTranscriptionFrame struct {
	*DataFrame
	Text       string
	UserID     string
	Timestamp  time.Time
	Language   string
	Result     any
	Speaker    *TranscriptionSpeaker
	Perception []PerceptionSignal
}

func NewEagerTranscriptionFrame(text, userID string) *EagerTranscriptionFrame {
	return &EagerTranscriptionFrame{
		DataFrame: newDataFrame("EagerTranscriptionFrame"),
		Text:      text,
		UserID:    userID,
		Timestamp: time.Now(),
	}
}

type TranslationFrame struct {
	*DataFrame
	Text      string
	UserID    string
	Timestamp time.Time
	Language  string
}

func NewTranslationFrame(text, userID, language string) *TranslationFrame {
	return &TranslationFrame{
		DataFrame: newDataFrame("TranslationFrame"),
		Text:      text,
		UserID:    userID,
		Timestamp: time.Now(),
		Language:  language,
	}
}

type UserBackchannelFrame struct {
	*DataFrame
	Text      string
	UserID    string
	Timestamp time.Time
}

func NewUserBackchannelFrame(text, userID string) *UserBackchannelFrame {
	return &UserBackchannelFrame{
		DataFrame: newDataFrame("UserBackchannelFrame"),
		Text:      text,
		UserID:    userID,
		Timestamp: time.Now(),
	}
}

type InputTextRawFrame struct {
	*SystemFrame
	Text string
}

func NewInputTextRawFrame(text string) *InputTextRawFrame {
	return &InputTextRawFrame{
		SystemFrame: newSystemFrame("InputTextRawFrame"),
		Text:        text,
	}
}
