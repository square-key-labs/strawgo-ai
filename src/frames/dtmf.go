package frames

import (
	"fmt"
	"strings"
)

type KeypadEntry string

const (
	KeypadOne   KeypadEntry = "1"
	KeypadTwo   KeypadEntry = "2"
	KeypadThree KeypadEntry = "3"
	KeypadFour  KeypadEntry = "4"
	KeypadFive  KeypadEntry = "5"
	KeypadSix   KeypadEntry = "6"
	KeypadSeven KeypadEntry = "7"
	KeypadEight KeypadEntry = "8"
	KeypadNine  KeypadEntry = "9"
	KeypadZero  KeypadEntry = "0"
	KeypadPound KeypadEntry = "#"
	KeypadStar  KeypadEntry = "*"
)

func (k KeypadEntry) Valid() bool {
	switch k {
	case KeypadOne, KeypadTwo, KeypadThree, KeypadFour, KeypadFive, KeypadSix,
		KeypadSeven, KeypadEight, KeypadNine, KeypadZero, KeypadPound, KeypadStar:
		return true
	}
	return false
}

func ParseKeypadEntries(s string) ([]KeypadEntry, error) {
	out := make([]KeypadEntry, 0, len(s))
	for _, r := range s {
		k := KeypadEntry(string(r))
		if !k.Valid() {
			return nil, fmt.Errorf("invalid DTMF keypad entry %q", r)
		}
		out = append(out, k)
	}
	return out, nil
}

func KeypadEntriesString(buttons []KeypadEntry) string {
	var b strings.Builder
	for _, k := range buttons {
		b.WriteString(string(k))
	}
	return b.String()
}

type DTMFFrame interface {
	Frame
	DTMFButtons() []KeypadEntry
}

type InputDTMFFrame struct {
	*SystemFrame
	Button KeypadEntry
}

func NewInputDTMFFrame(button KeypadEntry) *InputDTMFFrame {
	return &InputDTMFFrame{
		SystemFrame: newSystemFrame("InputDTMFFrame"),
		Button:      button,
	}
}

func (f *InputDTMFFrame) DTMFButtons() []KeypadEntry {
	return []KeypadEntry{f.Button}
}

type OutputDTMFFrame struct {
	*DataFrame
	Buttons []KeypadEntry
}

func NewOutputDTMFFrame(buttons ...KeypadEntry) *OutputDTMFFrame {
	return &OutputDTMFFrame{
		DataFrame: newDataFrame("OutputDTMFFrame"),
		Buttons:   buttons,
	}
}

func NewOutputDTMFFrameFromString(s string) (*OutputDTMFFrame, error) {
	buttons, err := ParseKeypadEntries(s)
	if err != nil {
		return nil, err
	}
	return NewOutputDTMFFrame(buttons...), nil
}

func (f *OutputDTMFFrame) DTMFButtons() []KeypadEntry {
	return f.Buttons
}

func (f *OutputDTMFFrame) ToString() string {
	return KeypadEntriesString(f.Buttons)
}

type OutputDTMFUrgentFrame struct {
	*SystemFrame
	Buttons []KeypadEntry
}

func NewOutputDTMFUrgentFrame(buttons ...KeypadEntry) *OutputDTMFUrgentFrame {
	return &OutputDTMFUrgentFrame{
		SystemFrame: newSystemFrame("OutputDTMFUrgentFrame"),
		Buttons:     buttons,
	}
}

func NewOutputDTMFUrgentFrameFromString(s string) (*OutputDTMFUrgentFrame, error) {
	buttons, err := ParseKeypadEntries(s)
	if err != nil {
		return nil, err
	}
	return NewOutputDTMFUrgentFrame(buttons...), nil
}

func (f *OutputDTMFUrgentFrame) DTMFButtons() []KeypadEntry {
	return f.Buttons
}

func (f *OutputDTMFUrgentFrame) ToString() string {
	return KeypadEntriesString(f.Buttons)
}
