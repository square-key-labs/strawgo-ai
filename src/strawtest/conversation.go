package strawtest

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type observedCall struct {
	id     string
	name   string
	args   map[string]any
	result string
	done   bool
}

type botTurn struct {
	open       int
	responded  bool
	inFlight   map[string]bool
	runPending bool
	runOwed    bool
	speaking   bool
	tts        bool
	calls      []*observedCall
	taken      int
	llmText    strings.Builder
	ttsText    strings.Builder
	segments   []string
}

func newBotTurn() *botTurn {
	return &botTurn{inFlight: map[string]bool{}}
}

func (t *botTurn) settled() bool {
	return t.responded && t.open == 0 && len(t.inFlight) == 0 && !t.runOwed && !t.speaking && !t.tts
}

func (t *botTurn) reply() string {
	parts := append([]string(nil), t.segments...)
	if s := t.pendingSegment(); s != "" {
		parts = append(parts, s)
	}
	return strings.Join(parts, " ")
}

func (t *botTurn) pendingSegment() string {
	text := t.ttsText.String()
	if text == "" {
		text = t.llmText.String()
	}
	return strings.TrimSpace(text)
}

type conversation struct {
	mu         sync.Mutex
	changed    chan struct{}
	activity   uint64
	turn       *botTurn
	calls      map[string]*observedCall
	transcript []Message
}

func newConversation() *conversation {
	return &conversation{changed: make(chan struct{}), turn: newBotTurn(), calls: map[string]*observedCall{}}
}

func (c *conversation) notifyLocked() {
	c.activity++
	close(c.changed)
	c.changed = make(chan struct{})
}

func (c *conversation) flushSegmentLocked() {
	if s := c.turn.pendingSegment(); s != "" {
		c.turn.segments = append(c.turn.segments, s)
		c.transcript = append(c.transcript, Message{Role: RoleBot, Text: s})
	}
	c.turn.llmText.Reset()
	c.turn.ttsText.Reset()
}

func (c *conversation) userSays(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.flushSegmentLocked()
	c.turn = newBotTurn()
	c.transcript = append(c.transcript, Message{Role: RoleUser, Text: text})
	c.notifyLocked()
}

func (c *conversation) finishTurn() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.flushSegmentLocked()
	return c.turn.reply()
}

func (c *conversation) currentReply() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.turn.reply()
}

func (c *conversation) snapshotTranscript() []Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]Message(nil), c.transcript...)
}

func (c *conversation) observe(frame frames.Frame) {
	c.mu.Lock()
	defer c.mu.Unlock()
	t := c.turn
	switch f := frame.(type) {
	case *frames.LLMFullResponseStartFrame:
		t.open++
		t.runOwed = false
	case *frames.LLMFullResponseEndFrame:
		if t.open > 0 {
			t.open--
		}
		t.responded = true
	case *frames.LLMTextFrame:
		t.llmText.WriteString(f.Text)
	case *frames.TTSTextFrame:
		if f.Text == "" {
			return
		}
		if (f.TextType == frames.TextTypeWord || f.TextType == frames.TextTypeSentence) && t.ttsText.Len() > 0 {
			t.ttsText.WriteByte(' ')
		}
		t.ttsText.WriteString(f.Text)
	case *frames.BotStartedSpeakingFrame:
		t.speaking = true
	case *frames.BotStoppedSpeakingFrame:
		t.speaking = false
	case *frames.TTSStartedFrame:
		t.tts = true
	case *frames.TTSStoppedFrame:
		t.tts = false
	case *frames.FunctionCallInProgressFrame:
		if _, seen := c.calls[f.ToolCallID]; seen {
			return
		}
		call := &observedCall{id: f.ToolCallID, name: f.FunctionName, args: f.Arguments}
		c.calls[f.ToolCallID] = call
		c.flushSegmentLocked()
		t.calls = append(t.calls, call)
		t.inFlight[f.ToolCallID] = true
		c.transcript = append(c.transcript, Message{Role: RoleFunctionCall, Text: callSignature(f.FunctionName, f.Arguments)})
	case *frames.FunctionCallResultFrame:
		call, ok := c.calls[f.ToolCallID]
		if !ok || call.done {
			return
		}
		final := f.Properties == nil || f.Properties.IsFinal
		if final {
			call.done = true
			call.result = encodeValue(f.Result)
			if f.Error != "" {
				call.result = "error: " + f.Error
			}
			delete(t.inFlight, f.ToolCallID)
			c.transcript = append(c.transcript, Message{Role: RoleFunctionResult, Text: f.FunctionName + " -> " + call.result})
		}
		if final || f.Result != nil {
			run := true
			if f.Properties != nil && f.Properties.RunLLM != nil {
				run = *f.Properties.RunLLM
			} else if f.RunLLM != nil {
				run = *f.RunLLM
			}
			if run {
				t.runPending = true
			}
		}
		if t.runPending && len(t.inFlight) == 0 {
			t.runPending = false
			t.runOwed = true
		}
	case *frames.FunctionCallCancelFrame:
		call, ok := c.calls[f.ToolCallID]
		if !ok || call.done {
			return
		}
		call.done = true
		call.result = "cancelled"
		delete(t.inFlight, f.ToolCallID)
		c.transcript = append(c.transcript, Message{Role: RoleFunctionResult, Text: f.FunctionName + " -> cancelled"})
		if t.runPending && len(t.inFlight) == 0 {
			t.runPending = false
			t.runOwed = true
		}
	default:
		return
	}
	c.notifyLocked()
}

func callSignature(name string, args map[string]any) string {
	if len(args) == 0 {
		return name + "()"
	}
	return name + "(" + encodeValue(args) + ")"
}

func encodeValue(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	return string(b)
}
