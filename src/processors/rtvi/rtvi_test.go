package rtvi

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

type sink struct {
	mu        sync.Mutex
	got       []frames.Frame
	onReceive func(frames.Frame)
}

func (s *sink) ProcessFrame(context.Context, frames.Frame, frames.FrameDirection) error { return nil }
func (s *sink) QueueFrame(f frames.Frame, _ frames.FrameDirection) error {
	s.mu.Lock()
	s.got = append(s.got, f)
	cb := s.onReceive
	s.mu.Unlock()
	if cb != nil {
		cb(f)
	}
	return nil
}
func (s *sink) PushFrame(frames.Frame, frames.FrameDirection) error { return nil }
func (s *sink) Link(processors.FrameProcessor)                      {}
func (s *sink) SetPrev(processors.FrameProcessor)                   {}
func (s *sink) Start(context.Context) error                         { return nil }
func (s *sink) Stop() error                                         { return nil }
func (s *sink) Name() string                                        { return "sink" }

func (s *sink) frames() []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]frames.Frame(nil), s.got...)
}

func (s *sink) reset() {
	s.mu.Lock()
	s.got = nil
	s.mu.Unlock()
}

type wireMsg struct {
	Label string         `json:"label"`
	Type  string         `json:"type"`
	ID    *string        `json:"id"`
	Data  map[string]any `json:"data"`
}

func (s *sink) messages(t *testing.T) []wireMsg {
	t.Helper()
	var out []wireMsg
	for _, f := range s.frames() {
		u, ok := f.(*frames.OutputTransportMessageUrgentFrame)
		if !ok {
			continue
		}
		b, err := json.Marshal(u.Message)
		if err != nil {
			t.Fatal(err)
		}
		var m wireMsg
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatal(err)
		}
		if m.Label != MessageLabel {
			t.Fatalf("label = %q", m.Label)
		}
		out = append(out, m)
	}
	return out
}

func types(msgs []wireMsg) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.Type
	}
	return out
}

func findType(msgs []wireMsg, typ string) (wireMsg, bool) {
	for _, m := range msgs {
		if m.Type == typ {
			return m, true
		}
	}
	return wireMsg{}, false
}

func newRTVI(t *testing.T, params Params) (*Processor, *sink, *sink) {
	t.Helper()
	p := New(params)
	down, up := &sink{}, &sink{}
	p.Link(down)
	p.SetPrev(up)
	return p, down, up
}

func client(t *testing.T, p *Processor, typ, id string, data map[string]any) {
	t.Helper()
	msg := map[string]any{"label": MessageLabel, "type": typ, "id": id}
	if data != nil {
		msg["data"] = data
	}
	if err := p.HandleFrame(context.Background(), frames.NewInputTransportMessageFrame(msg), frames.Downstream); err != nil {
		t.Fatal(err)
	}
}

func TestClientReadyBotReadyHandshake(t *testing.T) {
	readyCalls := 0
	p, down, _ := newRTVI(t, Params{About: map[string]any{"name": "bot"}, OnClientReady: func(*Processor) { readyCalls++ }})
	client(t, p, "client-ready", "r1", map[string]any{"version": "2.3.0", "about": map[string]any{"library": "js"}})
	fs := down.frames()
	if len(fs) != 2 {
		t.Fatalf("got %d frames, want 2", len(fs))
	}
	if _, ok := fs[0].(*frames.InputTransportStartAudioStreamingFrame); !ok {
		t.Fatalf("first frame = %T", fs[0])
	}
	msgs := down.messages(t)
	if len(msgs) != 1 || msgs[0].Type != TypeBotReady || msgs[0].ID == nil || *msgs[0].ID != "r1" {
		t.Fatalf("messages = %+v", msgs)
	}
	if msgs[0].Data["version"] != ProtocolVersion || msgs[0].Data["about"].(map[string]any)["name"] != "bot" {
		t.Fatalf("bot-ready data = %v", msgs[0].Data)
	}
	if !p.IsClientReady() || !p.IsBotReady() || readyCalls != 1 || p.ClientVersion() != [3]int{2, 3, 0} {
		t.Fatalf("ready=%v bot=%v calls=%d version=%v", p.IsClientReady(), p.IsBotReady(), readyCalls, p.ClientVersion())
	}
}

func TestClientReadyVersionProblems(t *testing.T) {
	cases := []struct {
		version   string
		wantError bool
		botVer    string
	}{
		{"3.0.0", true, ProtocolVersion},
		{"abc", true, ProtocolVersion},
		{"", true, ProtocolVersion},
		{"1.2.3", false, "1.2.3"},
		{"2.0.0", false, ProtocolVersion},
	}
	for _, c := range cases {
		p, down, _ := newRTVI(t, Params{})
		data := map[string]any{}
		if c.version != "" {
			data["version"] = c.version
		}
		client(t, p, "client-ready", "x", data)
		msgs := down.messages(t)
		errMsg, hasErr := findType(msgs, TypeErrorResponse)
		if hasErr != c.wantError {
			t.Fatalf("version %q: error-response=%v, want %v (%v)", c.version, hasErr, c.wantError, types(msgs))
		}
		if hasErr && (errMsg.ID == nil || *errMsg.ID != "x" || errMsg.Data["error"] == "") {
			t.Fatalf("version %q: error = %+v", c.version, errMsg)
		}
		ready, ok := findType(msgs, TypeBotReady)
		if !ok || ready.Data["version"] != c.botVer {
			t.Fatalf("version %q: bot-ready = %+v", c.version, ready)
		}
	}
}

func TestDisableAutoBotReady(t *testing.T) {
	p, down, _ := newRTVI(t, Params{DisableAutoBotReady: true})
	client(t, p, "client-ready", "r1", map[string]any{"version": "2.3.0"})
	if _, ok := findType(down.messages(t), TypeBotReady); ok || p.IsBotReady() {
		t.Fatal("bot-ready sent automatically")
	}
	p.SetBotReady(map[string]any{"custom": true})
	ready, ok := findType(down.messages(t), TypeBotReady)
	if !ok || *ready.ID != "r1" || ready.Data["about"].(map[string]any)["custom"] != true || !p.IsBotReady() {
		t.Fatalf("manual bot-ready = %+v", ready)
	}
}

func TestOnBotStartedOnce(t *testing.T) {
	calls := 0
	p, down, _ := newRTVI(t, Params{OnBotStarted: func(*Processor) { calls++ }})
	_ = p.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream)
	_ = p.HandleFrame(context.Background(), frames.NewStartFrame(), frames.Downstream)
	if calls != 1 || len(down.frames()) != 2 {
		t.Fatalf("calls=%d forwarded=%d", calls, len(down.frames()))
	}
}

func pushAll(o *Observer, source string, fs ...frames.Frame) {
	for _, f := range fs {
		o.OnPushFrame(pipeline.PushFrameEvent{ProcessorName: source, Frame: f, Direction: frames.Downstream, Timestamp: time.Now()})
	}
}

func botTurn(o *Observer) {
	tr := frames.NewTranscriptionFrame("what time is it", true)
	tts := frames.NewTTSTextFrame("Hello there.", frames.TextTypeSentence, "ctx")
	tts.WillBeSpoken = true
	pushAll(o, DefaultOutputSource,
		frames.NewUserStartedSpeakingFrame(),
		tr,
		frames.NewUserStoppedSpeakingFrame(),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Hello there."),
		frames.NewTTSStartedFrame(),
		tts,
		frames.NewBotStartedSpeakingFrame(),
		frames.NewTTSStoppedFrame(),
		frames.NewLLMFullResponseEndFrame(),
		frames.NewBotStoppedSpeakingFrame(),
	)
}

func TestFullBotTurnMessageOrder(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	o := p.NewObserver(nil)
	botTurn(o)
	got := types(down.messages(t))
	want := []string{
		TypeUserStartedSpeaking,
		TypeUserInput, TypeUserTranscription,
		TypeUserStoppedSpeaking,
		TypeBotLLMStarted,
		TypeBotLLMText, TypeBotTranscription,
		TypeBotTTSStarted,
		TypeBotStartedSpeaking, TypeBotOutput, TypeBotTTSText,
		TypeBotTTSStopped,
		TypeBotLLMStopped,
		TypeBotStoppedSpeaking,
	}
	if len(got) != len(want) {
		t.Fatalf("types = %v\nwant    %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("types = %v\nwant    %v", got, want)
		}
	}
	msgs := down.messages(t)
	if m, _ := findType(msgs, TypeUserTranscription); m.Data["text"] != "what time is it" || m.Data["final"] != true {
		t.Fatalf("transcription = %v", m.Data)
	}
	if m, _ := findType(msgs, TypeBotTTSText); m.Data["text"] != "Hello there." {
		t.Fatalf("tts text = %v", m.Data)
	}
}

func TestObserverReportsEachFrameOnce(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	o := p.NewObserver(nil)
	f := frames.NewBotStartedSpeakingFrame()
	pushAll(o, "A", f)
	pushAll(o, "B", f)
	if got := types(down.messages(t)); len(got) != 1 {
		t.Fatalf("duplicate reports: %v", got)
	}
}

func TestObserverDisableFlags(t *testing.T) {
	cases := []struct {
		name    string
		disable func(*ObserverParams)
		gone    []string
	}{
		{"user speaking", func(p *ObserverParams) { p.UserSpeakingEnabled = false }, []string{TypeUserStartedSpeaking, TypeUserStoppedSpeaking}},
		{"user transcription", func(p *ObserverParams) { p.UserTranscriptionEnabled = false }, []string{TypeUserTranscription}},
		{"user input", func(p *ObserverParams) { p.UserInputEnabled = false }, []string{TypeUserInput}},
		{"bot llm", func(p *ObserverParams) { p.BotLLMEnabled = false }, []string{TypeBotLLMStarted, TypeBotLLMText, TypeBotLLMStopped, TypeBotTranscription}},
		{"bot tts", func(p *ObserverParams) { p.BotTTSEnabled = false }, []string{TypeBotTTSStarted, TypeBotTTSStopped, TypeBotTTSText}},
		{"bot speaking", func(p *ObserverParams) { p.BotSpeakingEnabled = false }, []string{TypeBotStartedSpeaking, TypeBotStoppedSpeaking}},
		{"bot output", func(p *ObserverParams) { p.BotOutputEnabled = false }, []string{TypeBotOutput}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p, down, _ := newRTVI(t, Params{})
			params := DefaultObserverParams()
			c.disable(&params)
			botTurn(p.NewObserver(&params))
			got := types(down.messages(t))
			for _, g := range c.gone {
				for _, typ := range got {
					if typ == g {
						t.Fatalf("%s still sent: %v", g, got)
					}
				}
			}
			if len(got) == 0 {
				t.Fatal("everything disabled")
			}
		})
	}
}

func TestObserverIgnoresTTSTextFromOtherSources(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	o := p.NewObserver(nil)
	pushAll(o, "SomeTTS", frames.NewBotStartedSpeakingFrame())
	pushAll(o, "SomeTTS", frames.NewTTSTextFrame("hidden.", frames.TextTypeSentence, ""))
	if _, ok := findType(down.messages(t), TypeBotTTSText); ok {
		t.Fatal("tts text from a non-output source was reported")
	}
}

func TestObserverVADAndMetrics(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	params := DefaultObserverParams()
	params.VADUserSpeakingEnabled = true
	o := p.NewObserver(&params)
	pushAll(o, "x", frames.NewVADUserStartedSpeakingFrame(0.2), frames.NewMetricsFrame(frames.TTFBMetricsData{MetricsBase: frames.MetricsBase{Processor: "llm", Model: "m"}, Value: 250 * time.Millisecond}))
	msgs := down.messages(t)
	if _, ok := findType(msgs, TypeVADUserStartedSpeaking); !ok {
		t.Fatalf("vad not reported: %v", types(msgs))
	}
	m, ok := findType(msgs, TypeMetrics)
	if !ok {
		t.Fatal("metrics not reported")
	}
	ttfb := m.Data["ttfb"].([]any)[0].(map[string]any)
	if ttfb["processor"] != "llm" || ttfb["value"] != 0.25 {
		t.Fatalf("ttfb = %v", ttfb)
	}
}

func TestObserverFunctionCallReportLevels(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	params := DefaultObserverParams()
	params.FunctionCallReportLevel = map[string]FunctionCallReportLevel{"*": FunctionCallReportNone, "weather": FunctionCallReportFull, "secret": FunctionCallReportDisabled}
	o := p.NewObserver(&params)
	pushAll(o, "llm",
		frames.NewFunctionCallInProgressFrame("c1", "weather", map[string]any{"city": "Paris"}, false),
		frames.NewFunctionCallResultFrame("c1", "weather", "sunny", nil),
		frames.NewFunctionCallInProgressFrame("c2", "other", map[string]any{"a": 1}, false),
		frames.NewFunctionCallInProgressFrame("c3", "secret", nil, false),
	)
	msgs := down.messages(t)
	if len(msgs) != 3 {
		t.Fatalf("messages = %v", types(msgs))
	}
	if msgs[0].Data["function_name"] != "weather" || msgs[0].Data["arguments"].(map[string]any)["city"] != "Paris" {
		t.Fatalf("full in-progress = %v", msgs[0].Data)
	}
	if msgs[1].Type != TypeLLMFunctionCallStopped || msgs[1].Data["result"] != "sunny" {
		t.Fatalf("full stopped = %v", msgs[1].Data)
	}
	if _, named := msgs[2].Data["function_name"]; named || msgs[2].Data["arguments"] != nil {
		t.Fatalf("level none leaked details: %v", msgs[2].Data)
	}
}

func TestSendText(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	client(t, p, "send-text", "s1", map[string]any{"content": "hello", "options": map[string]any{"audio_response": false}})
	fs := down.frames()
	if len(fs) != 5 {
		names := make([]string, len(fs))
		for i, f := range fs {
			names[i] = f.Name()
		}
		t.Fatalf("frames = %v", names)
	}
	if _, ok := fs[0].(*frames.InterruptionFrame); !ok {
		t.Fatalf("frame 0 = %T, want interruption", fs[0])
	}
	if c, ok := fs[1].(*frames.LLMConfigureOutputFrame); !ok || !c.SkipTTS {
		t.Fatalf("frame 1 = %#v, want skip-tts on", fs[1])
	}
	app, ok := fs[2].(*frames.LLMMessagesAppendFrame)
	if !ok || !app.RunLLM || len(app.Messages) != 1 || app.Messages[0].Role != llm.RoleUser || app.Messages[0].Text() != "hello" {
		t.Fatalf("frame 2 = %#v", fs[2])
	}
	if st, ok := fs[3].(*SendTextFrame); !ok || st.MsgID != "s1" || st.Text != "hello" {
		t.Fatalf("frame 3 = %#v", fs[3])
	}
	if c, ok := fs[4].(*frames.LLMConfigureOutputFrame); !ok || c.SkipTTS {
		t.Fatalf("frame 4 = %#v, want skip-tts restored", fs[4])
	}
}

func TestSendTextWithoutRunOrAudioToggle(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	client(t, p, "send-text", "s2", map[string]any{"content": "later", "options": map[string]any{"run_immediately": false}})
	fs := down.frames()
	if len(fs) != 2 {
		t.Fatalf("got %d frames, want append + send-text", len(fs))
	}
	if app, ok := fs[0].(*frames.LLMMessagesAppendFrame); !ok || app.RunLLM {
		t.Fatalf("frame 0 = %#v", fs[0])
	}
	down.reset()
	client(t, p, "send-text", "s3", map[string]any{})
	if m, ok := findType(down.messages(t), TypeErrorResponse); !ok || *m.ID != "s3" {
		t.Fatal("missing content not rejected")
	}
}

func TestSendTextWaitsForInterruptionToReachOutput(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	o := p.NewObserver(nil)
	reached := make(chan struct{}, 1)
	down.onReceive = func(f frames.Frame) {
		if _, ok := f.(*frames.InterruptionFrame); ok {
			pushAll(o, DefaultOutputSource, f)
			reached <- struct{}{}
		}
	}
	start := time.Now()
	client(t, p, "send-text", "s4", map[string]any{"content": "hi"})
	select {
	case <-reached:
	default:
		t.Fatal("interruption never reached output")
	}
	if el := time.Since(start); el >= flushTimeout {
		t.Fatalf("send-text waited %v for flush despite the interruption reaching output", el)
	}
	if _, ok := findType(down.messages(t), TypeBotInterrupted); !ok {
		t.Fatal("bot-interrupted not reported")
	}
}

func TestClientMessage(t *testing.T) {
	var got ClientMessage
	p, down, _ := newRTVI(t, Params{OnClientMessage: func(_ *Processor, m ClientMessage) { got = m }})
	client(t, p, "client-message", "c1", map[string]any{"t": "ping", "d": map[string]any{"n": 1.0}})
	fs := down.frames()
	cm, ok := fs[0].(*ClientMessageFrame)
	if !ok || cm.MsgID != "c1" || cm.Type != "ping" || got.Type != "ping" || got.MsgID != "c1" {
		t.Fatalf("client message frame=%#v callback=%+v", fs[0], got)
	}
	down.reset()
	client(t, p, "client-message", "c2", map[string]any{"d": 1})
	if m, ok := findType(down.messages(t), TypeErrorResponse); !ok || *m.ID != "c2" {
		t.Fatal("missing t not rejected")
	}
}

func TestUnsupportedTypeAndNonRTVIMessages(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	client(t, p, "bogus", "b1", nil)
	m, ok := findType(down.messages(t), TypeErrorResponse)
	if !ok || m.Data["error"] != "Unsupported type bogus" {
		t.Fatalf("unsupported = %+v", m)
	}
	down.reset()
	other := frames.NewInputTransportMessageFrame(map[string]any{"label": "x", "type": "client-ready"})
	_ = p.HandleFrame(context.Background(), other, frames.Downstream)
	if fs := down.frames(); len(fs) != 1 || fs[0] != other {
		t.Fatal("non-rtvi message not forwarded unchanged")
	}
	if p.IsClientReady() {
		t.Fatal("non-rtvi message handled as client-ready")
	}
}

func TestErrorFrameReported(t *testing.T) {
	p, down, _ := newRTVI(t, Params{})
	ef := frames.NewErrorFrame(errors.New("boom"))
	ef.Fatal = true
	_ = p.HandleFrame(context.Background(), ef, frames.Upstream)
	m, ok := findType(down.messages(t), TypeError)
	if !ok || m.Data["error"] != "boom" || m.Data["fatal"] != true {
		t.Fatalf("error message = %+v", m)
	}
}

func TestDTMFAudioAndDisconnect(t *testing.T) {
	p, down, up := newRTVI(t, Params{})
	client(t, p, "dtmf", "d1", map[string]any{"buttons": []any{"1", "#"}})
	fs := down.frames()
	if len(fs) != 2 || fs[0].(*frames.InputDTMFFrame).Button != "1" || fs[1].(*frames.InputDTMFFrame).Button != "#" {
		t.Fatalf("dtmf frames = %v", fs)
	}
	down.reset()
	client(t, p, "dtmf", "d2", map[string]any{"buttons": []any{"z"}})
	if _, ok := findType(down.messages(t), TypeErrorResponse); !ok {
		t.Fatal("invalid dtmf not rejected")
	}
	down.reset()
	client(t, p, "raw-audio", "a1", map[string]any{"base64Audio": "AAECAw==", "sampleRate": 16000.0, "numChannels": 1.0})
	af, ok := down.frames()[0].(*frames.AudioFrame)
	if !ok || af.SampleRate != 16000 || string(af.Data) != "\x00\x01\x02\x03" {
		t.Fatalf("raw audio = %#v", down.frames())
	}
	client(t, p, "disconnect-bot", "x", nil)
	if fs := up.frames(); len(fs) != 1 {
		t.Fatalf("upstream = %v", fs)
	} else if _, ok := fs[0].(*frames.EndTaskFrame); !ok {
		t.Fatalf("upstream frame = %T", fs[0])
	}
}
