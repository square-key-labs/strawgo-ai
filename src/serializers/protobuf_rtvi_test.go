package serializers

import (
	"encoding/base64"
	"testing"
	"time"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func pbRoundTrip(t *testing.T, s *ProtobufFrameSerializer, f frames.Frame) frames.Frame {
	t.Helper()
	msg := one(t)(s.Serialize(nil, f))
	if msg.Type != BinaryMessage {
		t.Fatal("protobuf message is not binary")
	}
	data := append([]byte(nil), msg.Data...)
	out, err := s.Deserialize(Binary(data))
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestProtobufTextGoldenShape(t *testing.T) {
	s := NewProtobufFrameSerializer()
	f := frames.NewTextFrame("hi")
	msg := one(t)(s.Serialize(nil, f))
	num, typ, n := protowire.ConsumeTag(msg.Data)
	if num != 1 || typ != protowire.BytesType {
		t.Fatalf("outer tag = %d/%d", num, typ)
	}
	inner, _ := protowire.ConsumeBytes(msg.Data[n:])
	var want []byte
	want = protowire.AppendTag(want, 1, protowire.VarintType)
	want = protowire.AppendVarint(want, f.ID())
	want = protowire.AppendTag(want, 2, protowire.BytesType)
	want = protowire.AppendString(want, f.Name())
	want = protowire.AppendTag(want, 3, protowire.BytesType)
	want = protowire.AppendString(want, "hi")
	if string(inner) != string(want) {
		t.Fatalf("inner = %x, want %x", inner, want)
	}
}

func TestProtobufRoundTrips(t *testing.T) {
	s := NewProtobufFrameSerializer()
	if tf, ok := pbRoundTrip(t, s, frames.NewTextFrame("hello")).(*frames.TextFrame); !ok || tf.Text != "hello" {
		t.Fatalf("text round trip = %#v", tf)
	}
	tr := frames.NewTranscriptionFrame("said", true)
	tr.SetMetadata("user_id", "u1")
	tr.Timestamp = time.Date(2026, 1, 2, 3, 4, 5, 600, time.UTC)
	got, ok := pbRoundTrip(t, s, tr).(*frames.TranscriptionFrame)
	if !ok || got.Text != "said" || got.Metadata()["user_id"] != "u1" || !got.Timestamp.Equal(tr.Timestamp) || !got.IsFinal {
		t.Fatalf("transcription round trip = %#v", got)
	}
	if _, ok := pbRoundTrip(t, s, frames.NewInterruptionFrame()).(*frames.InterruptionFrame); !ok {
		t.Fatal("interruption round trip failed")
	}
	pcm := tonePCM(160, 16000, 440, 8000)
	af, ok := pbRoundTrip(t, s, frames.NewTTSAudioFrame(pcm, 16000, 1)).(*frames.AudioFrame)
	if !ok || string(af.Data) != string(pcm) || af.SampleRate != 16000 || af.Channels != 1 || af.Metadata()["codec"] != "linear16" {
		t.Fatalf("audio round trip = %#v", af)
	}
	mf, ok := pbRoundTrip(t, s, frames.NewOutputTransportMessageFrame(map[string]any{"a": 1.0})).(*frames.InputTransportMessageFrame)
	if !ok || mf.Message.(map[string]any)["a"] != 1.0 {
		t.Fatalf("message round trip = %#v", mf)
	}
	none(t)(s.Serialize(nil, frames.NewEndFrame()))
}

func TestProtobufAudioDataIsCopied(t *testing.T) {
	s := NewProtobufFrameSerializer()
	msg := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame([]byte{1, 2, 3, 4}, 8000, 1)))
	buf := append([]byte(nil), msg.Data...)
	f, _ := s.Deserialize(Binary(buf))
	for i := range buf {
		buf[i] = 0
	}
	if af := f.(*frames.AudioFrame); string(af.Data) != string([]byte{1, 2, 3, 4}) {
		t.Fatal("audio frame aliases the wire buffer")
	}
}

func TestProtobufRejectsGarbage(t *testing.T) {
	s := NewProtobufFrameSerializer()
	for _, b := range [][]byte{{0xff}, {0x08, 0x01}, {0x0a, 0x05, 0x01}} {
		if _, err := s.Deserialize(Binary(b)); err == nil {
			t.Fatalf("garbage %x accepted", b)
		}
	}
}

func TestRTVIClientSerializeAudio(t *testing.T) {
	s := NewRTVIClientSerializer()
	pcm := tonePCM(80, 16000, 440, 8000)
	m1 := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 16000, 1))))
	m2 := jsonMsg(t, one(t)(s.SerializeAudio(nil, pcm, 16000)))
	if m1["label"] != "rtvi-ai" || m1["type"] != "raw-audio" || m1["id"] == m2["id"] {
		t.Fatalf("raw-audio = %v / %v", m1, m2)
	}
	raw, _ := base64.StdEncoding.DecodeString(path(m1, "data", "base64Audio").(string))
	if string(raw) != string(pcm) || path(m1, "data", "sampleRate") != float64(16000) || path(m1, "data", "numChannels") != float64(1) {
		t.Fatalf("raw-audio data = %v", m1["data"])
	}
}

func TestRTVIClientSerializeOnlyRTVIMessages(t *testing.T) {
	s := NewRTVIClientSerializer()
	none(t)(s.Serialize(nil, frames.NewOutputTransportMessageFrame(map[string]any{"type": "x"})))
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewOutputTransportMessageUrgentFrame(map[string]any{"label": "rtvi-ai", "type": "client-ready"}))))
	if m["type"] != "client-ready" {
		t.Fatalf("rtvi = %v", m)
	}
	none(t)(s.Serialize(nil, frames.NewTextFrame("x")))
}

func rtviIn(t *testing.T, s *RTVIClientSerializer, typ string, data map[string]any) frames.Frame {
	t.Helper()
	f, err := s.Deserialize(textMsg(t, map[string]any{"label": "rtvi-ai", "type": typ, "data": data}))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestRTVIClientDeserialize(t *testing.T) {
	s := NewRTVIClientSerializer()
	checks := []struct {
		typ  string
		data map[string]any
		ok   func(frames.Frame) bool
	}{
		{"bot-llm-started", nil, func(f frames.Frame) bool { _, ok := f.(*frames.LLMFullResponseStartFrame); return ok }},
		{"bot-llm-text", map[string]any{"text": "hey"}, func(f frames.Frame) bool { x, ok := f.(*frames.LLMTextFrame); return ok && x.Text == "hey" }},
		{"bot-llm-stopped", nil, func(f frames.Frame) bool { _, ok := f.(*frames.LLMFullResponseEndFrame); return ok }},
		{"bot-llm-marker", map[string]any{"raw": "r", "text": "t", "kind": "k", "markers": []any{"a", "b"}}, func(f frames.Frame) bool {
			x, ok := f.(*frames.LLMMarkerResponseFrame)
			return ok && x.Raw == "r" && x.Marker == "t" && x.Kind == "k" && len(x.Markers) == 2
		}},
		{"bot-tts-text", map[string]any{"text": "spoken"}, func(f frames.Frame) bool { x, ok := f.(*frames.TTSTextFrame); return ok && x.Text == "spoken" }},
		{"bot-started-speaking", nil, func(f frames.Frame) bool { _, ok := f.(*frames.BotStartedSpeakingFrame); return ok }},
		{"bot-stopped-speaking", nil, func(f frames.Frame) bool { _, ok := f.(*frames.BotStoppedSpeakingFrame); return ok }},
		{"bot-interrupted", nil, func(f frames.Frame) bool { _, ok := f.(*frames.InterruptionFrame); return ok }},
		{"llm-function-call-in-progress", map[string]any{"tool_call_id": "c1", "function_name": "fn", "arguments": map[string]any{"x": 1.0}}, func(f frames.Frame) bool {
			x, ok := f.(*frames.FunctionCallInProgressFrame)
			return ok && x.ToolCallID == "c1" && x.FunctionName == "fn" && x.Arguments["x"] == 1.0
		}},
		{"llm-function-call-stopped", map[string]any{"tool_call_id": "c1", "function_name": "fn", "result": "r"}, func(f frames.Frame) bool {
			x, ok := f.(*frames.FunctionCallResultFrame)
			return ok && x.ToolCallID == "c1" && x.Result == "r"
		}},
		{"llm-function-call-stopped", map[string]any{"tool_call_id": "c2", "cancelled": true}, func(f frames.Frame) bool {
			x, ok := f.(*frames.FunctionCallCancelFrame)
			return ok && x.ToolCallID == "c2"
		}},
		{"user-transcription", map[string]any{"text": "hi", "user_id": "u", "final": true, "timestamp": "2026-01-01T00:00:00Z"}, func(f frames.Frame) bool {
			x, ok := f.(*frames.TranscriptionFrame)
			return ok && x.Text == "hi" && x.Metadata()["user_id"] == "u" && x.Timestamp.Year() == 2026
		}},
		{"user-transcription", map[string]any{"text": "h", "user_id": "u", "final": false}, func(f frames.Frame) bool {
			x, ok := f.(*frames.InterimTranscriptionFrame)
			return ok && x.Text == "h" && x.UserID == "u"
		}},
	}
	for _, c := range checks {
		if f := rtviIn(t, s, c.typ, c.data); !c.ok(f) {
			t.Fatalf("%s %v -> %#v", c.typ, c.data, f)
		}
	}
	if f := rtviIn(t, s, "unknown", nil); f != nil {
		t.Fatalf("unknown type -> %#v", f)
	}
	f, err := s.Deserialize(Text(`{"label":"other","type":"bot-llm-text"}`))
	if f != nil || err != nil {
		t.Fatal("non-rtvi label accepted")
	}
	if _, err := s.Deserialize(Text(`nope`)); err == nil {
		t.Fatal("garbage accepted")
	}
}
