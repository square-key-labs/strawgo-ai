package serializers

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func genesysReply(t *testing.T, f frames.Frame, err error) map[string]any {
	t.Helper()
	u := assertType[*frames.OutputTransportMessageUrgentFrame](t, f, err)
	b, err := json.Marshal(u.Message)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

func genesysOpen(t *testing.T, cfg GenesysConfig, channels []string) (*GenesysAudioHookSerializer, map[string]any) {
	t.Helper()
	s := NewGenesysAudioHookSerializer(cfg)
	f, err := s.Deserialize(textMsg(t, map[string]any{
		"version": "2", "type": "open", "seq": 1, "id": "sess-1", "position": "PT0S",
		"parameters": map[string]any{
			"conversationId": "conv-1",
			"participant":    map[string]any{"ani": "+1555", "dnis": "+1666"},
			"inputVariables": map[string]any{"foo": "bar"},
			"media":          []any{map[string]any{"type": "audio", "format": "PCMU", "channels": channels, "rate": 8000}},
		},
	}))
	return s, genesysReply(t, f, err)
}

func TestGenesysOpenHandshake(t *testing.T) {
	var opened map[string]any
	s, reply := genesysOpen(t, GenesysConfig{SupportedLanguages: []string{"en-US"}, OnOpen: func(m map[string]any) { opened = m }}, []string{"external"})
	if opened == nil {
		t.Fatal("OnOpen not invoked")
	}
	if reply["type"] != "opened" || reply["id"] != "sess-1" || reply["version"] != GenesysProtocolVersion || reply["clientseq"] != float64(1) || reply["seq"] != float64(1) {
		t.Fatalf("opened = %v", reply)
	}
	media := path(reply, "parameters", "media").([]any)[0].(map[string]any)
	if media["format"] != "PCMU" || media["rate"] != float64(8000) {
		t.Fatalf("opened media = %v", media)
	}
	if !s.IsOpen() || s.IsPaused() || s.ConversationID() != "conv-1" || s.InputVariables()["foo"] != "bar" {
		t.Fatal("open state not recorded")
	}
	info := s.CallInfo()
	if info.From != "+1555" || info.To != "+1666" || info.CallID != "conv-1" || info.StreamID != "sess-1" {
		t.Fatalf("info = %+v", info)
	}
	f, err := s.Deserialize(Text(`{"version":"2","type":"ping","seq":2,"id":"sess-1","position":"PT1.5S"}`))
	pong := genesysReply(t, f, err)
	if pong["type"] != "pong" || pong["seq"] != float64(2) || pong["clientseq"] != float64(2) || pong["position"] != "PT1.500S" {
		t.Fatalf("pong = %v", pong)
	}
}

func TestGenesysAudioGatedByOpenAndPause(t *testing.T) {
	s := NewGenesysAudioHookSerializer(GenesysConfig{})
	pcm := tonePCM(160, 8000, 440, 8000)
	ulaw := mustEncode(t, pcm, audio.CodecMulaw)
	f, err := s.Deserialize(Binary(append([]byte{}, ulaw...)))
	if f != nil || err != nil {
		t.Fatal("audio accepted before open")
	}
	none(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1)))
	s, _ = genesysOpen(t, GenesysConfig{}, []string{"external"})
	f, err = s.Deserialize(Binary(append([]byte{}, ulaw...)))
	assertClosePCM(t, audioFrame(t, f, err).Data, pcm, 300)
	out := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1)))
	if out.Type != BinaryMessage || string(out.Data) != string(ulaw) {
		t.Fatal("outbound ulaw mismatch")
	}
	_, _ = s.Deserialize(Text(`{"type":"paused","seq":2}`))
	if !s.IsPaused() {
		t.Fatal("not paused")
	}
	none(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1)))
	if f, _ := s.Deserialize(Binary(append([]byte{}, ulaw...))); f != nil {
		t.Fatal("audio accepted while paused")
	}
	_, _ = s.Deserialize(Text(`{"type":"resumed","seq":3}`))
	one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1)))
}

func TestGenesysStereoTakesExternalChannel(t *testing.T) {
	s, _ := genesysOpen(t, GenesysConfig{}, []string{"external", "internal"})
	ext := mustEncode(t, tonePCM(80, 8000, 440, 8000), audio.CodecMulaw)
	inter := mustEncode(t, tonePCM(80, 8000, 1000, 2000), audio.CodecMulaw)
	stereo := make([]byte, 0, 160)
	for i := range ext {
		stereo = append(stereo, ext[i], inter[i])
	}
	f, err := s.Deserialize(Binary(stereo))
	want, _ := audio.DecodePCM16(nil, ext, audio.CodecMulaw)
	af := audioFrame(t, f, err)
	if string(af.Data) != string(want) {
		t.Fatal("stereo input did not keep the external channel")
	}
}

func TestGenesysControlMessages(t *testing.T) {
	var dtmf, closed bool
	s, _ := genesysOpen(t, GenesysConfig{OnDTMF: func(map[string]any) { dtmf = true }, OnClose: func(map[string]any) { closed = true }}, []string{"external"})
	f, err := s.Deserialize(Text(`{"type":"dtmf","seq":2,"parameters":{"digit":"4"}}`))
	assertDTMF(t, f, err, frames.KeypadFour)
	if !dtmf {
		t.Fatal("OnDTMF not invoked")
	}
	barge := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewInterruptionFrame())))
	ent := path(barge, "parameters", "entities").([]any)[0].(map[string]any)
	if barge["type"] != "event" || ent["type"] != "barge_in" {
		t.Fatalf("barge-in = %v", barge)
	}
	s.SetOutputVariables(map[string]any{"result": "ok"})
	disc := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewEndFrame())))
	if disc["type"] != "disconnect" || path(disc, "parameters", "reason") != "completed" || path(disc, "parameters", "outputVariables", "result") != "ok" || path(disc, "parameters", "outputVariables", "action") != "transfer" {
		t.Fatalf("disconnect = %v", disc)
	}
	f, err = s.Deserialize(Text(`{"type":"close","seq":5,"parameters":{"reason":"end"}}`))
	cl := genesysReply(t, f, err)
	if cl["type"] != "closed" || path(cl, "parameters", "outputVariables", "result") != "ok" || s.IsOpen() || !closed {
		t.Fatalf("closed = %v open=%v", cl, s.IsOpen())
	}
	f, err = s.Deserialize(Text(`{"type":"update","seq":6,"parameters":{"participant":{"ani":"+1999"}}}`))
	if f != nil || err != nil || s.CallInfo().From != "+1999" {
		t.Fatal("update not applied")
	}
}

func TestGenesysTransportPassthrough(t *testing.T) {
	s, _ := genesysOpen(t, GenesysConfig{}, []string{"external"})
	none(t)(s.Serialize(nil, frames.NewOutputTransportMessageFrame(map[string]any{"label": "rtvi-ai", "version": "1"})))
	none(t)(s.Serialize(nil, frames.NewOutputTransportMessageFrame(map[string]any{"foo": "bar"})))
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewOutputTransportMessageUrgentFrame(s.PauseRequest()))))
	if m["type"] != "pause" {
		t.Fatalf("pause = %v", m)
	}
	m = jsonMsg(t, one(t)(s.Serialize(nil, frames.NewOutputTransportMessageFrame(map[string]any{"version": "2", "type": "custom"}))))
	if m["type"] != "custom" {
		t.Fatalf("custom = %v", m)
	}
}

func jambonzReady(t *testing.T, cfg JambonzConfig, preamble string) *JambonzFrameSerializer {
	t.Helper()
	s := NewJambonzFrameSerializer(cfg)
	done, err := s.Handshake(Text(preamble))
	if err != nil || !done {
		t.Fatalf("handshake = %v %v", done, err)
	}
	return s
}

func TestJambonzPreambleAndAudio(t *testing.T) {
	s := jambonzReady(t, JambonzConfig{}, `{"callSid":"js-1","traceId":"tr-1","sampleRate":16000,"mixType":"mono","from":"a","to":"b"}`)
	info := s.CallInfo()
	if info.CallID != "js-1" || info.StreamID != "tr-1" || info.SampleRate != 16000 || info.From != "a" || info.To != "b" {
		t.Fatalf("info = %+v", info)
	}
	if in, out := s.WireSampleRates(); in != 16000 || out != 16000 {
		t.Fatalf("rates %d %d", in, out)
	}
	pcm := tonePCM(320, 16000, 440, 8000)
	f, err := s.Deserialize(Binary(pcm))
	if af := audioFrame(t, f, err); string(af.Data) != string(pcm) || af.SampleRate != 16000 {
		t.Fatal("inbound mismatch")
	}
	out := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 16000, 1)))
	if out.Type != BinaryMessage || string(out.Data) != string(pcm) {
		t.Fatal("outbound mismatch")
	}
	if string(one(t)(s.Serialize(nil, frames.NewInterruptionFrame())).Data) != `{"type":"killAudio"}` {
		t.Fatal("killAudio not sent")
	}
	if string(one(t)(s.Serialize(nil, frames.NewEndFrame())).Data) != `{"type":"disconnect"}` {
		t.Fatal("disconnect not sent")
	}
	none(t)(s.Serialize(nil, frames.NewCancelFrame()))
	f, err = s.Deserialize(Text(`{"event":"dtmf","dtmf":"6","duration":250}`))
	assertDTMF(t, f, err, frames.KeypadSix)
}

func TestJambonzConfiguredRatesAndDisableDisconnect(t *testing.T) {
	s := jambonzReady(t, JambonzConfig{AudioInSampleRate: 8000, AudioOutSampleRate: 24000, DisableDisconnectEnd: true}, `{"sampleRate":16000}`)
	if in, out := s.WireSampleRates(); in != 8000 || out != 24000 {
		t.Fatalf("rates %d %d", in, out)
	}
	none(t)(s.Serialize(nil, frames.NewEndFrame()))
}

func TestJambonzNonMonoMixDropsAudio(t *testing.T) {
	s := jambonzReady(t, JambonzConfig{}, `{"sampleRate":8000,"mixType":"stereo"}`)
	f, err := s.Deserialize(Binary(tonePCM(160, 8000, 440, 8000)))
	if f != nil || err != nil {
		t.Fatal("stereo mix audio was not dropped")
	}
}

func TestJambonzFirstTextMessageIsPreamble(t *testing.T) {
	s := NewJambonzFrameSerializer(JambonzConfig{})
	if f, _ := s.Deserialize(Binary(tonePCM(10, 8000, 440, 8000))); f != nil {
		t.Fatal("audio accepted with unknown rate")
	}
	none(t)(s.SerializeAudio(nil, tonePCM(10, 8000, 440, 8000), 8000))
	f, err := s.Deserialize(Text(`{"callSid":"late","sampleRate":8000}`))
	if f != nil || err != nil || s.CallSID() != "late" {
		t.Fatalf("late preamble = %v %v %q", f, err, s.CallSID())
	}
	f, err = s.Deserialize(Binary(tonePCM(10, 8000, 440, 8000)))
	audioFrame(t, f, err)
}

func TestGenesysTransferSetsOutputVariableForDisconnect(t *testing.T) {
	s := NewGenesysAudioHookSerializer(GenesysConfig{})
	if s.CanTransfer() {
		t.Fatal("CanTransfer before open")
	}
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "+14155550100"}); !errors.Is(err, ErrCallEnded) {
		t.Fatalf("closed session err = %v", err)
	}
	s, _ = genesysOpen(t, GenesysConfig{}, []string{"external"})
	s.SetOutputVariables(map[string]any{"intent": "billing"})
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "not a number"}); !errors.Is(err, ErrInvalidDestination) {
		t.Fatalf("invalid destination err = %v", err)
	}
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "+14155550100"}); err != nil {
		t.Fatal(err)
	}
	disc := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewEndFrame())))
	vars := path(disc, "parameters", "outputVariables")
	if disc["type"] != "disconnect" || path(disc, "parameters", "reason") != "completed" {
		t.Fatalf("disconnect = %v", disc)
	}
	if path(disc, "parameters", "outputVariables", "action") != "transfer" || path(disc, "parameters", "outputVariables", GenesysTransferVariable) != "+14155550100" || path(disc, "parameters", "outputVariables", "intent") != "billing" {
		t.Fatalf("outputVariables = %v", vars)
	}
}
