package serializers

import (
	"encoding/base64"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func TestExotelHandshakeSampleRateStringOrNumber(t *testing.T) {
	for _, rate := range []any{"16000", 16000} {
		s := NewExotelFrameSerializer(ExotelConfig{})
		done, err := s.Handshake(textMsg(t, map[string]any{
			"event": "start",
			"start": map[string]any{
				"stream_sid": "ES1", "call_sid": "EC1", "account_sid": "EA1", "from": "a", "to": "b",
				"custom_parameters": map[string]string{"k": "v"},
				"media_format":      map[string]any{"sample_rate": rate},
			},
		}))
		if err != nil || !done {
			t.Fatalf("handshake = %v %v", done, err)
		}
		info := s.CallInfo()
		if info.SampleRate != 16000 || info.StreamID != "ES1" || info.CallID != "EC1" || info.AccountID != "EA1" || info.From != "a" || info.To != "b" || info.CustomParameters["k"] != "v" {
			t.Fatalf("rate %v: call info = %+v", rate, info)
		}
		if in, out := s.WireSampleRates(); in != 16000 || out != 16000 {
			t.Fatalf("wire rates = %d %d", in, out)
		}
	}
}

func TestExotelConfiguredRateWins(t *testing.T) {
	s := NewExotelFrameSerializer(ExotelConfig{SampleRate: 8000})
	_, _ = s.Handshake(Text(`{"event":"start","start":{"stream_sid":"ES1","media_format":{"sample_rate":"16000"}}}`))
	if s.CallInfo().SampleRate != 8000 {
		t.Fatalf("rate = %d", s.CallInfo().SampleRate)
	}
}

func TestExotelMediaRoundTrip(t *testing.T) {
	s := NewExotelFrameSerializer(ExotelConfig{})
	_, _ = s.Handshake(Text(`{"event":"start","start":{"stream_sid":"ES1","call_sid":"EC1"}}`))
	pcm := tonePCM(160, 8000, 440, 8000)
	f, err := s.Deserialize(textMsg(t, map[string]any{"event": "media", "media": map[string]any{"payload": b64(append(append([]byte{}, pcm...), 0x01))}}))
	af := audioFrame(t, f, err)
	if string(af.Data) != string(pcm) || af.SampleRate != 8000 {
		t.Fatalf("inbound pcm mismatch (len %d rate %d)", len(af.Data), af.SampleRate)
	}
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1))))
	if m["event"] != "media" || m["stream_sid"] != "ES1" || path(m, "media", "payload") != b64(pcm) {
		t.Fatalf("outbound = %v", m)
	}
}

func TestExotelControl(t *testing.T) {
	s := NewExotelFrameSerializer(ExotelConfig{StreamSID: "ES1"})
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewInterruptionFrame())))
	if m["event"] != "clear" || m["stream_sid"] != "ES1" {
		t.Fatalf("clear = %v", m)
	}
	f, err := s.Deserialize(Text(`{"event":"dtmf","dtmf":{"digit":"0"}}`))
	assertDTMF(t, f, err, frames.KeypadZero)
	f, err = s.Deserialize(Text(`{"event":"mark","mark":{"name":"m"}}`))
	if assertType[*frames.PlaybackCompleteFrame](t, f, err).Metadata()["correlation_id"] != "m" {
		t.Fatal("mark id lost")
	}
	f, err = s.Deserialize(Text(`{"event":"stop"}`))
	assertType[*frames.EndFrame](t, f, err)
	ack, _ := s.SerializePlaybackAck("x")
	if path(jsonMsg(t, ack), "mark", "name") != "x" {
		t.Fatalf("ack = %s", ack.Data)
	}
}

func TestVonageHandshake(t *testing.T) {
	s := NewVonageFrameSerializer(VonageConfig{})
	done, err := s.Handshake(Text(`{"event":"websocket:connected","content-type":"audio/l16;rate=8000","call_uuid":"cu-1","caller":"bob"}`))
	if err != nil || !done {
		t.Fatalf("handshake = %v %v", done, err)
	}
	info := s.CallInfo()
	if info.SampleRate != 8000 || info.CallID != "cu-1" || info.CustomParameters["caller"] != "bob" {
		t.Fatalf("info = %+v", info)
	}
	fixed := NewVonageFrameSerializer(VonageConfig{SampleRate: 16000})
	_, _ = fixed.Handshake(Text(`{"event":"websocket:connected","content-type":"audio/l16;rate=8000"}`))
	if fixed.CallInfo().SampleRate != 16000 {
		t.Fatalf("fixed rate overwritten: %d", fixed.CallInfo().SampleRate)
	}
	other := NewVonageFrameSerializer(VonageConfig{})
	if done, _ := other.Handshake(Text(`{"event":"other"}`)); done {
		t.Fatal("non-connected event finished handshake")
	}
	if done, _ := other.Handshake(Binary([]byte{0, 0})); !done {
		t.Fatal("binary audio did not finish handshake")
	}
}

func TestVonageAudioAndControl(t *testing.T) {
	s := NewVonageFrameSerializer(VonageConfig{})
	pcm := tonePCM(320, 16000, 440, 8000)
	f, err := s.Deserialize(Binary(append(append([]byte{}, pcm...), 7)))
	af := audioFrame(t, f, err)
	if string(af.Data) != string(pcm) || af.SampleRate != 16000 {
		t.Fatal("inbound l16 mismatch")
	}
	out := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 16000, 1)))
	if out.Type != BinaryMessage || string(out.Data) != string(pcm) {
		t.Fatal("outbound l16 mismatch")
	}
	clr := one(t)(s.Serialize(nil, frames.NewInterruptionFrame()))
	if string(clr.Data) != `{"action":"clear"}` {
		t.Fatalf("clear = %s", clr.Data)
	}
	f, err = s.Deserialize(Text(`{"event":"websocket:dtmf","digit":5}`))
	assertDTMF(t, f, err, frames.KeypadFive)
	f, err = s.Deserialize(Text(`{"event":"websocket:dtmf","dtmf":{"digit":"9"}}`))
	assertDTMF(t, f, err, frames.KeypadNine)
	f, err = s.Deserialize(Text(`{"event":"websocket:notify","payload":{"name":"n1"}}`))
	if assertType[*frames.PlaybackCompleteFrame](t, f, err).Metadata()["correlation_id"] != "n1" {
		t.Fatal("notify id lost")
	}
	ack, _ := s.SerializePlaybackAck("n2")
	am := jsonMsg(t, ack)
	if am["action"] != "notify" || path(am, "payload", "name") != "n2" {
		t.Fatalf("ack = %v", am)
	}
}

func TestAsteriskTextHandshakeAndAudio(t *testing.T) {
	s := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})
	done, err := s.Handshake(Text("MEDIA_START connection_id:conn-1 channel:PJSIP/100-01 format:ulaw optimal_frame_size:160"))
	if err != nil || !done {
		t.Fatalf("handshake = %v %v", done, err)
	}
	info := s.CallInfo()
	if info.StreamID != "conn-1" || info.CallID != "PJSIP/100-01" || info.Encoding != audio.CodecMulaw || info.SampleRate != 8000 {
		t.Fatalf("info = %+v", info)
	}
	pcm := tonePCM(160, 8000, 440, 8000)
	ulaw := mustEncode(t, pcm, audio.CodecMulaw)
	f, err := s.Deserialize(Binary(ulaw))
	assertClosePCM(t, audioFrame(t, f, err).Data, pcm, 300)
	out := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1)))
	if out.Type != BinaryMessage || string(out.Data) != string(ulaw) {
		t.Fatal("outbound ulaw mismatch")
	}
	if string(one(t)(s.Serialize(nil, frames.NewInterruptionFrame())).Data) != "FLUSH_MEDIA" {
		t.Fatal("flush not sent")
	}
	if string(one(t)(s.Serialize(nil, frames.NewEndFrame())).Data) != "HANGUP" {
		t.Fatal("hangup not sent on EndFrame")
	}
	if string(one(t)(s.Serialize(nil, frames.NewCancelFrame())).Data) != "HANGUP" {
		t.Fatal("hangup not sent on CancelFrame")
	}
}

func TestAsteriskSlinFormats(t *testing.T) {
	cases := map[string]int{"slin": 8000, "slin16": 16000, "slin44": 44100, "slin48": 48000}
	for format, rate := range cases {
		s := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})
		_, _ = s.Handshake(Text("MEDIA_START connection_id:c channel:ch format:" + format))
		if s.Codec() != audio.CodecLinear16 || s.SampleRate() != rate {
			t.Fatalf("%s: codec=%s rate=%d, want linear16/%d", format, s.Codec(), s.SampleRate(), rate)
		}
		pcm := tonePCM(rate/50, rate, 440, 8000)
		f, err := s.Deserialize(Binary(pcm))
		af := audioFrame(t, f, err)
		if af.SampleRate != rate || string(af.Data) != string(pcm) {
			t.Fatalf("%s: inbound mismatch", format)
		}
	}
}

func TestAsteriskJSONModeCommands(t *testing.T) {
	s := NewAsteriskFrameSerializer(AsteriskSerializerConfig{})
	_, err := s.Handshake(Text(`{"event":"MEDIA_START","connection_id":"c1","channel":"ch1","format":"alaw","optimal_frame_size":160}`))
	if err != nil {
		t.Fatal(err)
	}
	if s.Codec() != audio.CodecAlaw || s.ChannelID() != "ch1" {
		t.Fatalf("codec %s channel %s", s.Codec(), s.ChannelID())
	}
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewInterruptionFrame())))
	if m["command"] != "FLUSH_MEDIA" {
		t.Fatalf("json flush = %v", m)
	}
	ack, _ := s.SerializePlaybackAck("")
	am := jsonMsg(t, ack)
	if am["command"] != "MARK_MEDIA" || am["correlation_id"] != "playback-done" {
		t.Fatalf("json ack = %v", am)
	}
	f, err := s.Deserialize(Text(`{"event":"DTMF_END","digit":3}`))
	assertDTMF(t, f, err, frames.KeypadThree)
}

func TestAsteriskControlEvents(t *testing.T) {
	s := NewAsteriskFrameSerializer(AsteriskSerializerConfig{DisableAutoHangUp: true})
	none(t)(s.Serialize(nil, frames.NewEndFrame()))
	f, err := s.Deserialize(Text("DTMF_END digit:8"))
	assertDTMF(t, f, err, frames.KeypadEight)
	f, err = s.Deserialize(Text("MEDIA_XOFF"))
	if !assertType[*MediaFlowFrame](t, f, err).Paused {
		t.Fatal("XOFF did not pause")
	}
	f, err = s.Deserialize(Text("MEDIA_XON"))
	if assertType[*MediaFlowFrame](t, f, err).Paused {
		t.Fatal("XON did not resume")
	}
	f, err = s.Deserialize(Text("HANGUP"))
	assertType[*frames.EndFrame](t, f, err)
	if _, err := s.Deserialize(Text("   ")); err == nil {
		t.Fatal("empty control accepted")
	}
}

func TestAsteriskConfiguredCodecWins(t *testing.T) {
	s := NewAsteriskFrameSerializer(AsteriskSerializerConfig{Codec: "slin16"})
	_, _ = s.Handshake(Text("MEDIA_START connection_id:c channel:ch format:ulaw"))
	if s.Codec() != audio.CodecLinear16 || s.SampleRate() != 16000 {
		t.Fatalf("codec %s rate %d", s.Codec(), s.SampleRate())
	}
	pcm := tonePCM(320, 16000, 440, 8000)
	out := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 16000, 1)))
	if string(out.Data) != string(pcm) {
		t.Fatal("slin16 out mismatch")
	}
	native := one(t)(s.SerializeNativeAudio(nil, []byte{1, 2}))
	if native.Type != BinaryMessage || base64.StdEncoding.EncodeToString(native.Data) != b64([]byte{1, 2}) {
		t.Fatal("native passthrough mismatch")
	}
}
