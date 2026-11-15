package serializers

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func telnyxStarted(t *testing.T, cfg TelnyxConfig, encoding string) *TelnyxFrameSerializer {
	t.Helper()
	s := NewTelnyxFrameSerializer(cfg)
	done, err := s.Handshake(textMsg(t, map[string]any{
		"event":     "start",
		"stream_id": "st-1",
		"start": map[string]any{
			"call_control_id": "v3:cc-1",
			"from":            "+15550001",
			"to":              "+15550002",
			"client_state":    "abc",
			"media_format":    map[string]any{"encoding": encoding, "sample_rate": 8000},
		},
	}))
	if err != nil || !done {
		t.Fatalf("handshake = %v, %v", done, err)
	}
	return s
}

func TestTelnyxHandshakeCallInfo(t *testing.T) {
	s := telnyxStarted(t, TelnyxConfig{}, "PCMU")
	info := s.CallInfo()
	if info.Provider != "telnyx" || info.StreamID != "st-1" || info.CallID != "v3:cc-1" || info.From != "+15550001" || info.To != "+15550002" {
		t.Fatalf("call info = %+v", info)
	}
	if info.CustomParameters["client_state"] != "abc" || info.Encoding != audio.CodecMulaw {
		t.Fatalf("call info = %+v", info)
	}
	done, err := s.Handshake(Text(`{"event":"connected"}`))
	if done || err != nil {
		t.Fatalf("non-start handshake = %v %v", done, err)
	}
}

func TestTelnyxAlawNegotiatedFromStart(t *testing.T) {
	s := telnyxStarted(t, TelnyxConfig{}, "PCMA")
	if codec, _ := s.NativeAudioFormat(); codec != audio.CodecAlaw {
		t.Fatalf("native codec = %s, want alaw", codec)
	}
	pcm := tonePCM(160, 8000, 500, 6000)
	alaw := mustEncode(t, pcm, audio.CodecAlaw)
	f, err := s.Deserialize(textMsg(t, map[string]any{"event": "media", "media": map[string]any{"payload": b64(alaw)}}))
	assertClosePCM(t, audioFrame(t, f, err).Data, pcm, 300)
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1))))
	raw, _ := base64.StdEncoding.DecodeString(path(m, "media", "payload").(string))
	if string(raw) != string(alaw) {
		t.Fatal("outbound audio not alaw encoded")
	}
}

func TestTelnyxConfiguredEncodingWins(t *testing.T) {
	s := telnyxStarted(t, TelnyxConfig{InboundEncoding: "PCMU"}, "PCMA")
	if s.CallInfo().Encoding != audio.CodecMulaw {
		t.Fatalf("encoding = %s, want mulaw", s.CallInfo().Encoding)
	}
}

func TestTelnyxMediaClearDTMFStopMark(t *testing.T) {
	s := telnyxStarted(t, TelnyxConfig{}, "PCMU")
	pcm := tonePCM(160, 8000, 440, 8000)
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1))))
	if m["event"] != "media" {
		t.Fatalf("media = %v", m)
	}
	raw, _ := base64.StdEncoding.DecodeString(path(m, "media", "payload").(string))
	if len(raw) != 160 {
		t.Fatalf("payload = %d bytes", len(raw))
	}
	clr := one(t)(s.Serialize(nil, frames.NewInterruptionFrame()))
	if string(clr.Data) != `{"event":"clear"}` {
		t.Fatalf("clear = %s", clr.Data)
	}
	f, err := s.Deserialize(Text(`{"event":"dtmf","dtmf":{"digit":"7"}}`))
	assertDTMF(t, f, err, frames.KeypadSeven)
	f, err = s.Deserialize(Text(`{"event":"stop"}`))
	assertType[*frames.EndFrame](t, f, err)
	f, err = s.Deserialize(Text(`{"event":"mark","mark":{"name":"m1"}}`))
	pc := assertType[*frames.PlaybackCompleteFrame](t, f, err)
	if pc.Metadata()["correlation_id"] != "m1" {
		t.Fatalf("mark id = %v", pc.Metadata()["correlation_id"])
	}
	ack, err := s.SerializePlaybackAck("m2")
	if err != nil || path(jsonMsg(t, ack), "mark", "name") != "m2" {
		t.Fatalf("ack = %s %v", ack.Data, err)
	}
}

func TestTelnyxHangup(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := telnyxStarted(t, TelnyxConfig{APIKey: "KEY", BaseURL: rec.srv.URL}, "PCMU")
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	r := rec.only(t)
	if r.Method != http.MethodPost || r.Path != "/v2/calls/v3:cc-1/actions/hangup" {
		t.Fatalf("request = %s %s", r.Method, r.Path)
	}
	if r.Header.Get("Authorization") != "Bearer KEY" || r.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("headers = %v", r.Header)
	}
}

func TestTelnyxHangupAlreadyEndedAndErrors(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusUnprocessableEntity, `{"errors":[{"code":"90018"}]}`)
	s := telnyxStarted(t, TelnyxConfig{APIKey: "KEY", BaseURL: rec.srv.URL}, "PCMU")
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatalf("already-ended = %v", err)
	}
	bad := newAPIRecorder(t, http.StatusUnauthorized, `nope`)
	s = telnyxStarted(t, TelnyxConfig{APIKey: "KEY", BaseURL: bad.srv.URL}, "PCMU")
	if err := s.Hangup(context.Background()); err == nil {
		t.Fatal("401 not reported")
	}
	skip := newAPIRecorder(t, http.StatusOK, `{}`)
	s = telnyxStarted(t, TelnyxConfig{BaseURL: skip.srv.URL}, "PCMU")
	_ = s.Hangup(context.Background())
	s = telnyxStarted(t, TelnyxConfig{APIKey: "KEY", BaseURL: skip.srv.URL, DisableAutoHangUp: true}, "PCMU")
	_ = s.Hangup(context.Background())
	if n := len(skip.requests()); n != 0 {
		t.Fatalf("skipped hangup made %d requests", n)
	}
}

func TestTelnyxSendDTMF(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := telnyxStarted(t, TelnyxConfig{APIKey: "KEY", BaseURL: rec.srv.URL}, "PCMU")
	if !s.CanSendDTMF() {
		t.Fatal("CanSendDTMF = false")
	}
	if err := s.SendDTMF(context.Background(), []frames.KeypadEntry{"1", "#"}); err != nil {
		t.Fatal(err)
	}
	r := rec.only(t)
	if r.Path != "/v2/calls/v3:cc-1/actions/send_dtmf" || r.Header.Get("Authorization") != "Bearer KEY" {
		t.Fatalf("request = %s %v", r.Path, r.Header)
	}
	var body map[string]any
	_ = json.Unmarshal([]byte(r.Body), &body)
	if body["digits"] != "1#" {
		t.Fatalf("body = %s", r.Body)
	}
}

func plivoStarted(t *testing.T, cfg PlivoConfig) *PlivoFrameSerializer {
	t.Helper()
	s := NewPlivoFrameSerializer(cfg)
	done, err := s.Handshake(textMsg(t, map[string]any{
		"event":         "start",
		"start":         map[string]any{"callId": "call-1", "streamId": "str-1", "accountId": "MA1"},
		"extra_headers": `{"x-foo":"bar"}`,
	}))
	if err != nil || !done {
		t.Fatalf("handshake = %v %v", done, err)
	}
	return s
}

func TestPlivoHandshakeCallInfo(t *testing.T) {
	s := plivoStarted(t, PlivoConfig{})
	info := s.CallInfo()
	if info.Provider != "plivo" || info.CallID != "call-1" || info.StreamID != "str-1" || info.AccountID != "MA1" || info.CustomParameters["x-foo"] != "bar" {
		t.Fatalf("call info = %+v", info)
	}
}

func TestPlivoMediaRoundTrip(t *testing.T) {
	s := plivoStarted(t, PlivoConfig{})
	pcm := tonePCM(160, 8000, 440, 8000)
	ulaw := mustEncode(t, pcm, audio.CodecMulaw)
	f, err := s.Deserialize(textMsg(t, map[string]any{"event": "media", "media": map[string]any{"payload": b64(ulaw)}}))
	assertClosePCM(t, audioFrame(t, f, err).Data, pcm, 300)
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1))))
	if m["event"] != "playAudio" || m["streamId"] != "str-1" || path(m, "media", "contentType") != "audio/x-mulaw" || path(m, "media", "sampleRate") != float64(8000) {
		t.Fatalf("playAudio = %v", m)
	}
	if path(m, "media", "payload") != b64(ulaw) {
		t.Fatal("payload mismatch")
	}
}

func TestPlivoControlMessages(t *testing.T) {
	s := plivoStarted(t, PlivoConfig{})
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewInterruptionFrame())))
	if m["event"] != "clearAudio" || m["streamId"] != "str-1" {
		t.Fatalf("clear = %v", m)
	}
	f, err := s.Deserialize(Text(`{"event":"dtmf","dtmf":{"digit":"*"}}`))
	assertDTMF(t, f, err, frames.KeypadStar)
	f, err = s.Deserialize(Text(`{"event":"playedStream","name":"cp1"}`))
	pc := assertType[*frames.PlaybackCompleteFrame](t, f, err)
	if pc.Metadata()["correlation_id"] != "cp1" {
		t.Fatalf("checkpoint id = %v", pc.Metadata()["correlation_id"])
	}
	f, err = s.Deserialize(Text(`{"event":"stop"}`))
	assertType[*frames.EndFrame](t, f, err)
	ack, _ := s.SerializePlaybackAck("cp2")
	am := jsonMsg(t, ack)
	if am["event"] != "checkpoint" || am["name"] != "cp2" || am["streamId"] != "str-1" {
		t.Fatalf("ack = %v", am)
	}
}

func TestPlivoHangup(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusNoContent, ``)
	s := plivoStarted(t, PlivoConfig{AuthID: "MA1", AuthToken: "tok", BaseURL: rec.srv.URL})
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	_ = s.Hangup(context.Background())
	r := rec.only(t)
	if r.Method != http.MethodDelete || r.Path != "/v1/Account/MA1/Call/call-1/" {
		t.Fatalf("request = %s %s", r.Method, r.Path)
	}
	if !r.Basic || r.User != "MA1" || r.Pass != "tok" {
		t.Fatalf("auth = %q %q", r.User, r.Pass)
	}
	gone := newAPIRecorder(t, http.StatusNotFound, ``)
	s = plivoStarted(t, PlivoConfig{AuthID: "MA1", AuthToken: "tok", BaseURL: gone.srv.URL})
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatalf("404 hangup = %v", err)
	}
	bad := newAPIRecorder(t, http.StatusInternalServerError, ``)
	s = plivoStarted(t, PlivoConfig{AuthID: "MA1", AuthToken: "tok", BaseURL: bad.srv.URL})
	if err := s.Hangup(context.Background()); err == nil {
		t.Fatal("500 hangup not reported")
	}
}

func TestPlivoSendDTMF(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusAccepted, `{}`)
	s := plivoStarted(t, PlivoConfig{AuthID: "MA1", AuthToken: "tok", BaseURL: rec.srv.URL})
	if err := s.SendDTMF(context.Background(), []frames.KeypadEntry{"4", "2"}); err != nil {
		t.Fatal(err)
	}
	r := rec.only(t)
	if r.Method != http.MethodPost || r.Path != "/v1/Account/MA1/Call/call-1/DTMF/" || r.User != "MA1" {
		t.Fatalf("request = %s %s %s", r.Method, r.Path, r.User)
	}
	var body map[string]string
	_ = json.Unmarshal([]byte(r.Body), &body)
	if body["digits"] != "42" {
		t.Fatalf("body = %s", r.Body)
	}
}
