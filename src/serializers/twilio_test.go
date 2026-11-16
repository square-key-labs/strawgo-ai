package serializers

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
)

func twilioStarted(t *testing.T, cfg TwilioConfig) *TwilioFrameSerializer {
	t.Helper()
	s := NewTwilioFrameSerializer(cfg)
	done, err := s.Handshake(textMsg(t, map[string]any{
		"event":     "start",
		"streamSid": "MZ1",
		"start": map[string]any{
			"streamSid":        "MZ1",
			"callSid":          "CA1",
			"accountSid":       "AC1",
			"customParameters": map[string]string{"lang": "en"},
			"mediaFormat":      map[string]any{"encoding": "audio/x-mulaw", "sampleRate": 8000},
		},
	}))
	if err != nil || !done {
		t.Fatalf("handshake = %v, %v", done, err)
	}
	return s
}

func TestTwilioHandshakeIgnoresNonStartEvents(t *testing.T) {
	s := NewTwilioFrameSerializer(TwilioConfig{})
	done, err := s.Handshake(Text(`{"event":"connected","protocol":"Call"}`))
	if err != nil || done {
		t.Fatalf("handshake(connected) = %v, %v; want false, nil", done, err)
	}
	if _, err := s.Handshake(Text(`not json`)); err == nil {
		t.Fatal("handshake(garbage) returned nil error")
	}
}

func TestTwilioHandshakeParsesCallInfo(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	info := s.CallInfo()
	if info.Provider != "twilio" || info.StreamID != "MZ1" || info.CallID != "CA1" || info.AccountID != "AC1" {
		t.Fatalf("call info = %+v", info)
	}
	if info.Encoding != audio.CodecMulaw || info.SampleRate != 8000 {
		t.Fatalf("media format = %s/%d", info.Encoding, info.SampleRate)
	}
	if info.CustomParameters["lang"] != "en" {
		t.Fatalf("custom params = %v", info.CustomParameters)
	}
	if in, out := s.WireSampleRates(); in != 8000 || out != 8000 {
		t.Fatalf("wire rates = %d/%d", in, out)
	}
}

func TestTwilioInboundMediaDecodesToPCM(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	pcm := tonePCM(160, 8000, 440, 8000)
	ulaw := mustEncode(t, pcm, audio.CodecMulaw)
	f, err := s.Deserialize(textMsg(t, map[string]any{"event": "media", "streamSid": "MZ1", "media": map[string]any{"payload": b64(ulaw)}}))
	af := audioFrame(t, f, err)
	if af.SampleRate != 8000 || af.Channels != 1 {
		t.Fatalf("audio frame %d Hz %d ch", af.SampleRate, af.Channels)
	}
	if af.Metadata()["codec"] != audio.CodecLinear16 {
		t.Fatalf("codec = %v", af.Metadata()["codec"])
	}
	assertClosePCM(t, af.Data, pcm, 300)
}

func TestTwilioNativeInputPassesMulawThrough(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	s.SetNativeAudioInput(true)
	ulaw := []byte{0xff, 0x7f, 0x00, 0x80}
	f, err := s.Deserialize(textMsg(t, map[string]any{"event": "media", "media": map[string]any{"payload": b64(ulaw)}}))
	af := audioFrame(t, f, err)
	if string(af.Data) != string(ulaw) || af.Metadata()["codec"] != audio.CodecMulaw {
		t.Fatalf("native frame data=%v codec=%v", af.Data, af.Metadata()["codec"])
	}
	codec, rate := s.NativeAudioFormat()
	if codec != audio.CodecMulaw || rate != 8000 {
		t.Fatalf("native format = %s/%d", codec, rate)
	}
	msg := one(t)(s.SerializeNativeAudio(nil, ulaw))
	m := jsonMsg(t, msg)
	if path(m, "media", "payload") != b64(ulaw) {
		t.Fatalf("native out payload = %v", path(m, "media", "payload"))
	}
}

func TestTwilioEmptyAndBadMediaPayloads(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	f, err := s.Deserialize(Text(`{"event":"media","media":{"payload":""}}`))
	if f != nil || err != nil {
		t.Fatalf("empty payload = %v, %v", f, err)
	}
	if _, err := s.Deserialize(Text(`{"event":"media","media":{"payload":"***"}}`)); err == nil {
		t.Fatal("bad base64 accepted")
	}
}

func TestTwilioOutboundAudioAtWireRate(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	pcm := tonePCM(160, 8000, 440, 8000)
	msg := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(pcm, 8000, 1)))
	m := jsonMsg(t, msg)
	if m["event"] != "media" || m["streamSid"] != "MZ1" {
		t.Fatalf("media envelope = %v", m)
	}
	raw, err := base64.StdEncoding.DecodeString(path(m, "media", "payload").(string))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != 160 {
		t.Fatalf("payload bytes = %d, want 160", len(raw))
	}
	back, _ := audio.DecodePCM16(nil, raw, audio.CodecMulaw)
	assertClosePCM(t, back, pcm, 300)
}

func TestTwilioOutboundAudioResamplesTo8k(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	total := 0
	for i := 0; i < 20; i++ {
		msg := one(t)(s.Serialize(nil, frames.NewTTSAudioFrame(tonePCM(320, 16000, 300, 8000), 16000, 1)))
		raw, err := base64.StdEncoding.DecodeString(path(jsonMsg(t, msg), "media", "payload").(string))
		if err != nil {
			t.Fatal(err)
		}
		total += len(raw)
	}
	want := 20 * 160
	if total < want-80 || total > want+8 {
		t.Fatalf("resampled payload bytes = %d, want ~%d", total, want)
	}
}

func TestTwilioOutboundAudioFrameWithMulawCodec(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	ulaw := mustEncode(t, tonePCM(160, 8000, 440, 8000), audio.CodecMulaw)
	f := frames.NewAudioFrame(ulaw, 8000, 1)
	f.SetMetadata("codec", audio.CodecMulaw)
	msg := one(t)(s.Serialize(nil, f))
	if path(jsonMsg(t, msg), "media", "payload") != b64(ulaw) {
		t.Fatal("mulaw frame was not re-encoded losslessly")
	}
}

func TestTwilioInterruptionSendsClear(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewInterruptionFrame())))
	if m["event"] != "clear" || m["streamSid"] != "MZ1" || len(m) != 2 {
		t.Fatalf("clear = %v", m)
	}
}

func TestTwilioDTMFMarkStop(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	f, err := s.Deserialize(Text(`{"event":"dtmf","streamSid":"MZ1","dtmf":{"track":"inbound_track","digit":"#"}}`))
	assertDTMF(t, f, err, frames.KeypadPound)
	f, err = s.Deserialize(Text(`{"event":"dtmf","dtmf":{"digit":"x"}}`))
	if f != nil || err != nil {
		t.Fatalf("invalid digit = %v, %v", f, err)
	}
	f, err = s.Deserialize(Text(`{"event":"stop","streamSid":"MZ1"}`))
	assertType[*frames.EndFrame](t, f, err)
	f, err = s.Deserialize(Text(`{"event":"unknown"}`))
	if f != nil || err != nil {
		t.Fatalf("unknown event = %v, %v", f, err)
	}
}

func TestTwilioTransportMessagesSkipRTVI(t *testing.T) {
	s := twilioStarted(t, TwilioConfig{})
	m := jsonMsg(t, one(t)(s.Serialize(nil, frames.NewOutputTransportMessageFrame(map[string]any{"hello": "world"}))))
	if m["hello"] != "world" {
		t.Fatalf("passthrough = %v", m)
	}
	none(t)(s.Serialize(nil, frames.NewOutputTransportMessageUrgentFrame(map[string]any{"label": "rtvi-ai", "type": "bot-ready"})))
	msg := one(t)(s.Serialize(nil, frames.NewOutputTransportMessageFrame("raw text")))
	if string(msg.Data) != "raw text" {
		t.Fatalf("string passthrough = %q", msg.Data)
	}
	none(t)(s.Serialize(nil, frames.NewTextFrame("ignored")))
}

func TestTwilioHangupWithAuthToken(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: rec.srv.URL + "/"})
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	r := rec.only(t)
	if r.Method != http.MethodPost || r.Path != "/2010-04-01/Accounts/AC1/Calls/CA1.json" {
		t.Fatalf("request = %s %s", r.Method, r.Path)
	}
	if !r.Basic || r.User != "AC1" || r.Pass != "secret" {
		t.Fatalf("basic auth = %v %q %q", r.Basic, r.User, r.Pass)
	}
	if r.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
		t.Fatalf("content-type = %q", r.Header.Get("Content-Type"))
	}
	vals, _ := url.ParseQuery(r.Body)
	if vals.Get("Status") != "completed" {
		t.Fatalf("body = %q", r.Body)
	}
}

func TestTwilioHangupWithAPIKey(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := twilioStarted(t, TwilioConfig{APIKeySID: "SK1", APIKeySecret: "ks", BaseURL: rec.srv.URL})
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	r := rec.only(t)
	if r.User != "SK1" || r.Pass != "ks" {
		t.Fatalf("api key auth = %q %q", r.User, r.Pass)
	}
}

func TestTwilioHangupAlreadyEnded(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusNotFound, `{"code":20404,"message":"not found"}`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: rec.srv.URL})
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatalf("already-ended hangup returned %v", err)
	}
}

func TestTwilioHangupServerError(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusInternalServerError, `boom`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: rec.srv.URL})
	if err := s.Hangup(context.Background()); err == nil {
		t.Fatal("hangup error not reported")
	}
}

func TestTwilioHangupSkipped(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	for _, cfg := range []TwilioConfig{
		{AuthToken: "secret", BaseURL: rec.srv.URL, DisableAutoHangUp: true},
		{BaseURL: rec.srv.URL},
	} {
		s := twilioStarted(t, cfg)
		if err := s.Hangup(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if n := len(rec.requests()); n != 0 {
		t.Fatalf("got %d requests, want 0", n)
	}
}

func TestTwilioHangupHonoursContext(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: rec.srv.URL})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.Hangup(ctx); err == nil {
		t.Fatal("cancelled hangup returned nil")
	}
}

func TestValidateTransferRequest(t *testing.T) {
	for _, ok := range []TransferRequest{
		{Destination: "+14155550100"},
		{Destination: "sip:agent@pbx.example.com"},
		{Destination: "sips:queue-1@pbx.example.com:5061;transport=tls", CallerID: "+442071838750"},
	} {
		if err := ValidateTransferRequest(ok); err != nil {
			t.Fatalf("%+v rejected: %v", ok, err)
		}
	}
	for _, bad := range []TransferRequest{
		{},
		{Destination: "14155550100"},
		{Destination: "+1415"},
		{Destination: "+14155550100</Number><Number>+19005550100"},
		{Destination: "tel:+14155550100"},
		{Destination: "sip:pbx.example.com"},
		{Destination: "sip:a@b\"><Play>x"},
		{Destination: "+14155550100", CallerID: "anonymous"},
	} {
		if err := ValidateTransferRequest(bad); err == nil {
			t.Fatalf("%+v accepted", bad)
		}
	}
}

func TestTwilioTransferUpdatesCallWithDialTwiML(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: rec.srv.URL})
	if !s.CanTransfer() {
		t.Fatal("CanTransfer = false")
	}
	err := s.Transfer(context.Background(), TransferRequest{Destination: "+14155550100", CallerID: "+14155550199", Announcement: "Connecting you <now> & thanks"})
	if err != nil {
		t.Fatal(err)
	}
	r := rec.only(t)
	if r.Method != http.MethodPost || r.Path != "/2010-04-01/Accounts/AC1/Calls/CA1.json" || r.User != "AC1" || r.Pass != "secret" {
		t.Fatalf("request = %s %s %q", r.Method, r.Path, r.User)
	}
	vals, _ := url.ParseQuery(r.Body)
	want := `<Response><Say>Connecting you &lt;now&gt; &amp; thanks</Say><Dial callerId="+14155550199"><Number>+14155550100</Number></Dial></Response>`
	if vals.Get("Twiml") != want || vals.Get("Status") != "" {
		t.Fatalf("body = %q", r.Body)
	}
	if err := s.Hangup(context.Background()); err != nil {
		t.Fatal(err)
	}
	if n := len(rec.requests()); n != 1 {
		t.Fatalf("hang-up after transfer sent a request: %d requests", n)
	}
}

func TestTwilioTransferToSIP(t *testing.T) {
	rec := newAPIRecorder(t, http.StatusOK, `{}`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: rec.srv.URL})
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "sip:agent@pbx.example.com"}); err != nil {
		t.Fatal(err)
	}
	vals, _ := url.ParseQuery(rec.only(t).Body)
	if vals.Get("Twiml") != `<Response><Dial><Sip>sip:agent@pbx.example.com</Sip></Dial></Response>` {
		t.Fatalf("twiml = %q", vals.Get("Twiml"))
	}
}

func TestTwilioTransferFailures(t *testing.T) {
	bad := newAPIRecorder(t, http.StatusBadRequest, `{"code":21220,"message":"invalid"}`)
	s := twilioStarted(t, TwilioConfig{AuthToken: "sekrit-token", BaseURL: bad.srv.URL})
	err := s.Transfer(context.Background(), TransferRequest{Destination: "+14155550100"})
	if err == nil || strings.Contains(err.Error(), "sekrit-token") {
		t.Fatalf("err = %v", err)
	}
	_ = s.Hangup(context.Background())
	if n := len(bad.requests()); n != 2 {
		t.Fatalf("failed transfer must leave hang-up armed: %d requests", n)
	}

	gone := newAPIRecorder(t, http.StatusNotFound, `{"code":20404}`)
	s = twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: gone.srv.URL})
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "+14155550100"}); !errors.Is(err, ErrCallEnded) {
		t.Fatalf("ended call err = %v", err)
	}

	none := newAPIRecorder(t, http.StatusOK, `{}`)
	s = twilioStarted(t, TwilioConfig{BaseURL: none.srv.URL})
	if s.CanTransfer() {
		t.Fatal("CanTransfer without credentials")
	}
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "+14155550100"}); !errors.Is(err, ErrTransferUnsupported) {
		t.Fatalf("no credentials err = %v", err)
	}
	s = twilioStarted(t, TwilioConfig{AuthToken: "secret", BaseURL: none.srv.URL})
	if err := s.Transfer(context.Background(), TransferRequest{Destination: "0800 CALL ME"}); !errors.Is(err, ErrInvalidDestination) {
		t.Fatalf("invalid destination err = %v", err)
	}
	if n := len(none.requests()); n != 0 {
		t.Fatalf("rejected transfers made %d requests", n)
	}
}
