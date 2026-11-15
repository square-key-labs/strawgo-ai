package serializers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type TwilioConfig struct {
	StreamSID         string
	CallSID           string
	AccountSID        string
	AuthToken         string
	APIKeySID         string
	APIKeySecret      string
	Region            string
	Edge              string
	BaseURL           string
	DisableAutoHangUp bool
	HTTPClient        *http.Client
}

type TwilioFrameSerializer struct {
	cfg        TwilioConfig
	info       CallInfo
	prefix     []byte
	wire       wireAudio
	hungUp     atomic.Bool
	ignoreRTVI bool
}

type twilioMessage struct {
	Event     string        `json:"event"`
	StreamSid string        `json:"streamSid,omitempty"`
	Media     *twilioMedia  `json:"media,omitempty"`
	Start     *twilioStart  `json:"start,omitempty"`
	Mark      *twilioMark   `json:"mark,omitempty"`
	DTMF      *twilioDigits `json:"dtmf,omitempty"`
}

type twilioMedia struct {
	Payload string `json:"payload"`
}

type twilioStart struct {
	StreamSid        string            `json:"streamSid"`
	CallSid          string            `json:"callSid"`
	AccountSid       string            `json:"accountSid"`
	CustomParameters map[string]string `json:"customParameters,omitempty"`
	MediaFormat      struct {
		Encoding   string `json:"encoding"`
		SampleRate int    `json:"sampleRate"`
	} `json:"mediaFormat"`
}

type twilioMark struct {
	Name string `json:"name"`
}

type twilioDigits struct {
	Digit string `json:"digit"`
}

const twilioSampleRate = 8000

func NewTwilioFrameSerializer(cfg TwilioConfig) *TwilioFrameSerializer {
	return &TwilioFrameSerializer{
		cfg:        cfg,
		ignoreRTVI: true,
		wire:       newWireAudio(audio.CodecMulaw, audio.CodecMulaw, twilioSampleRate),
		info: CallInfo{
			Provider:   "twilio",
			StreamID:   cfg.StreamSID,
			CallID:     cfg.CallSID,
			AccountID:  cfg.AccountSID,
			Encoding:   audio.CodecMulaw,
			SampleRate: twilioSampleRate,
		},
	}
}

func (s *TwilioFrameSerializer) WireSampleRates() (int, int) {
	return twilioSampleRate, twilioSampleRate
}

func (s *TwilioFrameSerializer) CallInfo() CallInfo { return s.info }

func (s *TwilioFrameSerializer) StreamSID() string { return s.cfg.StreamSID }

func (s *TwilioFrameSerializer) CallSID() string { return s.cfg.CallSID }

func (s *TwilioFrameSerializer) Handshake(msg Message) (bool, error) {
	var m twilioMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return false, fmt.Errorf("twilio handshake: %w", err)
	}
	if m.Event != "start" || m.Start == nil {
		return false, nil
	}
	st := m.Start
	s.cfg.StreamSID = firstNonEmpty(st.StreamSid, m.StreamSid, s.cfg.StreamSID)
	s.cfg.CallSID = firstNonEmpty(st.CallSid, s.cfg.CallSID)
	s.cfg.AccountSID = firstNonEmpty(s.cfg.AccountSID, st.AccountSid)
	s.info.StreamID = s.cfg.StreamSID
	s.info.CallID = s.cfg.CallSID
	s.info.AccountID = firstNonEmpty(st.AccountSid, s.cfg.AccountSID)
	s.info.CustomParameters = st.CustomParameters
	if st.MediaFormat.SampleRate > 0 {
		s.info.SampleRate = st.MediaFormat.SampleRate
	}
	if st.MediaFormat.Encoding != "" {
		s.info.Encoding = audio.NormalizeCodec(st.MediaFormat.Encoding)
	}
	s.prefix = nil
	return true, nil
}

func (s *TwilioFrameSerializer) NativeAudioFormat() (string, int) {
	return audio.CodecMulaw, twilioSampleRate
}

func (s *TwilioFrameSerializer) SetNativeAudioInput(enabled bool) { s.wire.native = enabled }

func (s *TwilioFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	enc, err := s.wire.encode(pcm, sampleRate)
	if err != nil {
		return dst, err
	}
	return s.SerializeNativeAudio(dst, enc)
}

func (s *TwilioFrameSerializer) SerializeNativeAudio(dst []Message, enc []byte) ([]Message, error) {
	if s.prefix == nil {
		s.prefix = append(s.prefix, `{"event":"media","streamSid":`...)
		s.prefix = appendJSONString(s.prefix, s.cfg.StreamSID)
		s.prefix = append(s.prefix, `,"media":{"payload":"`...)
	}
	return appendMediaMessage(dst, s.prefix, enc), nil
}

func (s *TwilioFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	switch frame.(type) {
	case *frames.InterruptionFrame:
		s.wire.reset()
		return s.control(dst, twilioMessage{Event: "clear", StreamSid: s.cfg.StreamSID})
	}
	if out, ok, err := serializeAudioFrame(s, &s.wire.rs, dst, frame, twilioSampleRate); ok {
		return out, err
	}
	out, _, err := transportMessage(dst, frame, s.ignoreRTVI)
	return out, err
}

func (s *TwilioFrameSerializer) control(dst []Message, m twilioMessage) ([]Message, error) {
	data, err := json.Marshal(m)
	if err != nil {
		return dst, err
	}
	return append(dst, Message{Type: TextMessage, Data: data}), nil
}

func (s *TwilioFrameSerializer) SerializePlaybackAck(correlationID string) (Message, error) {
	out, err := s.control(nil, twilioMessage{Event: "mark", StreamSid: s.cfg.StreamSID, Mark: &twilioMark{Name: correlationID}})
	if err != nil {
		return Message{}, err
	}
	return out[0], nil
}

func (s *TwilioFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	var m twilioMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("twilio: %w", err)
	}
	switch m.Event {
	case "media":
		if m.Media == nil || m.Media.Payload == "" {
			return nil, nil
		}
		raw, err := decodeBase64(m.Media.Payload)
		if err != nil {
			return nil, fmt.Errorf("twilio media payload: %w", err)
		}
		return s.wire.decode(raw)
	case "dtmf":
		if m.DTMF == nil {
			return nil, nil
		}
		return dtmfFrame(m.DTMF.Digit), nil
	case "mark":
		f := frames.NewPlaybackCompleteFrame()
		if m.Mark != nil {
			f.SetMetadata("correlation_id", m.Mark.Name)
		}
		return f, nil
	case "stop":
		return frames.NewEndFrame(), nil
	}
	return nil, nil
}

func (s *TwilioFrameSerializer) hangupCredentials() (string, string, bool) {
	if s.cfg.AccountSID == "" || s.cfg.CallSID == "" {
		return "", "", false
	}
	if s.cfg.AuthToken != "" {
		return s.cfg.AccountSID, s.cfg.AuthToken, true
	}
	if s.cfg.APIKeySID != "" && s.cfg.APIKeySecret != "" {
		return s.cfg.APIKeySID, s.cfg.APIKeySecret, true
	}
	return "", "", false
}

func (s *TwilioFrameSerializer) callURL() string {
	root := strings.TrimRight(s.cfg.BaseURL, "/")
	if root == "" {
		host := "api."
		if s.cfg.Edge != "" && s.cfg.Region != "" {
			host += s.cfg.Edge + "." + s.cfg.Region + "."
		}
		root = "https://" + host + "twilio.com"
	}
	return fmt.Sprintf("%s/2010-04-01/Accounts/%s/Calls/%s.json", root, url.PathEscape(s.cfg.AccountSID), url.PathEscape(s.cfg.CallSID))
}

func (s *TwilioFrameSerializer) Hangup(ctx context.Context) error {
	if s.cfg.DisableAutoHangUp {
		return nil
	}
	user, pass, ok := s.hangupCredentials()
	if !ok || !s.hungUp.CompareAndSwap(false, true) {
		return nil
	}
	req, err := http.NewRequest(http.MethodPost, s.callURL(), strings.NewReader(url.Values{"Status": {"completed"}}.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.SetBasicAuth(user, pass)
	status, body, err := apiRequest(ctx, s.cfg.HTTPClient, req, http.StatusOK)
	if status == http.StatusNotFound && strings.Contains(body, "20404") {
		logger.Debug("[Twilio] call %s already terminated", s.cfg.CallSID)
		return nil
	}
	if err != nil {
		return fmt.Errorf("twilio hangup %s: %w", s.cfg.CallSID, err)
	}
	logger.Info("[Twilio] terminated call %s", s.cfg.CallSID)
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
