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

type TelnyxConfig struct {
	StreamID          string
	CallControlID     string
	APIKey            string
	InboundEncoding   string
	OutboundEncoding  string
	BaseURL           string
	DisableAutoHangUp bool
	HTTPClient        *http.Client
}

var telnyxMediaPrefix = []byte(`{"event":"media","media":{"payload":"`)

type TelnyxFrameSerializer struct {
	cfg    TelnyxConfig
	info   CallInfo
	inEnc  string
	outEnc string
	wire   wireAudio
	hungUp atomic.Bool
}

type telnyxMessage struct {
	Event    string `json:"event"`
	StreamID string `json:"stream_id"`
	Start    *struct {
		CallControlID string `json:"call_control_id"`
		From          string `json:"from"`
		To            string `json:"to"`
		ClientState   string `json:"client_state"`
		MediaFormat   struct {
			Encoding   string `json:"encoding"`
			SampleRate int    `json:"sample_rate"`
		} `json:"media_format"`
	} `json:"start"`
	Media *struct {
		Payload string `json:"payload"`
	} `json:"media"`
	DTMF *struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
	Mark *struct {
		Name string `json:"name"`
	} `json:"mark"`
}

const telnyxSampleRate = 8000

func NewTelnyxFrameSerializer(cfg TelnyxConfig) *TelnyxFrameSerializer {
	s := &TelnyxFrameSerializer{
		cfg:  cfg,
		info: CallInfo{Provider: "telnyx", StreamID: cfg.StreamID, CallID: cfg.CallControlID, SampleRate: telnyxSampleRate},
	}
	s.setEncodings(firstNonEmpty(cfg.InboundEncoding, "PCMU"), firstNonEmpty(cfg.OutboundEncoding, cfg.InboundEncoding, "PCMU"))
	return s
}

func (s *TelnyxFrameSerializer) setEncodings(in, out string) {
	s.inEnc = audio.NormalizeCodec(in)
	s.outEnc = audio.NormalizeCodec(out)
	s.info.Encoding = s.inEnc
	native := s.wire.native
	s.wire = newWireAudio(s.inEnc, s.outEnc, telnyxSampleRate)
	s.wire.native = native
}

func (s *TelnyxFrameSerializer) WireSampleRates() (int, int) {
	return telnyxSampleRate, telnyxSampleRate
}

func (s *TelnyxFrameSerializer) CallInfo() CallInfo { return s.info }

func (s *TelnyxFrameSerializer) Handshake(msg Message) (bool, error) {
	var m telnyxMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return false, fmt.Errorf("telnyx handshake: %w", err)
	}
	if m.Event != "start" || m.Start == nil {
		return false, nil
	}
	s.cfg.StreamID = firstNonEmpty(m.StreamID, s.cfg.StreamID)
	s.cfg.CallControlID = firstNonEmpty(m.Start.CallControlID, s.cfg.CallControlID)
	s.info.StreamID = s.cfg.StreamID
	s.info.CallID = s.cfg.CallControlID
	s.info.From = m.Start.From
	s.info.To = m.Start.To
	if m.Start.ClientState != "" {
		s.info.CustomParameters = map[string]string{"client_state": m.Start.ClientState}
	}
	if enc := m.Start.MediaFormat.Encoding; enc != "" && s.cfg.InboundEncoding == "" {
		out := s.cfg.OutboundEncoding
		if out == "" {
			out = enc
		}
		s.setEncodings(enc, out)
	}
	return true, nil
}

func (s *TelnyxFrameSerializer) NativeAudioFormat() (string, int) {
	return s.outEnc, telnyxSampleRate
}

func (s *TelnyxFrameSerializer) SetNativeAudioInput(enabled bool) { s.wire.native = enabled }

func (s *TelnyxFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	enc, err := s.wire.encode(pcm, sampleRate)
	if err != nil {
		return dst, err
	}
	return s.SerializeNativeAudio(dst, enc)
}

func (s *TelnyxFrameSerializer) SerializeNativeAudio(dst []Message, enc []byte) ([]Message, error) {
	return appendMediaMessage(dst, telnyxMediaPrefix, enc), nil
}

func (s *TelnyxFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.wire.reset()
		return append(dst, Text(`{"event":"clear"}`)), nil
	}
	if out, ok, err := serializeAudioFrame(s, &s.wire.rs, dst, frame, telnyxSampleRate); ok {
		return out, err
	}
	out, _, err := transportMessage(dst, frame, true)
	return out, err
}

func (s *TelnyxFrameSerializer) SerializePlaybackAck(correlationID string) (Message, error) {
	data, err := json.Marshal(map[string]any{"event": "mark", "mark": map[string]string{"name": correlationID}})
	return Message{Type: TextMessage, Data: data}, err
}

func (s *TelnyxFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	var m telnyxMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("telnyx: %w", err)
	}
	switch m.Event {
	case "media":
		if m.Media == nil || m.Media.Payload == "" {
			return nil, nil
		}
		raw, err := decodeBase64(m.Media.Payload)
		if err != nil {
			return nil, fmt.Errorf("telnyx media payload: %w", err)
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

func (s *TelnyxFrameSerializer) Hangup(ctx context.Context) error {
	if s.cfg.DisableAutoHangUp || s.cfg.APIKey == "" || s.cfg.CallControlID == "" || !s.hungUp.CompareAndSwap(false, true) {
		return nil
	}
	root := strings.TrimRight(firstNonEmpty(s.cfg.BaseURL, "https://api.telnyx.com"), "/")
	endpoint := root + "/v2/calls/" + url.PathEscape(s.cfg.CallControlID) + "/actions/hangup"
	req, err := http.NewRequest(http.MethodPost, endpoint, strings.NewReader("{}"))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	status, body, err := apiRequest(ctx, s.cfg.HTTPClient, req, http.StatusOK)
	if status == http.StatusUnprocessableEntity && strings.Contains(body, "90018") {
		logger.Debug("[Telnyx] call %s already terminated", s.cfg.CallControlID)
		return nil
	}
	if err != nil {
		return fmt.Errorf("telnyx hangup %s: %w", s.cfg.CallControlID, err)
	}
	logger.Info("[Telnyx] terminated call %s", s.cfg.CallControlID)
	return nil
}

func (s *TelnyxFrameSerializer) CanSendDTMF() bool {
	return s.cfg.APIKey != "" && s.cfg.CallControlID != ""
}

func (s *TelnyxFrameSerializer) SendDTMF(ctx context.Context, buttons []frames.KeypadEntry) error {
	if !s.CanSendDTMF() || len(buttons) == 0 {
		return nil
	}
	body, err := json.Marshal(map[string]any{"digits": frames.KeypadEntriesString(buttons), "duration_millis": 250})
	if err != nil {
		return err
	}
	root := strings.TrimRight(firstNonEmpty(s.cfg.BaseURL, "https://api.telnyx.com"), "/")
	req, err := http.NewRequest(http.MethodPost, root+"/v2/calls/"+url.PathEscape(s.cfg.CallControlID)+"/actions/send_dtmf", strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	if _, _, err := apiRequest(ctx, s.cfg.HTTPClient, req, http.StatusOK); err != nil {
		return fmt.Errorf("telnyx send_dtmf %s: %w", s.cfg.CallControlID, err)
	}
	return nil
}
