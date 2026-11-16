package serializers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/audio"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type PlivoConfig struct {
	StreamID          string
	CallID            string
	AuthID            string
	AuthToken         string
	BaseURL           string
	TransferURL       string
	DisableAutoHangUp bool
	HTTPClient        *http.Client
}

type PlivoFrameSerializer struct {
	cfg    PlivoConfig
	info   CallInfo
	prefix []byte
	wire   wireAudio
	call   callControl
}

type plivoMessage struct {
	Event    string `json:"event"`
	StreamID string `json:"streamId"`
	Name     string `json:"name"`
	Start    *struct {
		CallID      string `json:"callId"`
		StreamID    string `json:"streamId"`
		AccountID   string `json:"accountId"`
		MediaFormat struct {
			Encoding   string `json:"encoding"`
			SampleRate int    `json:"sampleRate"`
		} `json:"mediaFormat"`
	} `json:"start"`
	ExtraHeaders string `json:"extra_headers"`
	Media        *struct {
		Payload string `json:"payload"`
	} `json:"media"`
	DTMF *struct {
		Digit string `json:"digit"`
	} `json:"dtmf"`
}

const plivoSampleRate = 8000

func NewPlivoFrameSerializer(cfg PlivoConfig) *PlivoFrameSerializer {
	return &PlivoFrameSerializer{
		cfg:  cfg,
		wire: newWireAudio(audio.CodecMulaw, audio.CodecMulaw, plivoSampleRate),
		info: CallInfo{Provider: "plivo", StreamID: cfg.StreamID, CallID: cfg.CallID, AccountID: cfg.AuthID, Encoding: audio.CodecMulaw, SampleRate: plivoSampleRate},
	}
}

func (s *PlivoFrameSerializer) WireSampleRates() (int, int) {
	return plivoSampleRate, plivoSampleRate
}

func (s *PlivoFrameSerializer) CallInfo() CallInfo { return s.info }

func (s *PlivoFrameSerializer) Handshake(msg Message) (bool, error) {
	var m plivoMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return false, fmt.Errorf("plivo handshake: %w", err)
	}
	if m.Event != "start" || m.Start == nil {
		return false, nil
	}
	s.cfg.StreamID = firstNonEmpty(m.Start.StreamID, m.StreamID, s.cfg.StreamID)
	s.cfg.CallID = firstNonEmpty(m.Start.CallID, s.cfg.CallID)
	s.info.StreamID = s.cfg.StreamID
	s.info.CallID = s.cfg.CallID
	s.info.AccountID = firstNonEmpty(m.Start.AccountID, s.cfg.AuthID)
	if m.ExtraHeaders != "" {
		params := map[string]string{}
		if json.Unmarshal([]byte(m.ExtraHeaders), &params) == nil {
			s.info.CustomParameters = params
		}
	}
	s.prefix = nil
	return true, nil
}

func (s *PlivoFrameSerializer) NativeAudioFormat() (string, int) {
	return audio.CodecMulaw, plivoSampleRate
}

func (s *PlivoFrameSerializer) SetNativeAudioInput(enabled bool) { s.wire.native = enabled }

func (s *PlivoFrameSerializer) SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error) {
	enc, err := s.wire.encode(pcm, sampleRate)
	if err != nil {
		return dst, err
	}
	return s.SerializeNativeAudio(dst, enc)
}

func (s *PlivoFrameSerializer) SerializeNativeAudio(dst []Message, enc []byte) ([]Message, error) {
	if s.prefix == nil {
		s.prefix = append(s.prefix, `{"event":"playAudio","streamId":`...)
		s.prefix = appendJSONString(s.prefix, s.cfg.StreamID)
		s.prefix = append(s.prefix, `,"media":{"contentType":"audio/x-mulaw","sampleRate":`+strconv.Itoa(plivoSampleRate)+`,"payload":"`...)
	}
	return appendMediaMessage(dst, s.prefix, enc), nil
}

func (s *PlivoFrameSerializer) Serialize(dst []Message, frame frames.Frame) ([]Message, error) {
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		s.wire.reset()
		data, err := json.Marshal(map[string]string{"event": "clearAudio", "streamId": s.cfg.StreamID})
		if err != nil {
			return dst, err
		}
		return append(dst, Message{Type: TextMessage, Data: data}), nil
	}
	if out, ok, err := serializeAudioFrame(s, &s.wire.rs, dst, frame, plivoSampleRate); ok {
		return out, err
	}
	out, _, err := transportMessage(dst, frame, true)
	return out, err
}

func (s *PlivoFrameSerializer) SerializePlaybackAck(correlationID string) (Message, error) {
	data, err := json.Marshal(map[string]string{"event": "checkpoint", "streamId": s.cfg.StreamID, "name": correlationID})
	return Message{Type: TextMessage, Data: data}, err
}

func (s *PlivoFrameSerializer) Deserialize(msg Message) (frames.Frame, error) {
	var m plivoMessage
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return nil, fmt.Errorf("plivo: %w", err)
	}
	switch m.Event {
	case "media":
		if m.Media == nil || m.Media.Payload == "" {
			return nil, nil
		}
		raw, err := decodeBase64(m.Media.Payload)
		if err != nil {
			return nil, fmt.Errorf("plivo media payload: %w", err)
		}
		return s.wire.decode(raw)
	case "dtmf":
		if m.DTMF == nil {
			return nil, nil
		}
		return dtmfFrame(m.DTMF.Digit), nil
	case "playedStream":
		f := frames.NewPlaybackCompleteFrame()
		f.SetMetadata("correlation_id", m.Name)
		return f, nil
	case "stop":
		return frames.NewEndFrame(), nil
	}
	return nil, nil
}

func (s *PlivoFrameSerializer) Hangup(ctx context.Context) error {
	if s.cfg.DisableAutoHangUp || s.cfg.AuthID == "" || s.cfg.AuthToken == "" || s.cfg.CallID == "" || !s.call.beginHangup(ctx) {
		return nil
	}
	root := strings.TrimRight(firstNonEmpty(s.cfg.BaseURL, "https://api.plivo.com"), "/")
	endpoint := root + "/v1/Account/" + url.PathEscape(s.cfg.AuthID) + "/Call/" + url.PathEscape(s.cfg.CallID) + "/"
	req, err := http.NewRequest(http.MethodDelete, endpoint, nil)
	if err != nil {
		return err
	}
	req.SetBasicAuth(s.cfg.AuthID, s.cfg.AuthToken)
	status, _, err := apiRequest(ctx, s.cfg.HTTPClient, req, http.StatusNoContent, http.StatusOK)
	if status == http.StatusNotFound {
		logger.Debug("[Plivo] call %s already terminated", s.cfg.CallID)
		return nil
	}
	if err != nil {
		return fmt.Errorf("plivo hangup %s: %w", s.cfg.CallID, err)
	}
	logger.Info("[Plivo] terminated call %s", s.cfg.CallID)
	return nil
}

func (s *PlivoFrameSerializer) CanTransfer() bool {
	return s.cfg.AuthID != "" && s.cfg.AuthToken != "" && s.cfg.CallID != "" && s.cfg.TransferURL != ""
}

func (s *PlivoFrameSerializer) Transfer(ctx context.Context, req TransferRequest) error {
	if err := ValidateTransferRequest(req); err != nil {
		return err
	}
	if !s.CanTransfer() {
		return ErrTransferUnsupported
	}
	alegURL, err := plivoTransferURL(s.cfg.TransferURL, req)
	if err != nil {
		return err
	}
	body, err := json.Marshal(map[string]string{"legs": "aleg", "aleg_url": alegURL, "aleg_method": http.MethodGet})
	if err != nil {
		return err
	}
	root := strings.TrimRight(firstNonEmpty(s.cfg.BaseURL, "https://api.plivo.com"), "/")
	endpoint := root + "/v1/Account/" + url.PathEscape(s.cfg.AuthID) + "/Call/" + url.PathEscape(s.cfg.CallID) + "/"
	r, err := http.NewRequest(http.MethodPost, endpoint, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	r.Header.Set("Content-Type", "application/json")
	r.SetBasicAuth(s.cfg.AuthID, s.cfg.AuthToken)
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := s.call.beginTransfer(); err != nil {
		return err
	}
	status, _, err := apiRequest(ctx, s.cfg.HTTPClient, r, http.StatusOK, http.StatusAccepted, http.StatusNoContent)
	s.call.endTransfer(status, err)
	if err != nil {
		return fmt.Errorf("plivo transfer %s: %w", s.cfg.CallID, transferResult(status, err))
	}
	logger.Info("[Plivo] transfer of call %s accepted", s.cfg.CallID)
	return nil
}

func plivoTransferURL(base string, req TransferRequest) (string, error) {
	u, err := url.Parse(base)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return "", fmt.Errorf("plivo transfer: TransferURL must be an absolute http(s) URL")
	}
	q := u.Query()
	q.Set("destination", req.Destination)
	if req.CallerID != "" {
		q.Set("caller_id", req.CallerID)
	}
	if req.Announcement != "" {
		q.Set("announcement", req.Announcement)
	}
	u.RawQuery = q.Encode()
	return u.String(), nil
}

func (s *PlivoFrameSerializer) CanSendDTMF() bool {
	return s.cfg.AuthID != "" && s.cfg.AuthToken != "" && s.cfg.CallID != ""
}

func (s *PlivoFrameSerializer) SendDTMF(ctx context.Context, buttons []frames.KeypadEntry) error {
	if !s.CanSendDTMF() || len(buttons) == 0 {
		return nil
	}
	body, err := json.Marshal(map[string]string{"digits": frames.KeypadEntriesString(buttons)})
	if err != nil {
		return err
	}
	root := strings.TrimRight(firstNonEmpty(s.cfg.BaseURL, "https://api.plivo.com"), "/")
	endpoint := root + "/v1/Account/" + url.PathEscape(s.cfg.AuthID) + "/Call/" + url.PathEscape(s.cfg.CallID) + "/DTMF/"
	req, err := http.NewRequest(http.MethodPost, endpoint, strings.NewReader(string(body)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.SetBasicAuth(s.cfg.AuthID, s.cfg.AuthToken)
	if _, _, err := apiRequest(ctx, s.cfg.HTTPClient, req, http.StatusOK, http.StatusAccepted, http.StatusNoContent); err != nil {
		return fmt.Errorf("plivo dtmf %s: %w", s.cfg.CallID, err)
	}
	return nil
}
