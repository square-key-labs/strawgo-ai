package serializers

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptrace"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/square-key-labs/strawgo-ai/src/audio"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

type MessageType uint8

const (
	TextMessage   MessageType = 1
	BinaryMessage MessageType = 2
)

type Message struct {
	Type MessageType
	Data []byte
}

func Text(s string) Message { return Message{Type: TextMessage, Data: []byte(s)} }

func Binary(b []byte) Message { return Message{Type: BinaryMessage, Data: b} }

type FrameSerializer interface {
	SerializeAudio(dst []Message, pcm []byte, sampleRate int) ([]Message, error)
	Serialize(dst []Message, frame frames.Frame) ([]Message, error)
	Deserialize(msg Message) (frames.Frame, error)
}

type WireFormat interface {
	WireSampleRates() (in, out int)
}

type CallInfo struct {
	Provider         string
	StreamID         string
	CallID           string
	AccountID        string
	From             string
	To               string
	Encoding         string
	SampleRate       int
	CustomParameters map[string]string
}

type Handshaker interface {
	Handshake(msg Message) (done bool, err error)
	CallInfo() CallInfo
}

type Hangupper interface {
	Hangup(ctx context.Context) error
}

type TransferRequest struct {
	Destination  string
	CallerID     string
	Announcement string
}

type Transferer interface {
	CanTransfer() bool
	Transfer(ctx context.Context, req TransferRequest) error
}

var (
	ErrTransferUnsupported    = errors.New("call transfer is not supported for this call")
	ErrInvalidDestination     = errors.New("invalid transfer destination")
	ErrInvalidCallerID        = errors.New("invalid transfer caller id")
	ErrCallEnded              = errors.New("call already ended")
	ErrTransferInProgress     = errors.New("a transfer is already in progress")
	ErrTransferOutcomeUnknown = errors.New("transfer outcome unknown: the carrier did not answer")
	ErrAnnouncementTooLong    = errors.New("transfer announcement too long")
	ErrTransferCancelled      = errors.New("transfer cancelled before it started")
)

const maxAnnouncementLen = 1000

var (
	e164Re     = regexp.MustCompile(`^\+[1-9][0-9]{6,14}$`)
	sipUserRe  = regexp.MustCompile(`^[A-Za-z0-9._~!$&'()*+,=%-]{1,64}$`)
	sipLabelRe = regexp.MustCompile(`^[A-Za-z0-9]([A-Za-z0-9-]{0,61}[A-Za-z0-9])?$`)
	sipParamRe = regexp.MustCompile(`^[A-Za-z0-9._~%-]+(=[A-Za-z0-9._~%-]+)?$`)
	digitsRe   = regexp.MustCompile(`^[0-9]{1,5}$`)
)

func IsE164(s string) bool { return e164Re.MatchString(s) }

func IsSIPURI(s string) bool {
	rest, ok := strings.CutPrefix(s, "sip:")
	if !ok {
		if rest, ok = strings.CutPrefix(s, "sips:"); !ok {
			return false
		}
	}
	rest, params, hasParams := strings.Cut(rest, ";")
	user, hostport, ok := strings.Cut(rest, "@")
	if !ok || !sipUserRe.MatchString(user) {
		return false
	}
	host, port, hasPort := strings.Cut(hostport, ":")
	if hasPort {
		n, err := strconv.Atoi(port)
		if !digitsRe.MatchString(port) || err != nil || n < 1 || n > 65535 {
			return false
		}
	}
	if host == "" || len(host) > 253 {
		return false
	}
	for _, label := range strings.Split(host, ".") {
		if !sipLabelRe.MatchString(label) {
			return false
		}
	}
	if hasParams {
		list := strings.Split(params, ";")
		if len(list) > 8 {
			return false
		}
		for _, p := range list {
			if !sipParamRe.MatchString(p) {
				return false
			}
		}
	}
	return true
}

func ValidateTransferRequest(req TransferRequest) error {
	if !IsE164(req.Destination) && !IsSIPURI(req.Destination) {
		return ErrInvalidDestination
	}
	if req.CallerID != "" && !IsE164(req.CallerID) {
		return ErrInvalidCallerID
	}
	if len(req.Announcement) > maxAnnouncementLen {
		return fmt.Errorf("%w: longer than %d bytes", ErrAnnouncementTooLong, maxAnnouncementLen)
	}
	return nil
}

type callPhase uint8

const (
	phaseActive callPhase = iota
	phaseTransferring
	phaseTransferred
	phaseUnknown
	phaseEnded
)

type callControl struct {
	mu      sync.Mutex
	phase   callPhase
	settled chan struct{}
}

func (c *callControl) beginTransfer() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch c.phase {
	case phaseActive:
		c.phase = phaseTransferring
		c.settled = make(chan struct{})
		return nil
	case phaseTransferring:
		return ErrTransferInProgress
	}
	return ErrCallEnded
}

func (c *callControl) endTransfer(status int, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	switch {
	case err == nil:
		c.phase = phaseTransferred
	case errors.Is(err, ErrCallEnded):
		c.phase = phaseEnded
	case outcomeUnknown(status, err):
		c.phase = phaseUnknown
	default:
		c.phase = phaseActive
	}
	close(c.settled)
}

func (c *callControl) beginHangup(ctx context.Context) bool {
	for {
		c.mu.Lock()
		switch c.phase {
		case phaseActive:
			c.phase = phaseEnded
			c.mu.Unlock()
			return true
		case phaseTransferring:
			settled := c.settled
			c.mu.Unlock()
			select {
			case <-settled:
				continue
			case <-ctx.Done():
				return false
			}
		}
		c.mu.Unlock()
		return false
	}
}

func transferResult(status int, err error) error {
	if outcomeUnknown(status, err) {
		return fmt.Errorf("%w: %w", ErrTransferOutcomeUnknown, err)
	}
	return err
}

var errRequestNotSent = errors.New("request not sent")

func outcomeUnknown(status int, err error) bool {
	return err != nil && status == 0 && !errors.Is(err, ErrCallEnded) && !errors.Is(err, errRequestNotSent)
}

func transferRequest(ctx context.Context, client *http.Client, req *http.Request, ok ...int) (int, string, error) {
	var traced, wrote atomic.Bool
	ctx = httptrace.WithClientTrace(ctx, &httptrace.ClientTrace{
		GetConn:      func(string) { traced.Store(true) },
		WroteHeaders: func() { wrote.Store(true) },
	})
	status, body, err := apiRequest(ctx, client, req, ok...)
	if err != nil && status == 0 && ((traced.Load() && !wrote.Load()) || neverSent(err)) {
		return status, body, fmt.Errorf("%w: %w", errRequestNotSent, err)
	}
	return status, body, err
}

func neverSent(err error) bool {
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op == "dial" {
		return true
	}
	var recordErr tls.RecordHeaderError
	if errors.As(err, &recordErr) {
		return true
	}
	var certErr *tls.CertificateVerificationError
	if errors.As(err, &certErr) {
		return true
	}
	var unknownAuth x509.UnknownAuthorityError
	var hostErr x509.HostnameError
	return errors.As(err, &unknownAuth) || errors.As(err, &hostErr)
}

type PlaybackAckSerializer interface {
	SerializePlaybackAck(correlationID string) (Message, error)
}

type DTMFSerializer interface {
	SerializeDTMF(dst []Message, buttons []frames.KeypadEntry) ([]Message, error)
}

type DTMFSender interface {
	CanSendDTMF() bool
	SendDTMF(ctx context.Context, buttons []frames.KeypadEntry) error
}

type NativeAudioSerializer interface {
	NativeAudioFormat() (codec string, sampleRate int)
	SetNativeAudioInput(enabled bool)
	SerializeNativeAudio(dst []Message, data []byte) ([]Message, error)
}

type MediaFlowFrame struct {
	*frames.SystemFrame
	Paused bool
}

func NewMediaFlowFrame(paused bool) *MediaFlowFrame {
	return &MediaFlowFrame{SystemFrame: &frames.SystemFrame{BaseFrame: frames.NewBaseFrame("MediaFlowFrame")}, Paused: paused}
}

const rtviMessageLabel = "rtvi-ai"

type LabeledMessage interface {
	MessageLabel() string
}

func isRTVIMessage(msg any) bool {
	switch m := msg.(type) {
	case LabeledMessage:
		return m.MessageLabel() == rtviMessageLabel
	case map[string]any:
		label, _ := m["label"].(string)
		return label == rtviMessageLabel
	}
	return false
}

func transportMessage(dst []Message, frame frames.Frame, ignoreRTVI bool) ([]Message, bool, error) {
	var msg any
	switch f := frame.(type) {
	case *frames.OutputTransportMessageFrame:
		msg = f.Message
	case *frames.OutputTransportMessageUrgentFrame:
		msg = f.Message
	default:
		return dst, false, nil
	}
	if ignoreRTVI && isRTVIMessage(msg) {
		return dst, true, nil
	}
	switch v := msg.(type) {
	case string:
		return append(dst, Text(v)), true, nil
	case []byte:
		return append(dst, Binary(v)), true, nil
	case json.RawMessage:
		return append(dst, Message{Type: TextMessage, Data: v}), true, nil
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return dst, true, err
	}
	return append(dst, Message{Type: TextMessage, Data: data}), true, nil
}

func dtmfFrame(digit string) frames.Frame {
	k := frames.KeypadEntry(digit)
	if !k.Valid() {
		return nil
	}
	return frames.NewInputDTMFFrame(k)
}

func inputAudio(pcm []byte, sampleRate int) *frames.AudioFrame {
	f := frames.NewAudioFrame(pcm, sampleRate, 1)
	f.SetMetadata("codec", audio.CodecLinear16)
	return f
}

func audioFramePCM(rs *audio.Resampler, frame frames.Frame, outRate int) ([]byte, int, bool, error) {
	var data []byte
	var rate int
	var meta map[string]interface{}
	switch f := frame.(type) {
	case *frames.TTSAudioFrame:
		data, rate, meta = f.Data, f.SampleRate, f.Metadata()
	case *frames.AudioFrame:
		data, rate, meta = f.Data, f.SampleRate, f.Metadata()
	default:
		return nil, 0, false, nil
	}
	codec, _ := meta["codec"].(string)
	pcm, err := audio.DecodePCM16(nil, data, codec)
	if err != nil {
		return nil, 0, true, err
	}
	if outRate > 0 && rate > 0 && rate != outRate {
		pcm = rs.Resample(make([]byte, 0, rs.MaxOutputBytes(len(pcm), rate, outRate)), pcm, rate, outRate)
		rate = outRate
	}
	return pcm, rate, true, nil
}

func serializeAudioFrame(s FrameSerializer, rs *audio.Resampler, dst []Message, frame frames.Frame, outRate int) ([]Message, bool, error) {
	pcm, rate, ok, err := audioFramePCM(rs, frame, outRate)
	if !ok || err != nil {
		return dst, ok, err
	}
	dst, err = s.SerializeAudio(dst, pcm, rate)
	return dst, true, err
}

func apiRequest(ctx context.Context, client *http.Client, req *http.Request, ok ...int) (int, string, error) {
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req.WithContext(ctx))
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	for _, code := range ok {
		if resp.StatusCode == code {
			return resp.StatusCode, string(body), nil
		}
	}
	return resp.StatusCode, string(body), fmt.Errorf("status %d: %s", resp.StatusCode, body)
}

type wireAudio struct {
	inCodec  string
	outCodec string
	rate     int
	native   bool
	rs       audio.Resampler
	rsBuf    []byte
	enc      []byte
}

func newWireAudio(inCodec, outCodec string, rate int) wireAudio {
	return wireAudio{inCodec: audio.NormalizeCodec(inCodec), outCodec: audio.NormalizeCodec(outCodec), rate: rate}
}

func (w *wireAudio) encode(pcm []byte, sampleRate int) ([]byte, error) {
	if sampleRate > 0 && sampleRate != w.rate {
		w.rsBuf = w.rs.Resample(w.rsBuf[:0], pcm, sampleRate, w.rate)
		pcm = w.rsBuf
	}
	if w.outCodec == audio.CodecLinear16 {
		return pcm, nil
	}
	var err error
	w.enc, err = audio.EncodePCM16(w.enc[:0], pcm, w.outCodec)
	return w.enc, err
}

func (w *wireAudio) encodeOwned(pcm []byte, sampleRate int) ([]byte, error) {
	if w.outCodec == audio.CodecLinear16 && (sampleRate <= 0 || sampleRate == w.rate) {
		return pcm, nil
	}
	enc, err := w.encode(pcm, sampleRate)
	if err != nil {
		return nil, err
	}
	return bytes.Clone(enc), nil
}

func appendMediaMessage(dst []Message, prefix, payload []byte) []Message {
	buf := make([]byte, 0, len(prefix)+base64.StdEncoding.EncodedLen(len(payload))+len(mediaSuffix))
	buf = append(buf, prefix...)
	buf = base64.StdEncoding.AppendEncode(buf, payload)
	buf = append(buf, mediaSuffix...)
	return append(dst, Message{Type: TextMessage, Data: buf})
}

const mediaSuffix = `"}}`

func (w *wireAudio) decode(raw []byte) (frames.Frame, error) {
	if w.inCodec == audio.CodecLinear16 {
		if len(raw)%2 != 0 {
			raw = raw[:len(raw)-1]
		}
		return inputAudio(raw, w.rate), nil
	}
	if w.native {
		f := frames.NewAudioFrame(raw, w.rate, 1)
		f.SetMetadata("codec", w.inCodec)
		return f, nil
	}
	pcm, err := audio.DecodePCM16(make([]byte, 0, len(raw)*2), raw, w.inCodec)
	if err != nil {
		return nil, err
	}
	return inputAudio(pcm, w.rate), nil
}

func (w *wireAudio) reset() { w.rs.Reset() }

func appendJSONString(dst []byte, s string) []byte {
	b, _ := json.Marshal(s)
	return append(dst, b...)
}

func decodeBase64(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}
