package callcontrol

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
)

const (
	TransferCallTool = "transfer_call"
	EndCallTool      = "end_call"
	SendDTMFTool     = "send_dtmf"
)

const (
	DefaultTransferTimeout = 30 * time.Second
	DefaultMaxDTMFDigits   = 20
	maxGoodbyeLen          = 500
)

type Destination struct {
	Name         string
	Description  string
	Target       string
	CallerID     string
	Announcement string
}

type Resolver func(ctx context.Context, name string) (Destination, bool)

type Config struct {
	Destinations    []Destination
	Resolver        Resolver
	TransferMessage string
	TransferTimeout time.Duration
	EndCall         bool
	GoodbyeMessage  string
	EndCallGuard    func(ctx context.Context) error
	SendDTMF        bool
	MaxDTMFDigits   int
}

type Registrar interface {
	RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions)
}

type FrameQueuer interface {
	QueueFrame(frame frames.Frame, direction ...frames.FrameDirection) error
}

var nameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_ .-]{0,63}$`)

func Register(r Registrar, q FrameQueuer, cfg Config) ([]llm.FunctionSchema, error) {
	if r == nil || q == nil {
		return nil, errors.New("callcontrol: registrar and frame queuer are required")
	}
	t, err := newTools(q, cfg)
	if err != nil {
		return nil, err
	}
	var schemas []llm.FunctionSchema
	keep := llm.FunctionOptions{CancelOnInterruption: false}
	if t.transferEnabled() {
		r.RegisterFunction(TransferCallTool, t.transfer, keep)
		schemas = append(schemas, t.transferSchema())
	}
	if cfg.EndCall {
		r.RegisterFunction(EndCallTool, t.endCall, keep)
		schemas = append(schemas, endCallSchema())
	}
	if cfg.SendDTMF {
		r.RegisterFunction(SendDTMFTool, t.sendDTMF, keep)
		schemas = append(schemas, t.dtmfSchema())
	}
	if len(schemas) == 0 {
		return nil, errors.New("callcontrol: no tools enabled")
	}
	return schemas, nil
}

func EndCall(q FrameQueuer, closingLine string) error {
	if line := strings.TrimSpace(closingLine); line != "" {
		if err := q.QueueFrame(frames.NewTTSSpeakFrame(line)); err != nil {
			return err
		}
	}
	return q.QueueFrame(frames.NewEndFrame())
}

type tools struct {
	q        FrameQueuer
	cfg      Config
	allowed  map[string]Destination
	names    []string
	maxDigit int
}

func newTools(q FrameQueuer, cfg Config) (*tools, error) {
	t := &tools{q: q, cfg: cfg, allowed: make(map[string]Destination, len(cfg.Destinations)), maxDigit: cfg.MaxDTMFDigits}
	if t.maxDigit <= 0 {
		t.maxDigit = DefaultMaxDTMFDigits
	}
	if t.cfg.TransferTimeout <= 0 {
		t.cfg.TransferTimeout = DefaultTransferTimeout
	}
	for _, d := range cfg.Destinations {
		if !validName(d.Name) {
			return nil, fmt.Errorf("callcontrol: destination name %q must be 1-64 letters, digits, spaces, '.', '_' or '-', with fewer than %d digits", d.Name, maxNameDigits)
		}
		key := strings.ToLower(d.Name)
		if _, dup := t.allowed[key]; dup {
			return nil, fmt.Errorf("callcontrol: duplicate destination name %q", d.Name)
		}
		if err := serializers.ValidateTransferRequest(d.request()); err != nil {
			return nil, fmt.Errorf("callcontrol: destination %q: %w", d.Name, err)
		}
		t.allowed[key] = d
		t.names = append(t.names, d.Name)
	}
	sort.Strings(t.names)
	return t, nil
}

func (d Destination) request() serializers.TransferRequest {
	return serializers.TransferRequest{Destination: d.Target, CallerID: d.CallerID, Announcement: d.Announcement}
}

func (t *tools) transferEnabled() bool {
	return len(t.allowed) > 0 || t.cfg.Resolver != nil
}

func (t *tools) transferSchema() llm.FunctionSchema {
	prop := map[string]any{"type": "string"}
	desc := "Name of the destination to transfer to."
	if len(t.names) > 0 {
		var b strings.Builder
		b.WriteString("Name of the destination to transfer to. Only these names are accepted:")
		for _, n := range t.names {
			d := t.allowed[strings.ToLower(n)]
			b.WriteString(" " + n)
			if d.Description != "" {
				b.WriteString(" (" + d.Description + ")")
			}
			b.WriteString(";")
		}
		desc = strings.TrimSuffix(b.String(), ";") + "."
		if t.cfg.Resolver == nil {
			prop["enum"] = append([]string(nil), t.names...)
		}
	}
	prop["description"] = desc
	return llm.FunctionSchema{
		Name:        TransferCallTool,
		Description: "Transfer the caller to a person or team. Use only when the caller asks for it or the task requires it. Never pass a phone number the caller dictates; pass a destination name.",
		Properties:  map[string]any{"destination": prop},
		Required:    []string{"destination"},
	}
}

func endCallSchema() llm.FunctionSchema {
	return llm.FunctionSchema{
		Name:        EndCallTool,
		Description: "End the phone call once the conversation is finished. The goodbye line is spoken before the call ends.",
		Properties: map[string]any{
			"goodbye": map[string]any{"type": "string", "description": "Optional short closing line to say before hanging up."},
		},
	}
}

func (t *tools) dtmfSchema() llm.FunctionSchema {
	return llm.FunctionSchema{
		Name:        SendDTMFTool,
		Description: "Queue keypad key presses on the call, sent after anything already being said, for example to navigate a phone menu.",
		Properties: map[string]any{
			"digits": map[string]any{
				"type":        "string",
				"description": fmt.Sprintf("Keys to press in order, using only 0-9, * and #, at most %d keys.", t.maxDigit),
			},
		},
		Required: []string{"digits"},
	}
}

func failed(msg string) (llm.FunctionResult, error) {
	return llm.FunctionResult{Value: map[string]any{"status": "failed", "error": msg}}, nil
}

const maxNameDigits = 5

func validName(name string) bool {
	if !nameRe.MatchString(name) {
		return false
	}
	digits := 0
	for _, r := range name {
		if r >= '0' && r <= '9' {
			digits++
		}
	}
	return digits < maxNameDigits
}

func (t *tools) resolve(ctx context.Context, name string) (Destination, bool) {
	if !validName(name) {
		return Destination{}, false
	}
	if d, ok := t.allowed[strings.ToLower(name)]; ok {
		return d, true
	}
	if t.cfg.Resolver != nil {
		return t.cfg.Resolver(ctx, name)
	}
	return Destination{}, false
}

func (t *tools) transfer(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
	name, _ := p.Arguments["destination"].(string)
	name = strings.TrimSpace(name)
	if name == "" {
		return failed("destination is required")
	}
	d, ok := t.resolve(ctx, name)
	if !ok {
		msg := "destination " + quote(name) + " is not allowed"
		if len(t.names) > 0 {
			msg += "; allowed destinations: " + strings.Join(t.names, ", ")
		}
		return failed(msg)
	}
	if err := serializers.ValidateTransferRequest(d.request()); err != nil {
		logger.Warn("[callcontrol] resolver returned an invalid destination for %s: %v", quote(name), err)
		return failed("destination " + quote(name) + " is not configured correctly")
	}
	if t.cfg.TransferMessage != "" {
		if err := t.q.QueueFrame(frames.NewTTSSpeakFrame(t.cfg.TransferMessage)); err != nil {
			return failed("the call is no longer active")
		}
	}
	done := make(chan error, 1)
	f := frames.NewTransferCallFrame(d.Target)
	f.CallerID = d.CallerID
	f.Announcement = d.Announcement
	f.OnComplete = func(err error) { done <- err }
	if err := t.q.QueueFrame(f); err != nil {
		return failed("the call is no longer active")
	}
	timer := time.NewTimer(t.cfg.TransferTimeout)
	defer timer.Stop()
	var err error
	select {
	case err = <-done:
	case <-timer.C:
		if f.Cancel() {
			return failed("the transfer to " + quote(name) + " did not start in time and was cancelled; the caller is still with you")
		}
		select {
		case err = <-done:
		case <-ctx.Done():
			return llm.FunctionResult{}, ctx.Err()
		}
	case <-ctx.Done():
		f.Cancel()
		return llm.FunctionResult{}, ctx.Err()
	}
	if err != nil {
		return failed(transferFailure(name, err))
	}
	runLLM := false
	return llm.FunctionResult{Value: map[string]any{"status": "transfer_accepted", "destination": name}, RunLLM: &runLLM}, nil
}

func transferFailure(name string, err error) string {
	switch {
	case errors.Is(err, serializers.ErrTransferUnsupported):
		return "transfer is not available on this call"
	case errors.Is(err, serializers.ErrCallEnded):
		return "the call has already ended"
	case errors.Is(err, serializers.ErrTransferInProgress):
		return "another transfer is already in progress"
	case errors.Is(err, serializers.ErrTransferCancelled):
		return "the transfer was cancelled before it started"
	case errors.Is(err, serializers.ErrTransferOutcomeUnknown):
		return "the carrier did not confirm the transfer to " + quote(name) + "; it may or may not have gone through, so do not retry it"
	}
	return "the transfer to " + quote(name) + " failed; tell the caller and offer another option"
}

func quote(s string) string {
	if len(s) > 64 {
		s = s[:64]
	}
	return fmt.Sprintf("%q", s)
}

func (t *tools) endCall(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
	line, _ := p.Arguments["goodbye"].(string)
	line = strings.TrimSpace(line)
	if len(line) > maxGoodbyeLen {
		return failed(fmt.Sprintf("goodbye must be at most %d characters", maxGoodbyeLen))
	}
	if t.cfg.EndCallGuard != nil {
		if err := t.cfg.EndCallGuard(ctx); err != nil {
			return failed("the call cannot end yet: " + err.Error())
		}
	}
	if line == "" {
		line = t.cfg.GoodbyeMessage
	}
	if err := EndCall(t.q, line); err != nil {
		return failed("the call is no longer active")
	}
	runLLM := false
	return llm.FunctionResult{Value: map[string]any{"status": "ending"}, RunLLM: &runLLM}, nil
}

func (t *tools) sendDTMF(_ context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
	digits, _ := p.Arguments["digits"].(string)
	digits = strings.ReplaceAll(digits, " ", "")
	if digits == "" {
		return failed("digits is required")
	}
	if len(digits) > t.maxDigit {
		return failed(fmt.Sprintf("at most %d digits can be sent at once", t.maxDigit))
	}
	buttons, err := frames.ParseKeypadEntries(digits)
	if err != nil {
		return failed("digits may contain only 0-9, * and #")
	}
	if err := t.q.QueueFrame(frames.NewOutputDTMFFrame(buttons...)); err != nil {
		return failed("the call is no longer active")
	}
	return llm.FunctionResult{Value: map[string]any{"status": "queued", "digits": digits}}, nil
}
