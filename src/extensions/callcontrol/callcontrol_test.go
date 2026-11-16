package callcontrol

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/serializers"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

var (
	_ Registrar   = (*llmservice.Service)(nil)
	_ FrameQueuer = (*pipeline.PipelineTask)(nil)
)

type fakeRegistrar struct {
	handlers map[string]llm.FunctionHandler
	opts     map[string]llm.FunctionOptions
}

func (r *fakeRegistrar) RegisterFunction(name string, h llm.FunctionHandler, opts ...llm.FunctionOptions) {
	if r.handlers == nil {
		r.handlers = map[string]llm.FunctionHandler{}
		r.opts = map[string]llm.FunctionOptions{}
	}
	r.handlers[name] = h
	if len(opts) > 0 {
		r.opts[name] = opts[0]
	}
}

func (r *fakeRegistrar) call(t *testing.T, name string, args map[string]any) map[string]any {
	t.Helper()
	h, ok := r.handlers[name]
	if !ok {
		t.Fatalf("%s not registered", name)
	}
	res, err := h(context.Background(), llm.FunctionCallParams{FunctionName: name, Arguments: args})
	if err != nil {
		t.Fatalf("%s returned error %v", name, err)
	}
	v, _ := res.Value.(map[string]any)
	return v
}

type fakeQueue struct {
	mu          sync.Mutex
	frames      []frames.Frame
	transferErr error
	silent      bool
}

func (q *fakeQueue) QueueFrame(f frames.Frame, _ ...frames.FrameDirection) error {
	q.mu.Lock()
	q.frames = append(q.frames, f)
	err, silent := q.transferErr, q.silent
	q.mu.Unlock()
	if tf, ok := f.(*frames.TransferCallFrame); ok && !silent {
		go tf.OnComplete(err)
	}
	return nil
}

func (q *fakeQueue) names() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	var out []string
	for _, f := range q.frames {
		out = append(out, f.Name())
	}
	return out
}

func (q *fakeQueue) transfers() []*frames.TransferCallFrame {
	q.mu.Lock()
	defer q.mu.Unlock()
	var out []*frames.TransferCallFrame
	for _, f := range q.frames {
		if tf, ok := f.(*frames.TransferCallFrame); ok {
			out = append(out, tf)
		}
	}
	return out
}

var sales = Destination{Name: "sales", Description: "the sales team", Target: "+14155550100", CallerID: "+14155550199"}

func register(t *testing.T, cfg Config) (*fakeRegistrar, *fakeQueue, []llm.FunctionSchema) {
	t.Helper()
	r, q := &fakeRegistrar{}, &fakeQueue{}
	schemas, err := Register(r, q, cfg)
	if err != nil {
		t.Fatal(err)
	}
	return r, q, schemas
}

func TestRegisterExposesOnlyEnabledTools(t *testing.T) {
	r, _, schemas := register(t, Config{Destinations: []Destination{sales}, EndCall: true, SendDTMF: true})
	if len(schemas) != 3 || len(r.handlers) != 3 {
		t.Fatalf("schemas=%d handlers=%d", len(schemas), len(r.handlers))
	}
	prop := schemas[0].Properties["destination"].(map[string]any)
	if enum := prop["enum"].([]string); len(enum) != 1 || enum[0] != "sales" {
		t.Fatalf("enum = %v", prop["enum"])
	}
	if strings.Contains(prop["description"].(string), sales.Target) {
		t.Fatal("schema leaks the destination number to the model")
	}
	if r.opts[TransferCallTool].CancelOnInterruption {
		t.Fatal("transfer_call cancels on interruption")
	}
	r, _, schemas = register(t, Config{EndCall: true})
	if len(schemas) != 1 || schemas[0].Name != EndCallTool || len(r.handlers) != 1 {
		t.Fatalf("schemas = %+v", schemas)
	}
	if _, err := Register(&fakeRegistrar{}, &fakeQueue{}, Config{}); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestRegisterRejectsInvalidDestinations(t *testing.T) {
	for _, d := range []Destination{
		{Name: "sales", Target: "4155550100"},
		{Name: "", Target: "+14155550100"},
		{Name: "x", Target: "+14155550100", CallerID: "unknown"},
		{Name: "4155550100", Target: "+14155550100"},
		{Name: "x", Target: "+14155550100", Announcement: strings.Repeat("a", 1001)},
	} {
		if _, err := Register(&fakeRegistrar{}, &fakeQueue{}, Config{Destinations: []Destination{d}}); err == nil {
			t.Fatalf("%+v accepted", d)
		}
	}
	if _, err := Register(&fakeRegistrar{}, &fakeQueue{}, Config{Destinations: []Destination{sales, {Name: "SALES", Target: "+14155550101"}}}); err == nil {
		t.Fatal("duplicate name accepted")
	}
}

func TestTransferCallToAllowedDestination(t *testing.T) {
	r, q, _ := register(t, Config{Destinations: []Destination{sales}, TransferMessage: "Putting you through."})
	h := r.handlers[TransferCallTool]
	res, err := h(context.Background(), llm.FunctionCallParams{Arguments: map[string]any{"destination": "Sales"}})
	if err != nil {
		t.Fatal(err)
	}
	if v := res.Value.(map[string]any); v["status"] != "transfer_accepted" {
		t.Fatalf("result = %v", v)
	}
	if res.RunLLM == nil || *res.RunLLM {
		t.Fatal("successful transfer should not run the LLM again")
	}
	if got := q.names(); len(got) != 2 || got[0] != "TTSSpeakFrame" || got[1] != "TransferCallFrame" {
		t.Fatalf("queued = %v", got)
	}
	tf := q.transfers()[0]
	if tf.Destination != sales.Target || tf.CallerID != sales.CallerID {
		t.Fatalf("transfer frame = %+v", tf)
	}
}

func TestTransferCallRefusesUnlistedNumber(t *testing.T) {
	r, q, _ := register(t, Config{Destinations: []Destination{sales}})
	for _, dest := range []string{"+19005550100", "sip:attacker@evil.example.com", "", "support"} {
		v := r.call(t, TransferCallTool, map[string]any{"destination": dest})
		if v["status"] != "failed" || v["error"] == "" {
			t.Fatalf("%q: result = %v", dest, v)
		}
	}
	if n := len(q.names()); n != 0 {
		t.Fatalf("refused transfers queued %d frames", n)
	}
}

func TestTransferCallResolverIsValidated(t *testing.T) {
	resolver := func(_ context.Context, name string) (Destination, bool) {
		switch name {
		case "agent-7":
			return Destination{Name: name, Target: "sip:agent7@pbx.example.com"}, true
		case "broken":
			return Destination{Name: name, Target: "call me maybe"}, true
		}
		return Destination{}, false
	}
	var mu sync.Mutex
	var seen []string
	record := func(ctx context.Context, name string) (Destination, bool) {
		mu.Lock()
		seen = append(seen, name)
		mu.Unlock()
		return resolver(ctx, name)
	}
	r, q, schemas := register(t, Config{Resolver: record})
	if _, hasEnum := schemas[0].Properties["destination"].(map[string]any)["enum"]; hasEnum {
		t.Fatal("resolver-only schema has an enum")
	}
	if v := r.call(t, TransferCallTool, map[string]any{"destination": "agent-7"}); v["status"] != "transfer_accepted" {
		t.Fatalf("resolved = %v", v)
	}
	for _, dest := range []string{"broken", "+14155550100"} {
		if v := r.call(t, TransferCallTool, map[string]any{"destination": dest}); v["status"] != "failed" {
			t.Fatalf("%q: result = %v", dest, v)
		}
	}
	for _, dest := range []string{"4155550100", "415 555 0100", "sip:attacker@evil.example.com", "agent@evil", "+1 415 555", "a\nb"} {
		if v := r.call(t, TransferCallTool, map[string]any{"destination": dest}); v["status"] != "failed" {
			t.Fatalf("%q: result = %v", dest, v)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 || seen[0] != "agent-7" || seen[1] != "broken" {
		t.Fatalf("resolver called with %q", seen)
	}
	if n := len(q.transfers()); n != 1 {
		t.Fatalf("transfers queued = %d", n)
	}
}

func TestTransferCallReportsFailure(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{serializers.ErrTransferUnsupported, "not available"},
		{serializers.ErrCallEnded, "already ended"},
		{errors.New("status 500: token=abc"), "failed"},
		{serializers.ErrTransferInProgress, "already in progress"},
		{fmt.Errorf("twilio transfer: %w", serializers.ErrTransferOutcomeUnknown), "do not retry"},
	}
	for _, tc := range cases {
		r, q, _ := register(t, Config{Destinations: []Destination{sales}})
		q.transferErr = tc.err
		v := r.call(t, TransferCallTool, map[string]any{"destination": "sales"})
		msg, _ := v["error"].(string)
		if v["status"] != "failed" || !strings.Contains(msg, tc.want) || strings.Contains(msg, "token") {
			t.Fatalf("%v: result = %v", tc.err, v)
		}
	}
}

func TestTransferCallTimesOut(t *testing.T) {
	r, q, _ := register(t, Config{Destinations: []Destination{sales}, TransferTimeout: 20 * time.Millisecond})
	q.silent = true
	v := r.call(t, TransferCallTool, map[string]any{"destination": "sales"})
	if v["status"] != "failed" || !strings.Contains(v["error"].(string), "cancelled") {
		t.Fatalf("result = %v", v)
	}
	tf := q.transfers()[0]
	if tf.Claim() {
		t.Fatal("a transport reaching the frame after the tool gave up could still transfer the call")
	}
}

func TestTransferCallTimeoutWaitsForStartedTransfer(t *testing.T) {
	r, q, _ := register(t, Config{Destinations: []Destination{sales}, TransferTimeout: 20 * time.Millisecond})
	q.silent = true
	result := make(chan map[string]any, 1)
	go func() { result <- r.call(t, TransferCallTool, map[string]any{"destination": "sales"}) }()
	var tf *frames.TransferCallFrame
	deadline := time.Now().Add(2 * time.Second)
	for tf == nil && time.Now().Before(deadline) {
		if list := q.transfers(); len(list) > 0 {
			tf = list[0]
		}
		time.Sleep(time.Millisecond)
	}
	if tf == nil || !tf.Claim() {
		t.Fatal("frame not queued or already cancelled")
	}
	select {
	case v := <-result:
		t.Fatalf("tool returned while the transfer was running: %v", v)
	case <-time.After(60 * time.Millisecond):
	}
	tf.Complete(nil)
	if v := <-result; v["status"] != "transfer_accepted" {
		t.Fatalf("result = %v", v)
	}
}

func TestEndCallSpeaksThenEnds(t *testing.T) {
	r, q, _ := register(t, Config{EndCall: true, GoodbyeMessage: "Goodbye."})
	h := r.handlers[EndCallTool]
	res, err := h(context.Background(), llm.FunctionCallParams{Arguments: map[string]any{"goodbye": "Thanks for calling!"}})
	if err != nil {
		t.Fatal(err)
	}
	if res.RunLLM == nil || *res.RunLLM {
		t.Fatal("end_call should not run the LLM again")
	}
	q.mu.Lock()
	got := append([]frames.Frame(nil), q.frames...)
	q.mu.Unlock()
	if len(got) != 2 {
		t.Fatalf("queued = %v", q.names())
	}
	speak, ok := got[0].(*frames.TTSSpeakFrame)
	if !ok || speak.Text != "Thanks for calling!" {
		t.Fatalf("first frame = %v", got[0].Name())
	}
	if _, ok := got[1].(*frames.EndFrame); !ok {
		t.Fatalf("second frame = %v", got[1].Name())
	}

	r, q, _ = register(t, Config{EndCall: true, GoodbyeMessage: "Goodbye."})
	r.call(t, EndCallTool, map[string]any{})
	if names := q.names(); len(names) != 2 || names[0] != "TTSSpeakFrame" || names[1] != "EndFrame" {
		t.Fatalf("default goodbye queued = %v", names)
	}

	r, q, _ = register(t, Config{EndCall: true})
	r.call(t, EndCallTool, nil)
	if names := q.names(); len(names) != 1 || names[0] != "EndFrame" {
		t.Fatalf("silent end queued = %v", names)
	}
}

func TestSendDTMFValidatesDigits(t *testing.T) {
	r, q, _ := register(t, Config{SendDTMF: true, MaxDTMFDigits: 4})
	if v := r.call(t, SendDTMFTool, map[string]any{"digits": "1 2#"}); v["status"] != "queued" || v["digits"] != "12#" {
		t.Fatalf("result = %v", v)
	}
	q.mu.Lock()
	dtmf, ok := q.frames[0].(*frames.OutputDTMFFrame)
	q.mu.Unlock()
	if !ok || frames.KeypadEntriesString(dtmf.Buttons) != "12#" {
		t.Fatalf("queued = %v", q.names())
	}
	for _, digits := range []any{"", "12a", "12345", "+1", 42} {
		if v := r.call(t, SendDTMFTool, map[string]any{"digits": digits}); v["status"] != "failed" {
			t.Fatalf("%v: result = %v", digits, v)
		}
	}
	if n := len(q.names()); n != 1 {
		t.Fatalf("invalid digits queued frames: %v", q.names())
	}
}

func TestEndCallGuardCanVeto(t *testing.T) {
	disclosed := false
	guard := func(context.Context) error {
		if !disclosed {
			return errors.New("read the recording disclosure first")
		}
		return nil
	}
	r, q, _ := register(t, Config{EndCall: true, GoodbyeMessage: "Bye.", EndCallGuard: guard})
	v := r.call(t, EndCallTool, map[string]any{})
	if v["status"] != "failed" || !strings.Contains(v["error"].(string), "disclosure") {
		t.Fatalf("vetoed result = %v", v)
	}
	if n := len(q.names()); n != 0 {
		t.Fatalf("vetoed end_call queued %v", q.names())
	}
	disclosed = true
	if v := r.call(t, EndCallTool, map[string]any{}); v["status"] != "ending" {
		t.Fatalf("allowed result = %v", v)
	}
	if names := q.names(); len(names) != 2 || names[1] != "EndFrame" {
		t.Fatalf("queued = %v", names)
	}
}
