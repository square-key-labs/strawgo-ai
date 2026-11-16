package strawtest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/services/llmservice"
)

const DefaultStepTimeout = 10 * time.Second

type Bot struct {
	Pipeline *pipeline.Pipeline
	LLM      *llmservice.Service
	Config   *pipeline.PipelineTaskConfig
}

type Script struct {
	Build       func() Bot
	Steps       []Step
	Judge       Judge
	StepTimeout time.Duration
}

type Step interface {
	describe() string
	run(s *session) StepResult
}

type Matcher struct {
	desc  string
	match func(reply string) bool
}

func (m Matcher) String() string { return m.desc }

func normalizeSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func Contains(text string) Matcher {
	return Matcher{
		desc:  fmt.Sprintf("contains %q", text),
		match: func(r string) bool { return strings.Contains(normalizeSpace(r), normalizeSpace(text)) },
	}
}

func NotContains(text string) Matcher {
	return Matcher{
		desc:  fmt.Sprintf("does not contain %q", text),
		match: func(r string) bool { return !strings.Contains(normalizeSpace(r), normalizeSpace(text)) },
	}
}

func MatchesRegexp(expr string) Matcher {
	re, err := regexp.Compile(expr)
	if err != nil {
		return Matcher{desc: fmt.Sprintf("matches %q (invalid: %v)", expr, err), match: func(string) bool { return false }}
	}
	return Matcher{desc: fmt.Sprintf("matches /%s/", expr), match: re.MatchString}
}

func Satisfies(desc string, fn func(reply string) bool) Matcher {
	return Matcher{desc: desc, match: fn}
}

type ArgsMatcher struct {
	desc  string
	match func(args map[string]any) bool
}

func (m ArgsMatcher) String() string { return m.desc }

func ArgsInclude(want map[string]any) ArgsMatcher {
	return ArgsMatcher{
		desc: "arguments including " + encodeValue(want),
		match: func(args map[string]any) bool {
			for k, v := range want {
				got, ok := args[k]
				if !ok || !sameJSON(got, v) {
					return false
				}
			}
			return true
		},
	}
}

func ArgsSatisfy(desc string, fn func(args map[string]any) bool) ArgsMatcher {
	return ArgsMatcher{desc: desc, match: fn}
}

func sameJSON(a, b any) bool {
	ea, errA := json.Marshal(a)
	eb, errB := json.Marshal(b)
	if errA != nil || errB != nil {
		return false
	}
	var na, nb any
	if json.Unmarshal(ea, &na) != nil || json.Unmarshal(eb, &nb) != nil {
		return false
	}
	ca, _ := json.Marshal(na)
	cb, _ := json.Marshal(nb)
	return string(ca) == string(cb)
}

type sayStep struct{ text string }

func Say(text string) Step { return sayStep{text: text} }

func (s sayStep) describe() string { return fmt.Sprintf("user says %q", s.text) }

func (s sayStep) run(ss *session) StepResult {
	ss.conv.userSays(s.text)
	timer := time.NewTimer(ss.timeout)
	defer timer.Stop()
	if err := ss.sendAndWait(frames.NewUserStartedSpeakingFrame(), frames.Downstream, timer.C); err != nil {
		return StepResult{Status: StepFailed, Reason: err.Error()}
	}
	if err := ss.task.QueueFrame(frames.NewTranscriptionFrame(s.text, true)); err != nil {
		return StepResult{Status: StepFailed, Reason: fmt.Sprintf("queueing the transcription: %v", err)}
	}
	if err := ss.sendAndWait(newBarrier(), frames.Downstream, timer.C); err != nil {
		return StepResult{Status: StepFailed, Reason: err.Error()}
	}
	if err := ss.task.QueueFrame(frames.NewUserStoppedSpeakingFrame()); err != nil {
		return StepResult{Status: StepFailed, Reason: fmt.Sprintf("queueing UserStoppedSpeakingFrame: %v", err)}
	}
	return StepResult{Status: StepPassed}
}

type replyStep struct{ matchers []Matcher }

func ExpectReply(matchers ...Matcher) Step { return replyStep{matchers: matchers} }

func (s replyStep) expected() string {
	descs := make([]string, len(s.matchers))
	for i, m := range s.matchers {
		descs[i] = m.desc
	}
	if len(descs) == 0 {
		return "any reply"
	}
	return "reply " + strings.Join(descs, ", ")
}

func (s replyStep) describe() string { return "bot " + s.expected() }

func (s replyStep) run(ss *session) StepResult {
	res := StepResult{Expected: s.expected()}
	finished, err := ss.awaitTurn(nil)
	reply := ss.conv.finishTurn()
	res.Got = fmt.Sprintf("%q", reply)
	if err != nil {
		res.Status, res.Reason = StepFailed, err.Error()
		return res
	}
	if !finished {
		res.Status, res.Reason = StepFailed, "the bot turn did not finish"
		return res
	}
	for _, m := range s.matchers {
		if !m.match(reply) {
			res.Status, res.Reason = StepFailed, "reply does not satisfy: "+m.desc
			return res
		}
	}
	res.Status = StepPassed
	return res
}

type CallStep struct {
	name      string
	args      *ArgsMatcher
	result    any
	hasResult bool
}

func ExpectCall(name string, args ...ArgsMatcher) *CallStep {
	s := &CallStep{name: name}
	if len(args) > 0 {
		s.args = &args[0]
	}
	return s
}

func (s *CallStep) Returning(result any) *CallStep {
	s.result, s.hasResult = result, true
	return s
}

func (s *CallStep) expected() string {
	out := "call " + s.name
	if s.args != nil {
		out += " with " + s.args.desc
	}
	if s.hasResult {
		out += ", returning " + encodeValue(s.result)
	}
	return out
}

func (s *CallStep) describe() string { return "bot " + s.expected() }

func (s *CallStep) run(ss *session) StepResult {
	res := StepResult{Expected: s.expected()}
	var call *observedCall
	_, err := ss.awaitTurn(func(t *botTurn) bool {
		if t.taken < len(t.calls) {
			call = t.calls[t.taken]
			t.taken++
			return true
		}
		return false
	})
	if call == nil {
		res.Got = "no function call"
		if reply := ss.conv.currentReply(); reply != "" {
			res.Got += fmt.Sprintf("; bot said %q", reply)
		}
		res.Status = StepFailed
		res.Reason = "the bot finished its turn without calling " + s.name
		if err != nil {
			res.Reason = err.Error()
		}
		return res
	}
	res.Got = callSignature(call.name, call.args)
	switch {
	case call.name != s.name:
		res.Status, res.Reason = StepFailed, fmt.Sprintf("the bot called %s instead of %s", call.name, s.name)
	case s.args != nil && !s.args.match(call.args):
		res.Status, res.Reason = StepFailed, "arguments do not satisfy: "+s.args.desc
	default:
		res.Status = StepPassed
	}
	return res
}

type waitStep struct{}

func WaitForBot() Step { return waitStep{} }

func (waitStep) describe() string { return "bot finishes its turn" }

func (waitStep) run(ss *session) StepResult {
	res := StepResult{Expected: "the bot turn to finish"}
	finished, err := ss.awaitTurn(nil)
	res.Got = fmt.Sprintf("%q", ss.conv.finishTurn())
	switch {
	case err != nil:
		res.Status, res.Reason = StepFailed, err.Error()
	case !finished:
		res.Status, res.Reason = StepFailed, "the bot turn did not finish"
	default:
		res.Status = StepPassed
	}
	return res
}

type judgeStep struct{ rubric string }

func ExpectJudge(rubric string) Step { return judgeStep{rubric: rubric} }

func (s judgeStep) describe() string { return fmt.Sprintf("judge passes %q", s.rubric) }

func (s judgeStep) run(ss *session) StepResult {
	res := StepResult{Expected: s.rubric}
	if ss.script.Judge == nil {
		res.Status, res.Reason, res.Got = StepFailed, "Script.Judge is nil", "no verdict"
		return res
	}
	if _, err := ss.awaitTurn(nil); err != nil {
		res.Status, res.Reason, res.Got = StepFailed, err.Error(), "no verdict"
		return res
	}
	ss.conv.finishTurn()
	ctx, cancel := context.WithTimeout(ss.ctx, ss.timeout)
	defer cancel()
	v, err := ss.script.Judge.Judge(ctx, ss.conv.snapshotTranscript(), s.rubric)
	if err != nil {
		res.Status, res.Reason, res.Got = StepFailed, "judge error: "+err.Error(), "no verdict"
		return res
	}
	res.Got = "fail"
	if v.Passed {
		res.Got = "pass"
	}
	if v.Reason != "" {
		res.Got += ": " + v.Reason
	}
	if !v.Passed {
		res.Status, res.Reason = StepFailed, "the judge did not pass the conversation"
		return res
	}
	res.Status = StepPassed
	return res
}

type session struct {
	script  Script
	ctx     context.Context
	timeout time.Duration
	task    *pipeline.PipelineTask
	conv    *conversation

	awaitMu sync.Mutex
	awaited map[frames.Frame]chan struct{}
}

var errTimeout = errors.New("timed out")

func (s *session) awaitTurn(ready func(*botTurn) bool) (bool, error) {
	timer := time.NewTimer(s.timeout)
	defer timer.Stop()
	c := s.conv
	for {
		c.mu.Lock()
		if ready != nil && ready(c.turn) {
			c.mu.Unlock()
			return false, nil
		}
		settled, gen, ch := c.turn.settled(), c.activity, c.changed
		c.mu.Unlock()

		if settled {
			if err := s.sendAndWait(newBarrier(), frames.Upstream, timer.C); err != nil {
				return false, err
			}
			c.mu.Lock()
			if ready != nil && ready(c.turn) {
				c.mu.Unlock()
				return false, nil
			}
			same := c.activity == gen && c.turn.settled()
			c.mu.Unlock()
			if same {
				return true, nil
			}
			continue
		}

		select {
		case <-ch:
		case <-timer.C:
			return false, fmt.Errorf("%w after %v waiting for the bot to finish its turn", errTimeout, s.timeout)
		case <-s.ctx.Done():
			return false, s.ctx.Err()
		case <-s.task.Done():
			return false, errors.New("the pipeline stopped before the bot turn finished")
		}
	}
}

func (s *session) sendAndWait(f frames.Frame, dir frames.FrameDirection, deadline <-chan time.Time) error {
	arrived := make(chan struct{})
	s.awaitMu.Lock()
	s.awaited[f] = arrived
	s.awaitMu.Unlock()
	if err := s.task.QueueFrame(f, dir); err != nil {
		return fmt.Errorf("queueing %s: %v", f.Name(), err)
	}
	select {
	case <-arrived:
		return nil
	case <-deadline:
		return fmt.Errorf("%w after %v: a %s sent %s never reached the other end of the pipeline; a processor is not forwarding frames", errTimeout, s.timeout, f.Name(), dir)
	case <-s.ctx.Done():
		return s.ctx.Err()
	case <-s.task.Done():
		return errors.New("the pipeline stopped")
	}
}

func (s *session) onFrame(f frames.Frame) {
	s.awaitMu.Lock()
	ch, awaited := s.awaited[f]
	delete(s.awaited, f)
	s.awaitMu.Unlock()
	if awaited {
		close(ch)
	}
	if _, ok := f.(*barrierFrame); !ok {
		s.conv.observe(f)
	}
}

func (s Script) Run(ctx context.Context) Result {
	if s.Build == nil {
		return Result{Err: errors.New("Script.Build is nil")}
	}
	bot := s.Build()
	if bot.Pipeline == nil {
		return Result{Err: errors.New("Build returned a Bot without a Pipeline")}
	}
	timeout := s.StepTimeout
	if timeout <= 0 {
		timeout = DefaultStepTimeout
	}

	canned := map[string][]any{}
	for _, step := range s.Steps {
		if c, ok := step.(*CallStep); ok && c.hasResult {
			canned[c.name] = append(canned[c.name], c.result)
		}
	}
	if len(canned) > 0 && bot.LLM == nil {
		return Result{Err: errors.New("a step uses Returning but Build returned a Bot without LLM")}
	}
	var cannedMu sync.Mutex
	for name := range canned {
		bot.LLM.RegisterFunction(name, func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
			cannedMu.Lock()
			defer cannedMu.Unlock()
			queue := canned[p.FunctionName]
			if len(queue) == 0 {
				return llm.FunctionResult{}, fmt.Errorf("strawtest: no canned result left for %s", p.FunctionName)
			}
			canned[p.FunctionName] = queue[1:]
			return llm.FunctionResult{Value: queue[0]}, nil
		})
	}

	cfg := bot.Config
	if cfg == nil {
		cfg = pipeline.DefaultPipelineTaskConfig()
		cfg.CancelOnIdleTimeout = false
	}
	ss := &session{
		script:  s,
		ctx:     ctx,
		timeout: timeout,
		task:    pipeline.NewPipelineTaskWithConfig(bot.Pipeline, cfg),
		conv:    newConversation(),
		awaited: map[frames.Frame]chan struct{}{},
	}
	ss.task.OnFrameReachedDownstream(ss.onFrame)
	ss.task.OnFrameReachedUpstream(ss.onFrame)
	started := make(chan struct{})
	var startOnce sync.Once
	ss.task.OnStarted(func() { startOnce.Do(func() { close(started) }) })

	done := make(chan error, 1)
	go func() { done <- ss.task.Run(context.WithoutCancel(ctx)) }()

	res := Result{}
	startTimer := time.NewTimer(timeout)
	select {
	case <-started:
	case <-startTimer.C:
		res.Err = fmt.Errorf("the pipeline did not start within %v", timeout)
	case <-ctx.Done():
		res.Err = ctx.Err()
	}
	startTimer.Stop()

	failed := res.Err != nil
	for i, step := range s.Steps {
		var r StepResult
		if failed {
			r = StepResult{Status: StepSkipped}
		} else {
			r = step.run(ss)
		}
		r.Index, r.Step = i, step.describe()
		res.Steps = append(res.Steps, r)
		if r.Status == StepFailed {
			failed = true
		}
	}
	ss.conv.finishTurn()
	res.Transcript = ss.conv.snapshotTranscript()

	_ = ss.task.QueueFrame(frames.NewEndFrame())
	select {
	case <-done:
	case <-time.After(timeout):
		ss.task.Cancel()
		select {
		case <-done:
		case <-time.After(cfg.CancelTimeout + time.Second):
		}
	}
	return res
}

func RunScript(tb testing.TB, s Script) Result {
	tb.Helper()
	res := s.Run(context.Background())
	if !res.Passed() {
		tb.Errorf("%s", res)
	}
	return res
}
