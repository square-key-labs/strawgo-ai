package strawtest

import (
	"context"
	"fmt"
	"strings"
)

type Role string

const (
	RoleUser           Role = "user"
	RoleBot            Role = "bot"
	RoleFunctionCall   Role = "function call"
	RoleFunctionResult Role = "function result"
)

type Message struct {
	Role Role
	Text string
}

type Verdict struct {
	Passed bool
	Reason string
}

type Judge interface {
	Judge(ctx context.Context, transcript []Message, rubric string) (Verdict, error)
}

type JudgeFunc func(ctx context.Context, transcript []Message, rubric string) (Verdict, error)

func (f JudgeFunc) Judge(ctx context.Context, transcript []Message, rubric string) (Verdict, error) {
	return f(ctx, transcript, rubric)
}

type Status string

const (
	StepPassed  Status = "passed"
	StepFailed  Status = "failed"
	StepSkipped Status = "skipped"
)

type StepResult struct {
	Index    int
	Step     string
	Status   Status
	Expected string
	Got      string
	Reason   string
}

type Result struct {
	Steps      []StepResult
	Transcript []Message
	Err        error
}

func (r Result) Passed() bool {
	if r.Err != nil {
		return false
	}
	for _, s := range r.Steps {
		if s.Status != StepPassed {
			return false
		}
	}
	return true
}

func (r Result) String() string {
	var b strings.Builder
	failed := 0
	for _, s := range r.Steps {
		if s.Status == StepFailed {
			failed++
		}
	}
	switch {
	case r.Err != nil:
		fmt.Fprintf(&b, "script did not run: %v\n", r.Err)
	case r.Passed():
		fmt.Fprintf(&b, "script passed: %d steps\n", len(r.Steps))
	default:
		fmt.Fprintf(&b, "script failed: %d of %d steps failed\n", failed, len(r.Steps))
	}
	b.WriteString("transcript:\n")
	if len(r.Transcript) == 0 {
		b.WriteString("  (empty)\n")
	}
	for _, m := range r.Transcript {
		fmt.Fprintf(&b, "  %s: %s\n", m.Role, m.Text)
	}
	b.WriteString("steps:\n")
	for _, s := range r.Steps {
		mark := "ok  "
		switch s.Status {
		case StepFailed:
			mark = "FAIL"
		case StepSkipped:
			mark = "skip"
		}
		fmt.Fprintf(&b, "  %s %d. %s\n", mark, s.Index+1, s.Step)
		if s.Status != StepFailed {
			continue
		}
		if s.Reason != "" {
			fmt.Fprintf(&b, "         reason:   %s\n", s.Reason)
		}
		if s.Expected != "" {
			fmt.Fprintf(&b, "         expected: %s\n", s.Expected)
		}
		fmt.Fprintf(&b, "         got:      %s\n", s.Got)
	}
	return strings.TrimRight(b.String(), "\n")
}
