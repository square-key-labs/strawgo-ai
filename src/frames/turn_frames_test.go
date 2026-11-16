package frames

import (
	"context"
	"errors"
	"fmt"
	"net"
	"testing"
)

func TestUserMuteFramesMetadata(t *testing.T) {
	started := NewUserMuteStartedFrame()
	if started.Name() != "UserMuteStartedFrame" {
		t.Fatalf("expected UserMuteStartedFrame name, got %q", started.Name())
	}
	if started.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", started.Category())
	}

	stopped := NewUserMuteStoppedFrame()
	if stopped.Name() != "UserMuteStoppedFrame" {
		t.Fatalf("expected UserMuteStoppedFrame name, got %q", stopped.Name())
	}
	if stopped.Category() != SystemCategory {
		t.Fatalf("expected SystemCategory, got %v", stopped.Category())
	}
}

func TestTTSContextIDFields(t *testing.T) {
	started := NewTTSStartedFrameWithContext("ctx-start")
	if started.Name() != "TTSStartedFrame" {
		t.Fatalf("expected TTSStartedFrame name, got %q", started.Name())
	}
	if started.Category() != ControlCategory {
		t.Fatalf("expected ControlCategory, got %v", started.Category())
	}
	if started.ContextID != "ctx-start" {
		t.Fatalf("expected started context ID to be ctx-start, got %q", started.ContextID)
	}

	stopped := NewTTSStoppedFrame()
	stopped.ContextID = "ctx-stop"
	if stopped.Name() != "TTSStoppedFrame" {
		t.Fatalf("expected TTSStoppedFrame name, got %q", stopped.Name())
	}
	if stopped.Category() != ControlCategory {
		t.Fatalf("expected ControlCategory, got %v", stopped.Category())
	}
	if stopped.ContextID != "ctx-stop" {
		t.Fatalf("expected stopped context ID to be ctx-stop, got %q", stopped.ContextID)
	}

	audio := NewTTSAudioFrame([]byte{1, 2, 3}, 24000, 1)
	audio.ContextID = "ctx-audio"
	if audio.Name() != "TTSAudioFrame" {
		t.Fatalf("expected TTSAudioFrame name, got %q", audio.Name())
	}
	if audio.Category() != DataCategory {
		t.Fatalf("expected DataCategory, got %v", audio.Category())
	}
	if audio.ContextID != "ctx-audio" {
		t.Fatalf("expected audio context ID to be ctx-audio, got %q", audio.ContextID)
	}
}

type statusErr int

func (e statusErr) Error() string       { return fmt.Sprintf("HTTP %d", int(e)) }
func (e statusErr) HTTPStatusCode() int { return int(e) }

type categorizedErr struct{ statusErr }

func (categorizedErr) ErrorCategory() ErrorCategory { return ErrorCategoryContentFiltered }

func TestClassifyHTTPStatus(t *testing.T) {
	cases := map[int]ErrorCategory{
		400: ErrorCategoryInvalidRequest,
		401: ErrorCategoryAuthentication,
		402: ErrorCategoryQuota,
		403: ErrorCategoryAuthorization,
		404: ErrorCategoryInvalidRequest,
		408: ErrorCategoryUnknown,
		422: ErrorCategoryInvalidRequest,
		429: ErrorCategoryRateLimit,
		500: ErrorCategoryServer,
		503: ErrorCategoryServer,
		200: ErrorCategoryUnknown,
	}
	for code, want := range cases {
		if got := ClassifyHTTPStatus(code); got != want {
			t.Errorf("ClassifyHTTPStatus(%d) = %q, want %q", code, got, want)
		}
	}
}

func TestClassifyError(t *testing.T) {
	dial := &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
	dns := &net.DNSError{Err: "no such host", Name: "api.invalid", IsNotFound: true}
	cases := []struct {
		name string
		err  error
		want ErrorCategory
	}{
		{"nil", nil, ErrorCategoryUnknown},
		{"plain", errors.New("boom"), ErrorCategoryUnknown},
		{"canceled", context.Canceled, ErrorCategoryUnknown},
		{"deadline", fmt.Errorf("request: %w", context.DeadlineExceeded), ErrorCategoryConnectivity},
		{"dial", fmt.Errorf("post: %w", dial), ErrorCategoryConnectivity},
		{"dns", dns, ErrorCategoryConnectivity},
		{"wrapped 401", fmt.Errorf("generation failed: %w", statusErr(401)), ErrorCategoryAuthentication},
		{"wrapped 403", fmt.Errorf("x: %w", statusErr(403)), ErrorCategoryAuthorization},
		{"wrapped 429", fmt.Errorf("x: %w", statusErr(429)), ErrorCategoryRateLimit},
		{"wrapped 502", fmt.Errorf("x: %w", statusErr(502)), ErrorCategoryServer},
		{"self-categorized", fmt.Errorf("x: %w", categorizedErr{statusErr(400)}), ErrorCategoryContentFiltered},
	}
	for _, tc := range cases {
		if got := ClassifyError(tc.err); got != tc.want {
			t.Errorf("%s: ClassifyError = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestErrorCategoryPermanence(t *testing.T) {
	permanent := map[ErrorCategory]bool{
		ErrorCategoryAuthentication:  true,
		ErrorCategoryAuthorization:   true,
		ErrorCategoryQuota:           true,
		ErrorCategoryConfiguration:   true,
		ErrorCategoryInvalidRequest:  false,
		ErrorCategoryRateLimit:       false,
		ErrorCategoryConnectivity:    false,
		ErrorCategoryServer:          false,
		ErrorCategoryApplication:     false,
		ErrorCategoryContentFiltered: false,
		ErrorCategoryUnknown:         false,
		"":                           false,
	}
	for c, want := range permanent {
		if c.IsPermanent() != want {
			t.Errorf("%q.IsPermanent() = %v, want %v", c, !want, want)
		}
	}
	if f := NewErrorFrame(errors.New("x")); f.Category != "" {
		t.Errorf("NewErrorFrame category = %q, want unset", f.Category)
	}
	if f := NewCategorizedErrorFrame(errors.New("x"), ErrorCategoryRateLimit); f.Category != ErrorCategoryRateLimit {
		t.Errorf("NewCategorizedErrorFrame category = %q", f.Category)
	}
}
