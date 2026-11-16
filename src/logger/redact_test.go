package logger_test

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/square-key-labs/strawgo-ai/src/logger"
)

func TestRedactMasksCredentials(t *testing.T) {
	h := http.Header{}
	h.Set("Authorization", "Bearer abcdef123456")
	h.Set("Xi-Api-Key", "k123")
	cases := map[string]string{
		`Get "https://api.example.com/v1/live?model=a&api_key=SECRET1&key=S2&lang=en": refused`: `Get "https://api.example.com/v1/live?model=a&api_key=[REDACTED]&key=[REDACTED]&lang=en": refused`,
		"wss://h/p?access_token=S3&xi-api-key=S4&X-Amz-Signature=S6":                            "wss://h/p?access_token=[REDACTED]&xi-api-key=[REDACTED]&X-Amz-Signature=[REDACTED]",
		"Authorization: Bearer S8longtoken":                                                     "Authorization: Bearer [REDACTED]",
		"sending Bearer abcdefghij to server":                                                   "sending Bearer [REDACTED] to server",
		"sending Basic dXNlcjpwYXNz to server":                                                  "sending Basic [REDACTED] to server",
		fmt.Sprintf("headers %v", h):                                                            "headers map[Authorization:[Bearer [REDACTED]] Xi-Api-Key:[[REDACTED]]]",
		`HTTP 401: {"error":"bad","api_key":"S9","detail":"x"}`:                                 `HTTP 401: {"error":"bad","api_key":"[REDACTED]","detail":"x"}`,
		`{"token":"abc123def456"}`:                                                              `{"token":"[REDACTED]"}`,
		"retry with token=abc123def456":                                                         "retry with token=[REDACTED]",
		"postgres://user:hunter2@db:5432/x":                                                     "postgres://user:[REDACTED]@db:5432/x",
		"use sk-abcdefghijklmnopqrstuvwxyz now":                                                 "use [REDACTED] now",
		"google AIzaSyA1234567890abcdefghijklmnopqrstuv":                                        "google [REDACTED]",
		"aws AKIAABCDEFGHIJKLMNOP":                                                              "aws [REDACTED]",
		"jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc":                              "jwt [REDACTED]",
	}
	for in, want := range cases {
		if got := logger.Redact(in); got != want {
			t.Errorf("Redact(%q)\n got %q\nwant %q", in, got, want)
		}
	}
}

func TestRedactLeavesProseAlone(t *testing.T) {
	for _, in := range []string{
		"connection lost: read tcp 1.2.3.4:5->6.7.8.9:443: i/o timeout",
		"[STT] keyterm=hello max_tokens=5 model=nova-3",
		"vertex: token: oauth2: cannot fetch",
		"refresh failed: key: invalid format",
		"turn complete through high latency",
	} {
		if got := logger.Redact(in); got != in {
			t.Errorf("Redact(%q) = %q", in, got)
		}
	}
}

func TestRedactIsIdempotent(t *testing.T) {
	in := `GET https://h/p?api_key=abc123&x=1 Authorization: Bearer abcdefghij {"token":"t0k3n"}`
	once := logger.Redact(in)
	if twice := logger.Redact(once); twice != once {
		t.Fatalf("once %q\ntwice %q", once, twice)
	}
}

func TestRedactErrorKeepsChain(t *testing.T) {
	base := &fs.PathError{Op: "open", Path: "x", Err: fs.ErrNotExist}
	err := logger.RedactError(fmt.Errorf("fetch https://h/p?key=abc123: %w", base))
	if strings.Contains(err.Error(), "abc123") || !strings.Contains(err.Error(), "key=[REDACTED]") {
		t.Fatalf("err = %q", err)
	}
	var pe *fs.PathError
	if !errors.Is(err, fs.ErrNotExist) || !errors.As(err, &pe) || pe != base {
		t.Fatalf("chain lost: %v", err)
	}
	plain := errors.New("nothing secret")
	if logger.RedactError(plain) != plain || logger.RedactError(nil) != nil {
		t.Fatal("clean errors must pass through unchanged")
	}
}

func TestAddRedactorAppliesToLogLinesConcurrently(t *testing.T) {
	var buf bytes.Buffer
	log := logger.New(logger.DEBUG, &buf, false, "").WithPrefix("Call")
	phone := regexp.MustCompile(`\+\d{10,15}`)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			logger.AddRedactor(func(s string) string { return phone.ReplaceAllString(s, "+[PHONE]") })
		}()
		go func() {
			defer wg.Done()
			log.Debug("ping %d", 1)
		}()
	}
	wg.Wait()
	log.Info("call from +14155550123 via wss://h/p?token=abc123")
	out := buf.String()
	if strings.Contains(out, "14155550123") || strings.Contains(out, "abc123") || !strings.Contains(out, "call from +[PHONE] via wss://h/p?token=[REDACTED]") {
		t.Fatalf("log output = %q", out)
	}
}

func BenchmarkRedactPlainLine(b *testing.B) {
	for b.Loop() {
		logger.Redact("[STT] transcript final: hello there how are you doing today, latency 120ms")
	}
}
