package logger

import (
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
)

const Redacted = "[REDACTED]"

var (
	knownTokenRe = regexp.MustCompile(`\b(?:` +
		`sk-[A-Za-z0-9_-]{20,}` +
		`|(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}` +
		`|ABSK[A-Za-z0-9+/]{109,269}={0,2}` +
		`|bedrock-api-key-[A-Za-z0-9+/=_-]+` +
		`|AIza[0-9A-Za-z_-]{35}` +
		`|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{82,}` +
		`|xox[abeoprs]-[A-Za-z0-9-]{10,}|xapp-\d-[A-Za-z0-9-]{10,}` +
		`|(?:hf|api_org)_[A-Za-z]{34}` +
		`|pplx-[A-Za-z0-9]{48}` +
		`|SK[0-9a-fA-F]{32}` +
		`|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*` +
		`)` +
		`|-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----` +
		`(?:[\s\S]*?-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----|[\s\S]*)`)

	pairStartRe   = regexp.MustCompile(`["']?[A-Za-z][\w.-]*["']?\s*(?::=|=>|[:=])\s*`)
	schemeRe      = regexp.MustCompile(`^(?:Bearer|Basic|Token|Api-Token|Key|Digest|AWS4-HMAC-SHA256)\s+`)
	valueRe       = regexp.MustCompile(`^(?:"[^"\n]*"|'[^'\n]*'|["']?(?:\[REDACTED\]|[^\s,;&"'}\])\[])+)`)
	schemeOnlyRe  = regexp.MustCompile(`\b(Bearer|Basic)\s+[A-Za-z0-9._~+/-]{8,}=*`)
	urlPasswordRe = regexp.MustCompile(`(?i)(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s/@]+(@)`)
	camelRe       = regexp.MustCompile(`([a-z0-9])([A-Z])`)

	credentialWords = map[string]bool{
		"apikey": true, "auth": true, "authorization": true, "cookie": true,
		"credential": true, "credentials": true, "creds": true, "jwt": true,
		"key": true, "passwd": true, "password": true, "pwd": true,
		"secret": true, "sig": true, "signature": true, "token": true, "uploadid": true,
	}

	foldMarkers  = []string{"key", "auth", "token", "secret", "pass", "pwd", "sig", "cred", "cookie", "jwt", "bearer", "basic", "uploadid", "@"}
	exactMarkers = []string{"sk-", "AKIA", "ASIA", "ABIA", "ACCA", "A3T", "ABSK", "AIza", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "xox", "xapp-", "hf_", "api_org_", "pplx-", "SK", "eyJ"}
	foldByFirst  = byFirstByte(foldMarkers, true)
	exactByFirst = byFirstByte(exactMarkers, false)

	redactorsMu sync.Mutex
	redactors   atomic.Pointer[[]func(string) string]
)

func AddRedactor(fn func(string) string) {
	if fn == nil {
		return
	}
	redactorsMu.Lock()
	defer redactorsMu.Unlock()
	var next []func(string) string
	if cur := redactors.Load(); cur != nil {
		next = append(next, *cur...)
	}
	next = append(next, fn)
	redactors.Store(&next)
}

func Redact(s string) string {
	if hasMarker(s) {
		s = knownTokenRe.ReplaceAllString(s, Redacted)
		s = redactPairs(s)
		s = schemeOnlyRe.ReplaceAllString(s, "$1 "+Redacted)
		s = urlPasswordRe.ReplaceAllString(s, "${1}"+Redacted+"${2}")
	}
	if fns := redactors.Load(); fns != nil {
		for _, fn := range *fns {
			s = fn(s)
		}
	}
	return s
}

func RedactError(err error) error {
	if err == nil {
		return nil
	}
	msg := err.Error()
	if r := Redact(msg); r != msg {
		return &redactedError{msg: r, err: err}
	}
	return err
}

type redactedError struct {
	msg string
	err error
}

func (e *redactedError) Error() string { return e.msg }

func (e *redactedError) Unwrap() error { return e.err }

func hasMarker(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		for _, m := range exactByFirst[c] {
			if strings.HasPrefix(s[i:], m) {
				return true
			}
		}
		for _, m := range foldByFirst[lower(c)] {
			if hasPrefixFold(s[i:], m) {
				return true
			}
		}
	}
	return false
}

func byFirstByte(markers []string, fold bool) *[256][]string {
	var t [256][]string
	for _, m := range markers {
		c := m[0]
		if fold {
			c = lower(c)
		}
		t[c] = append(t[c], m)
	}
	return &t
}

func hasPrefixFold(s, lowerPrefix string) bool {
	if len(s) < len(lowerPrefix) {
		return false
	}
	for j := 0; j < len(lowerPrefix); j++ {
		if lower(s[j]) != lowerPrefix[j] {
			return false
		}
	}
	return true
}

func lower(c byte) byte {
	if 'A' <= c && c <= 'Z' {
		return c + 'a' - 'A'
	}
	return c
}

func isCredentialName(name string) bool {
	words := strings.ToLower(camelRe.ReplaceAllString(strings.Trim(name, `"'`), "${1}_${2}"))
	parts := strings.FieldsFunc(words, func(r rune) bool { return r == '-' || r == '_' || r == '.' })
	if credentialWords[strings.Join(parts, "")] {
		return true
	}
	for _, p := range parts {
		if credentialWords[p] {
			return true
		}
	}
	return false
}

func pairName(m string) string {
	end := strings.IndexAny(m, ":=")
	return strings.TrimSpace(m[:end])
}

func credentialValue(s string, start int) (valStart, valEnd int, ok bool) {
	p := start
	if strings.HasPrefix(s[p:], Redacted) {
		return 0, 0, false
	}
	if p < len(s) && s[p] == '[' {
		p++
	}
	if loc := schemeRe.FindStringIndex(s[p:]); loc != nil {
		p += loc[1]
	}
	loc := valueRe.FindStringIndex(s[p:])
	if loc == nil {
		return 0, 0, false
	}
	return p, p + loc[1], true
}

func redactPairs(s string) string {
	var b strings.Builder
	pos := 0
	for pos < len(s) {
		loc := pairStartRe.FindStringIndex(s[pos:])
		if loc == nil {
			break
		}
		mStart, mEnd := pos+loc[0], pos+loc[1]
		if !isCredentialName(pairName(s[mStart:mEnd])) {
			b.WriteString(s[pos:mEnd])
			pos = mEnd
			continue
		}
		vs, ve, ok := credentialValue(s, mEnd)
		if ok {
			if next := pairStartRe.FindStringIndex(s[vs:]); next != nil && next[0] == 0 && vs+next[1] > ve && isCredentialName(pairName(s[vs:vs+next[1]])) {
				ok = false
			}
		}
		if !ok {
			b.WriteString(s[pos:mEnd])
			pos = mEnd
			continue
		}
		b.WriteString(s[pos:vs])
		v := s[vs:ve]
		switch {
		case len(v) >= 2 && (v[0] == '"' || v[0] == '\'') && v[len(v)-1] == v[0]:
			b.WriteByte(v[0])
			b.WriteString(Redacted)
			b.WriteByte(v[0])
		case v[0] == '"' || v[0] == '\'':
			b.WriteByte(v[0])
			b.WriteString(Redacted)
		default:
			b.WriteString(Redacted)
		}
		pos = ve
	}
	if pos == 0 {
		return s
	}
	b.WriteString(s[pos:])
	return b.String()
}
