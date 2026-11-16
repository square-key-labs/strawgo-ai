package voicemail

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/audio/tone"
	"github.com/square-key-labs/strawgo-ai/src/llm"
)

type Verdict int

const (
	Undecided Verdict = iota
	Conversation
	Voicemail
)

func (v Verdict) String() string {
	switch v {
	case Conversation:
		return "conversation"
	case Voicemail:
		return "voicemail"
	}
	return "undecided"
}

type Classifier interface {
	RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error)
}

const DefaultPrompt = `You are a voicemail detection classifier for an outbound phone call. A bot placed the call, and you receive the transcription of what was heard after the call connected. Decide whether a person answered or the call went to voicemail.

CONVERSATION: a person answered. A greeting such as "hello?", "hi", "yeah?" or "John speaking"; a question to the caller such as "who is this?" or "can I help you?"; spontaneous speech that expects a reply.

VOICEMAIL: an automated greeting or carrier message. "You've reached", "leave a message", "I'm not available right now", "call me back", "mailbox is full", "not in service", "all circuits are busy", "our office is currently closed".

Respond with exactly one word: CONVERSATION or VOICEMAIL.`

const (
	DefaultVoicemailResponseDelay = 2 * time.Second
	DefaultDecisionTimeout        = time.Second
	DefaultClassifyTimeout        = 10 * time.Second
	DefaultMaxBufferedFrames      = 4096
	DefaultBeepTimeout            = 4 * time.Second
	StrongBeepPurity              = 0.85
	StrongBeepDuration            = 400 * time.Millisecond
	maxTranscriptLen              = 8192
)

var ErrInvalidPrompt = errors.New("voicemail classifier prompt must instruct the model to answer CONVERSATION or VOICEMAIL")

func ValidatePrompt(prompt string) error {
	upper := strings.ToUpper(prompt)
	if strings.TrimSpace(prompt) == "" || !strings.Contains(upper, "CONVERSATION") || !strings.Contains(upper, "VOICEMAIL") {
		return ErrInvalidPrompt
	}
	return nil
}

func ParseVerdict(reply string) Verdict {
	upper := strings.ToUpper(reply)
	iv := strings.Index(upper, "VOICEMAIL")
	ic := strings.Index(upper, "CONVERSATION")
	switch {
	case iv < 0 && ic < 0:
		return Undecided
	case ic < 0 || (iv >= 0 && iv < ic):
		return Voicemail
	default:
		return Conversation
	}
}

type Handler func(d *Detector)

type ResultHandler func(d *Detector, r Result)

type Option func(*config)

type config struct {
	prompt          string
	settings        llm.Settings
	responseDelay   time.Duration
	decisionTimeout time.Duration
	classifyTimeout time.Duration
	maxBuffered     int
	onConversation  Handler
	onVoicemail     Handler
	onResult        ResultHandler
	detailed        bool
	timing          *TimingThresholds
	beep            bool
	beepTimeout     time.Duration
	beepTone        tone.Config
}

func WithPrompt(prompt string) Option {
	return func(c *config) { c.prompt = prompt }
}

func WithClassifierSettings(settings llm.Settings) Option {
	return func(c *config) { c.settings = settings }
}

func WithVoicemailResponseDelay(d time.Duration) Option {
	return func(c *config) { c.responseDelay = d }
}

func WithDecisionTimeout(d time.Duration) Option {
	return func(c *config) { c.decisionTimeout = d }
}

func WithClassifyTimeout(d time.Duration) Option {
	return func(c *config) { c.classifyTimeout = d }
}

func WithMaxBufferedFrames(n int) Option {
	return func(c *config) { c.maxBuffered = n }
}

func OnConversationDetected(h Handler) Option {
	return func(c *config) { c.onConversation = h }
}

func OnVoicemailDetected(h Handler) Option {
	return func(c *config) { c.onVoicemail = h }
}

func OnResult(h ResultHandler) Option {
	return func(c *config) { c.onResult = h }
}

func WithDetailedClasses() Option {
	return func(c *config) { c.detailed = true }
}

func WithTimingHeuristic(t TimingThresholds) Option {
	return func(c *config) {
		t = t.withDefaults()
		c.timing = &t
	}
}

func WithBeepDetection(fallback time.Duration) Option {
	return func(c *config) {
		c.beep = true
		c.beepTimeout = fallback
	}
}

func WithBeepTone(cfg tone.Config) Option {
	return func(c *config) { c.beepTone = cfg }
}

type VoicemailDetector struct {
	detector *Detector
	gate     *TTSGate
}

func New(classifier Classifier, opts ...Option) (*VoicemailDetector, error) {
	if classifier == nil {
		return nil, errors.New("voicemail detector needs a classifier")
	}
	cfg := config{
		prompt:          DefaultPrompt,
		responseDelay:   DefaultVoicemailResponseDelay,
		decisionTimeout: DefaultDecisionTimeout,
		classifyTimeout: DefaultClassifyTimeout,
		maxBuffered:     DefaultMaxBufferedFrames,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.detailed && cfg.prompt == DefaultPrompt {
		cfg.prompt = DetailedPrompt
	}
	if cfg.beep && cfg.beepTimeout <= 0 {
		cfg.beepTimeout = DefaultBeepTimeout
	}
	if err := ValidatePrompt(cfg.prompt); err != nil {
		return nil, err
	}
	gate := newTTSGate(cfg.maxBuffered)
	return &VoicemailDetector{
		detector: newDetector(classifier, cfg, gate),
		gate:     gate,
	}, nil
}

func (v *VoicemailDetector) Detector() *Detector {
	return v.detector
}

func (v *VoicemailDetector) Gate() *TTSGate {
	return v.gate
}

func (v *VoicemailDetector) Decision() Verdict {
	return v.detector.Decision()
}

func (v *VoicemailDetector) Result() Result {
	return v.detector.Result()
}

func (v *VoicemailDetector) ReportCarrierResult(c Class) bool {
	return v.detector.ReportCarrierResult(c)
}

func (v *VoicemailDetector) ReportAnsweredBy(answeredBy string) bool {
	return v.detector.ReportAnsweredBy(answeredBy)
}
