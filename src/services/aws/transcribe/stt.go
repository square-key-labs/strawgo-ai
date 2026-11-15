package transcribe

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming"
	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming/types"
	"github.com/aws/smithy-go/transport/http/protocol/restjson1"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/services/aws/awsauth"
	"github.com/square-key-labs/strawgo-ai/src/services/stt"
)

const (
	DefaultLanguage       = "en-US"
	DefaultBufferDuration = 2 * time.Second
	DefaultCloseTimeout   = 5 * time.Second
	maxChunkDuration      = 100 * time.Millisecond
	maxConnectFailures    = 5
	maxBackoff            = 10 * time.Second
)

type STTConfig struct {
	AWS                               awsauth.Config
	Language                          string
	SampleRate                        int
	EnablePartialResultsStabilization bool
	PartialResultsStability           string
	ShowSpeakerLabel                  bool
	LanguageOptions                   []string
	IdentifyMultipleLanguages         bool
	PreferredLanguage                 string
	VocabularyName                    string
	VocabularyFilterName              string
	VocabularyFilterMethod            string
	LanguageModelName                 string
	BufferDuration                    time.Duration
	CloseTimeout                      time.Duration
}

type STTService struct {
	*stt.Base
	client *transcribestreaming.Client
	cfg    STTConfig
	log    *logger.Logger

	ring atomic.Pointer[audioRing]

	mu           sync.Mutex
	cancel       context.CancelFunc
	streamCancel context.CancelFunc
	reconnecting bool
	done         chan struct{}
	closing      atomic.Bool
	finish       chan struct{}
	finishOnce   sync.Once
}

func NewSTTService(ctx context.Context, cfg STTConfig) (*STTService, error) {
	awsCfg, err := awsauth.Load(ctx, cfg.AWS)
	if err != nil {
		return nil, fmt.Errorf("transcribe: load aws config: %w", err)
	}
	client := transcribestreaming.NewFromConfig(awsCfg, func(o *transcribestreaming.Options) {
		if ep := cfg.AWS.BaseEndpoint(); ep != nil {
			o.BaseEndpoint = ep
		}
	})
	return NewSTTServiceWithClient(client, cfg), nil
}

func NewSTTServiceWithClient(client *transcribestreaming.Client, cfg STTConfig) *STTService {
	if cfg.Language == "" && len(cfg.LanguageOptions) == 0 {
		cfg.Language = DefaultLanguage
	}
	if cfg.BufferDuration <= 0 {
		cfg.BufferDuration = DefaultBufferDuration
	}
	if cfg.CloseTimeout <= 0 {
		cfg.CloseTimeout = DefaultCloseTimeout
	}
	s := &STTService{client: client, cfg: cfg, log: logger.WithPrefix("AWSTranscribeSTT"), finish: make(chan struct{})}
	s.Base = stt.NewStreaming("AWSTranscribeSTT", s, stt.Options{
		Provider:       "aws_transcribe",
		Settings:       frames.STTSettings{Language: cfg.Language},
		SampleRate:     cfg.SampleRate,
		BytesPerSample: 2,
		TTFSP99:        700 * time.Millisecond,
	})
	return s
}

func (s *STTService) Connect(ctx context.Context, sink stt.Sink) error {
	rate := s.SampleRate()
	if rate < 8000 || rate > 48000 {
		return fmt.Errorf("transcribe: unsupported sample rate %d (8000-48000)", rate)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done != nil {
		return nil
	}
	s.ring.Store(newAudioRing(int(s.cfg.BufferDuration.Seconds() * float64(rate) * 2)))
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	s.done = make(chan struct{})
	go s.run(runCtx, sink, rate)
	return nil
}

func (s *STTService) SendAudio(audio []byte) {
	if r := s.ring.Load(); r != nil && !s.closing.Load() {
		r.write(audio)
	}
}

func (s *STTService) Disconnect(graceful bool) {
	s.mu.Lock()
	cancel, done := s.cancel, s.done
	s.mu.Unlock()
	s.closing.Store(true)
	if done == nil {
		return
	}
	if !graceful {
		cancel()
		<-done
		return
	}
	s.finishOnce.Do(func() { close(s.finish) })
	timer := time.NewTimer(s.cfg.CloseTimeout)
	select {
	case <-done:
	case <-timer.C:
		cancel()
		<-done
	}
	timer.Stop()
	cancel()
}

func (s *STTService) ApplySettings(frames.STTSettings) error {
	s.mu.Lock()
	if s.streamCancel != nil {
		s.reconnecting = true
		s.streamCancel()
	}
	s.mu.Unlock()
	return nil
}

func (s *STTService) run(ctx context.Context, sink stt.Sink, rate int) {
	defer close(s.done)
	failures := 0
	backoff := 250 * time.Millisecond
	for ctx.Err() == nil && !s.finished() {
		sctx, cancel := context.WithCancel(ctx)
		s.mu.Lock()
		s.streamCancel = cancel
		s.reconnecting = false
		s.mu.Unlock()
		connected, err := s.session(sctx, sink, rate)
		cancel()
		s.mu.Lock()
		s.streamCancel = nil
		reconnect := s.reconnecting
		s.mu.Unlock()
		if ctx.Err() != nil || s.finished() {
			return
		}
		if reconnect {
			continue
		}
		if connected {
			failures = 0
			backoff = 250 * time.Millisecond
		}
		if err != nil {
			if !connected {
				failures++
				if failures >= maxConnectFailures {
					sink.Error(fmt.Errorf("transcribe: giving up after %d failed connects: %w", failures, err), true)
					return
				}
			}
			sink.Error(err, false)
		} else {
			s.log.Debug("stream ended, reconnecting")
		}
		timer := time.NewTimer(backoff)
		select {
		case <-ctx.Done():
		case <-s.finish:
		case <-timer.C:
		}
		timer.Stop()
		backoff = min(backoff*2, maxBackoff)
	}
}

func (s *STTService) finished() bool {
	select {
	case <-s.finish:
		return true
	default:
		return false
	}
}

func (s *STTService) input(rate int) *transcribestreaming.StartStreamTranscriptionInput {
	cfg := s.cfg
	in := &transcribestreaming.StartStreamTranscriptionInput{
		MediaEncoding:        types.MediaEncodingPcm,
		MediaSampleRateHertz: aws.Int32(int32(rate)),
		ShowSpeakerLabel:     cfg.ShowSpeakerLabel,
	}
	if len(cfg.LanguageOptions) > 0 {
		in.LanguageOptions = aws.String(strings.Join(cfg.LanguageOptions, ","))
		if cfg.IdentifyMultipleLanguages {
			in.IdentifyMultipleLanguages = true
		} else {
			in.IdentifyLanguage = true
		}
		if cfg.PreferredLanguage != "" {
			in.PreferredLanguage = types.LanguageCode(cfg.PreferredLanguage)
		}
	} else {
		lang := s.Settings().Language
		if lang == "" {
			lang = DefaultLanguage
		}
		in.LanguageCode = types.LanguageCode(lang)
	}
	if cfg.EnablePartialResultsStabilization || cfg.PartialResultsStability != "" {
		in.EnablePartialResultsStabilization = true
		if cfg.PartialResultsStability != "" {
			in.PartialResultsStability = types.PartialResultsStability(cfg.PartialResultsStability)
		}
	}
	if cfg.VocabularyName != "" {
		in.VocabularyName = aws.String(cfg.VocabularyName)
	}
	if cfg.VocabularyFilterName != "" {
		in.VocabularyFilterName = aws.String(cfg.VocabularyFilterName)
		if cfg.VocabularyFilterMethod != "" {
			in.VocabularyFilterMethod = types.VocabularyFilterMethod(cfg.VocabularyFilterMethod)
		}
	}
	if cfg.LanguageModelName != "" {
		in.LanguageModelName = aws.String(cfg.LanguageModelName)
	}
	return in
}

func (s *STTService) session(ctx context.Context, sink stt.Sink, rate int) (bool, error) {
	out, err := s.client.StartStreamTranscription(ctx, s.input(rate), func(o *transcribestreaming.Options) {
		o.Protocol = restjson1.New(nil)
	})
	if err != nil {
		if ctx.Err() != nil {
			return false, nil
		}
		return false, fmt.Errorf("transcribe: start stream: %w", err)
	}
	es := out.GetStream()
	defer es.Close()

	wctx, wcancel := context.WithCancel(ctx)
	defer wcancel()
	writerDone := make(chan error, 1)
	go func() { writerDone <- s.writeLoop(wctx, es, rate) }()

	language := s.Settings().Language
	events := es.Events()
	for {
		select {
		case <-ctx.Done():
			wcancel()
			if writerDone != nil {
				<-writerDone
			}
			return true, nil
		case err := <-writerDone:
			if err != nil {
				return true, fmt.Errorf("transcribe: send audio: %w", err)
			}
			writerDone = nil
		case ev, ok := <-events:
			if !ok {
				wcancel()
				if writerDone != nil {
					<-writerDone
				}
				if err := es.Err(); err != nil && ctx.Err() == nil && !s.finished() {
					return true, fmt.Errorf("transcribe: %w", err)
				}
				return true, nil
			}
			if te, isTranscript := ev.(*types.TranscriptResultStreamMemberTranscriptEvent); isTranscript {
				emitResults(sink, te.Value.Transcript, language)
			}
		}
	}
}

func emitResults(sink stt.Sink, t *types.Transcript, language string) {
	if t == nil {
		return
	}
	for _, r := range t.Results {
		if len(r.Alternatives) == 0 {
			continue
		}
		text := strings.TrimSpace(aws.ToString(r.Alternatives[0].Transcript))
		if text == "" {
			continue
		}
		lang := string(r.LanguageCode)
		if lang == "" {
			lang = language
		}
		sink.Transcript(stt.Transcript{Text: text, Final: !r.IsPartial, Language: lang})
	}
}

func (s *STTService) writeLoop(ctx context.Context, es *transcribestreaming.StartStreamTranscriptionEventStream, rate int) error {
	ring := s.ring.Load()
	buf := make([]byte, int(maxChunkDuration.Seconds()*float64(rate))*2)
	event := &types.AudioStreamMemberAudioEvent{}
	send := func(chunk []byte) error {
		event.Value.AudioChunk = chunk
		return es.Send(ctx, event)
	}
	finish := s.finish
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ring.notify:
		case <-finish:
			for {
				n := ring.read(buf)
				if n == 0 {
					break
				}
				if err := send(buf[:n]); err != nil {
					return ignoreCanceled(ctx, err)
				}
			}
			if err := send([]byte{}); err != nil {
				return ignoreCanceled(ctx, err)
			}
			return ignoreCanceled(ctx, es.Writer.Close())
		}
		if d := ring.takeDropped(); d > 0 {
			s.log.Warn("audio queue full, dropped %d bytes", d)
		}
		for {
			n := ring.read(buf)
			if n == 0 {
				break
			}
			if err := send(buf[:n]); err != nil {
				return ignoreCanceled(ctx, err)
			}
		}
	}
}

func ignoreCanceled(ctx context.Context, err error) error {
	if err == nil || ctx.Err() != nil || errors.Is(err, context.Canceled) {
		return nil
	}
	return err
}
