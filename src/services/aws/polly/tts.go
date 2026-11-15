package polly

import (
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/polly"
	"github.com/aws/aws-sdk-go-v2/service/polly/types"
	"github.com/square-key-labs/strawgo-ai/src/audio/resample"
	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/services/aws/awsauth"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
)

const (
	DefaultVoice    = "Joanna"
	DefaultLanguage = "en-US"

	ExtraEngine       = "engine"
	ExtraPitch        = "pitch"
	ExtraRate         = "rate"
	ExtraVolume       = "volume"
	ExtraLexiconNames = "lexicon_names"
)

type TTSConfig struct {
	AWS          awsauth.Config
	Voice        string
	Language     string
	Engine       string
	Pitch        string
	Rate         string
	Volume       string
	LexiconNames []string
	SampleRate   int
}

type TTSService struct {
	*tts.Base
	client *polly.Client
}

func NewTTSService(ctx context.Context, cfg TTSConfig) (*TTSService, error) {
	awsCfg, err := awsauth.Load(ctx, cfg.AWS)
	if err != nil {
		return nil, fmt.Errorf("polly: load aws config: %w", err)
	}
	client := polly.NewFromConfig(awsCfg, func(o *polly.Options) {
		if ep := cfg.AWS.BaseEndpoint(); ep != nil {
			o.BaseEndpoint = ep
		}
	})
	return NewTTSServiceWithClient(client, cfg), nil
}

func NewTTSServiceWithClient(client *polly.Client, cfg TTSConfig) *TTSService {
	voice := cfg.Voice
	if voice == "" {
		voice = DefaultVoice
	}
	language := cfg.Language
	if language == "" {
		language = DefaultLanguage
	}
	extra := map[string]any{}
	setIf := func(k, v string) {
		if v != "" {
			extra[k] = v
		}
	}
	setIf(ExtraPitch, cfg.Pitch)
	setIf(ExtraRate, cfg.Rate)
	setIf(ExtraVolume, cfg.Volume)
	if len(cfg.LexiconNames) > 0 {
		extra[ExtraLexiconNames] = cfg.LexiconNames
	}
	s := &TTSService{client: client}
	s.Base = tts.NewHTTP("AWSPollyTTS", s, tts.Options{
		Settings:       frames.TTSSettings{Model: cfg.Engine, Voice: voice, Language: language, Extra: extra},
		SampleRate:     cfg.SampleRate,
		Codec:          "linear16",
		BytesPerSample: 2,
	})
	return s
}

func pollyRate(rate int) int {
	if rate == 8000 {
		return 8000
	}
	return 16000
}

func (s *TTSService) Synthesize(ctx context.Context, req tts.Request) (io.ReadCloser, error) {
	settings := req.Settings
	engine := settings.Model
	if v, ok := settings.Extra[ExtraEngine].(string); ok && v != "" {
		engine = v
	}
	srcRate := pollyRate(req.SampleRate)
	input := &polly.SynthesizeSpeechInput{
		Text:         aws.String(buildSSML(req.Text, settings, engine)),
		TextType:     types.TextTypeSsml,
		OutputFormat: types.OutputFormatPcm,
		VoiceId:      types.VoiceId(settings.Voice),
		SampleRate:   aws.String(strconv.Itoa(srcRate)),
		LexiconNames: lexicons(settings.Extra[ExtraLexiconNames]),
	}
	if engine != "" {
		input.Engine = types.Engine(engine)
	}
	out, err := s.client.SynthesizeSpeech(ctx, input)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("polly: %w", err)
	}
	if out.AudioStream == nil {
		return nil, errors.New("polly: response has no audio stream")
	}
	if req.SampleRate <= 0 || req.SampleRate == srcRate {
		return out.AudioStream, nil
	}
	rs, err := resample.New(srcRate, req.SampleRate)
	if err != nil {
		out.AudioStream.Close()
		return nil, err
	}
	return &resamplingReader{src: out.AudioStream, rs: rs, in: make([]byte, 4096)}, nil
}

func lexicons(v any) []string {
	switch l := v.(type) {
	case []string:
		return l
	case []any:
		out := make([]string, 0, len(l))
		for _, x := range l {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
		return out
	case string:
		if l != "" {
			return []string{l}
		}
	}
	return nil
}

func buildSSML(text string, settings frames.TTSSettings, engine string) string {
	str := func(k string) string {
		v, _ := settings.Extra[k].(string)
		return v
	}
	var attrs [][2]string
	if engine == string(types.EngineStandard) {
		if v := str(ExtraPitch); v != "" {
			attrs = append(attrs, [2]string{"pitch", v})
		}
	}
	rate := str(ExtraRate)
	if rate == "" && settings.Speed != nil && *settings.Speed > 0 && *settings.Speed != 1 {
		rate = strconv.Itoa(int(*settings.Speed*100+0.5)) + "%"
	}
	if rate != "" {
		attrs = append(attrs, [2]string{"rate", rate})
	}
	if v := str(ExtraVolume); v != "" {
		attrs = append(attrs, [2]string{"volume", v})
	}
	var b strings.Builder
	b.Grow(len(text) + 96)
	b.WriteString("<speak><lang xml:lang='")
	escape(&b, settings.Language)
	b.WriteString("'>")
	if len(attrs) > 0 {
		b.WriteString("<prosody")
		for _, a := range attrs {
			b.WriteByte(' ')
			b.WriteString(a[0])
			b.WriteString("='")
			escape(&b, a[1])
			b.WriteByte('\'')
		}
		b.WriteByte('>')
	}
	escape(&b, text)
	if len(attrs) > 0 {
		b.WriteString("</prosody>")
	}
	b.WriteString("</lang></speak>")
	return b.String()
}

func escape(b *strings.Builder, s string) {
	_ = xml.EscapeText(b, []byte(s))
}

type resamplingReader struct {
	src     io.ReadCloser
	rs      *resample.Resampler
	in      []byte
	carry   int
	out     []byte
	off     int
	eof     bool
	flushed bool
}

func (r *resamplingReader) Read(p []byte) (int, error) {
	for r.off >= len(r.out) {
		if r.flushed {
			return 0, io.EOF
		}
		r.out, r.off = r.out[:0], 0
		if r.eof {
			r.flushed = true
			tail := r.rs.Flush(nil)
			for _, v := range tail {
				r.out = append(r.out, byte(v), byte(uint16(v)>>8))
			}
			continue
		}
		n, err := r.src.Read(r.in[r.carry:])
		n += r.carry
		even := n &^ 1
		if even > 0 {
			r.out = r.rs.ProcessBytes(r.out, r.in[:even])
		}
		r.carry = n - even
		if r.carry > 0 {
			r.in[0] = r.in[even]
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				return 0, err
			}
			r.eof = true
		}
	}
	n := copy(p, r.out[r.off:])
	r.off += n
	return n, nil
}

func (r *resamplingReader) Close() error {
	return r.src.Close()
}
