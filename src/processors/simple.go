package processors

import (
	"context"
	"fmt"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

type TextGeneratorProcessor struct {
	*BaseProcessor
	messages []string
	started  bool
	log      *logger.Logger
}

func NewTextGeneratorProcessor(messages []string) *TextGeneratorProcessor {
	tg := &TextGeneratorProcessor{
		messages: messages,
		log:      logger.WithPrefix("TextGenerator"),
	}
	tg.BaseProcessor = NewBaseProcessor("TextGenerator", tg)
	return tg
}

func (p *TextGeneratorProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if _, ok := frame.(*frames.StartFrame); ok {
		if !p.started {
			p.started = true
			go p.generateText(ctx)
		}
		return p.PushFrame(frame, direction)
	}

	return p.PushFrame(frame, direction)
}

func (p *TextGeneratorProcessor) generateText(ctx context.Context) {
	time.Sleep(100 * time.Millisecond)

	for _, msg := range p.messages {
		select {
		case <-ctx.Done():
			return
		default:
			textFrame := frames.NewTextFrame(msg)
			p.log.Debug("Generated: %s", msg)
			if err := p.PushFrame(textFrame, frames.Downstream); err != nil {
				p.log.Error("Error pushing frame: %v", err)
				return
			}
			time.Sleep(200 * time.Millisecond)
		}
	}
}

type TextPrinterProcessor struct {
	*BaseProcessor
}

func NewTextPrinterProcessor() *TextPrinterProcessor {
	tp := &TextPrinterProcessor{}
	tp.BaseProcessor = NewBaseProcessor("TextPrinter", tp)
	return tp
}

func (p *TextPrinterProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		fmt.Printf("\n📝 [OUTPUT] %s\n", textFrame.Text)
	}

	return p.PushFrame(frame, direction)
}

type PassthroughProcessor struct {
	*BaseProcessor
	logFrames bool
	log       *logger.Logger
}

func NewPassthroughProcessor(name string, logFrames bool) *PassthroughProcessor {
	pp := &PassthroughProcessor{
		logFrames: logFrames,
		log:       logger.WithPrefix(name),
	}
	pp.BaseProcessor = NewBaseProcessor(name, pp)
	return pp
}

func (p *PassthroughProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if p.logFrames {
		p.log.Debug("%s frame %s", direction, frame.Name())
	}
	return p.PushFrame(frame, direction)
}

type TextTransformProcessor struct {
	*BaseProcessor
	transform func(string) string
	log       *logger.Logger
}

func NewTextTransformProcessor(name string, transform func(string) string) *TextTransformProcessor {
	tp := &TextTransformProcessor{
		transform: transform,
		log:       logger.WithPrefix(name),
	}
	tp.BaseProcessor = NewBaseProcessor(name, tp)
	return tp
}

type TextCarrier interface {
	GetText() string
	SetText(text string)
}

func (p *TextTransformProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if p.transform == nil {
		return p.PushFrame(frame, direction)
	}
	if carrier, ok := frame.(TextCarrier); ok {
		carrier.SetText(p.transform(carrier.GetText()))
	} else if text := textField(frame); text != nil {
		*text = p.transform(*text)
	}
	return p.PushFrame(frame, direction)
}

func textField(frame frames.Frame) *string {
	switch f := frame.(type) {
	case *frames.TextFrame:
		return &f.Text
	case *frames.LLMTextFrame:
		return &f.Text
	case *frames.TranscriptionFrame:
		return &f.Text
	case *frames.AggregatedTextFrame:
		return &f.Text
	case *frames.TTSTextFrame:
		return &f.Text
	case *frames.VisionTextFrame:
		return &f.Text
	case *frames.InterimTranscriptionFrame:
		return &f.Text
	case *frames.EagerTranscriptionFrame:
		return &f.Text
	case *frames.TranslationFrame:
		return &f.Text
	case *frames.InputTextRawFrame:
		return &f.Text
	}
	return nil
}
