package processors

import (
	"context"
	"fmt"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/logger"
)

// TextGeneratorProcessor generates text frames periodically
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
	// When we receive StartFrame, begin generating text
	if _, ok := frame.(*frames.StartFrame); ok {
		if !p.started {
			p.started = true
			go p.generateText(ctx)
		}
		return p.PushFrame(frame, direction)
	}

	// Pass all other frames through
	return p.PushFrame(frame, direction)
}

func (p *TextGeneratorProcessor) generateText(ctx context.Context) {
	// Wait a moment for pipeline to be fully ready
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

// TextPrinterProcessor prints received text frames
type TextPrinterProcessor struct {
	*BaseProcessor
}

func NewTextPrinterProcessor() *TextPrinterProcessor {
	tp := &TextPrinterProcessor{}
	tp.BaseProcessor = NewBaseProcessor("TextPrinter", tp)
	return tp
}

func (p *TextPrinterProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Print text frames
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		fmt.Printf("\n📝 [OUTPUT] %s\n", textFrame.Text)
	}

	// Pass all frames through
	return p.PushFrame(frame, direction)
}

// PassthroughProcessor simply passes frames through unchanged
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

// TextTransformProcessor transforms text frames
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

func (p *TextTransformProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Transform text frames
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		transformed := p.transform(textFrame.Text)
		newFrame := frames.NewTextFrame(transformed)
		p.log.Debug("Transformed: '%s' -> '%s'", textFrame.Text, transformed)
		return p.PushFrame(newFrame, direction)
	}

	// Pass all other frames through
	return p.PushFrame(frame, direction)
}
