package processors

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
)

// TextGeneratorProcessor generates text frames periodically
type TextGeneratorProcessor struct {
	*BaseProcessor
	messages []string
	started  bool
}

func NewTextGeneratorProcessor(messages []string) *TextGeneratorProcessor {
	tg := &TextGeneratorProcessor{
		messages: messages,
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
			log.Printf("[%s] Generated: %s", p.name, msg)
			if err := p.PushFrame(textFrame, frames.Downstream); err != nil {
				log.Printf("[%s] Error pushing frame: %v", p.name, err)
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
}

func NewPassthroughProcessor(name string, logFrames bool) *PassthroughProcessor {
	pp := &PassthroughProcessor{
		logFrames: logFrames,
	}
	pp.BaseProcessor = NewBaseProcessor(name, pp)
	return pp
}

func (p *PassthroughProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	if p.logFrames {
		log.Printf("[%s] %s frame %s", p.name, direction, frame.Name())
	}
	return p.PushFrame(frame, direction)
}

// TextTransformProcessor transforms text frames
type TextTransformProcessor struct {
	*BaseProcessor
	transform func(string) string
}

func NewTextTransformProcessor(name string, transform func(string) string) *TextTransformProcessor {
	tp := &TextTransformProcessor{
		transform: transform,
	}
	tp.BaseProcessor = NewBaseProcessor(name, tp)
	return tp
}

func (p *TextTransformProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Transform text frames
	if textFrame, ok := frame.(*frames.TextFrame); ok {
		transformed := p.transform(textFrame.Text)
		newFrame := frames.NewTextFrame(transformed)
		log.Printf("[%s] Transformed: '%s' -> '%s'", p.name, textFrame.Text, transformed)
		return p.PushFrame(newFrame, direction)
	}

	// Pass all other frames through
	return p.PushFrame(frame, direction)
}
