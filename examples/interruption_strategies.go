//go:build ignore

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/turns"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_start"
	"github.com/square-key-labs/strawgo-ai/src/turns/user_stop"
)

// InterruptionCheckProcessor demonstrates how to use interruption strategies
type InterruptionCheckProcessor struct {
	*processors.BaseProcessor
	botSpeaking    bool
	userTurnActive bool
}

func NewInterruptionCheckProcessor() *InterruptionCheckProcessor {
	p := &InterruptionCheckProcessor{
		botSpeaking: false,
	}
	p.BaseProcessor = processors.NewBaseProcessor("InterruptionChecker", p)
	return p
}

func (p *InterruptionCheckProcessor) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle StartFrame to configure interruptions
	if startFrame, ok := frame.(*frames.StartFrame); ok {
		p.HandleStartFrame(startFrame)
		strategies := p.TurnStrategies()
		totalStrategies := len(strategies.StartStrategies) + len(strategies.StopStrategies) + len(strategies.MuteStrategies)
		log.Printf("[%s] Configured with %d turn strategies", p.Name(), totalStrategies)
		return p.PushFrame(frame, direction)
	}

	// Handle InterruptionFrame
	if _, ok := frame.(*frames.InterruptionFrame); ok {
		log.Printf("[%s] ⚠️  Interruption received! Bot was speaking: %v", p.Name(), p.botSpeaking)
		p.HandleInterruptionFrame()
		p.botSpeaking = false
		p.userTurnActive = false
		return p.PushFrame(frame, direction)
	}

	// Handle TTS frames to track bot speaking state
	if _, ok := frame.(*frames.TTSStartedFrame); ok {
		p.botSpeaking = true
		log.Printf("[%s] Bot started speaking", p.Name())
		return p.PushFrame(frame, direction)
	}

	if _, ok := frame.(*frames.TTSStoppedFrame); ok {
		p.botSpeaking = false
		log.Printf("[%s] Bot stopped speaking", p.Name())
		return p.PushFrame(frame, direction)
	}

	strategies := p.TurnStrategies()
	if !p.userTurnActive {
		for _, strategy := range strategies.StartStrategies {
			if !strategy.ShouldStart(frame) {
				continue
			}

			p.userTurnActive = true
			if p.botSpeaking && p.InterruptionsAllowed() && strategy.EnableInterruptions() {
				log.Printf("[%s] 🔴 Turn start triggered interruption", p.Name())
				if err := p.PushInterruptionTaskFrame(); err != nil {
					log.Printf("[%s] Error pushing interruption task frame: %v", p.Name(), err)
				}
			}

			for _, startStrategy := range strategies.StartStrategies {
				startStrategy.Reset()
			}
			break
		}
	}

	if p.userTurnActive {
		for _, strategy := range strategies.StopStrategies {
			if !strategy.ShouldStop(frame) {
				continue
			}

			p.userTurnActive = false
			for _, stopStrategy := range strategies.StopStrategies {
				stopStrategy.Reset()
			}
			break
		}
	}

	// Pass through all other frames
	return p.PushFrame(frame, direction)
}

func main() {
	fmt.Println("=================================================")
	fmt.Println("StrawGo - Interruption Strategies Example")
	fmt.Println("Demonstrating allow_interruptions and interruption_strategy")
	fmt.Println("=================================================\n")

	// Create processors
	interruptionChecker := NewInterruptionCheckProcessor()
	printer := processors.NewTextPrinterProcessor()

	// Build pipeline
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		interruptionChecker,
		printer,
	})

	// Configure interruption strategies
	turnStrategies := turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewMinWordsUserTurnStartStrategy(3, true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(500*time.Millisecond, true),
		},
	}

	config := &pipeline.PipelineTaskConfig{
		AllowInterruptions: true,
		TurnStrategies:     turnStrategies,
	}

	// Create task with config
	task := pipeline.NewPipelineTaskWithConfig(pipe, config)

	// Set up event handlers
	task.OnStarted(func() {
		fmt.Println("\n✅ Pipeline started!\n")
	})

	task.OnFinished(func() {
		fmt.Println("\n✅ Pipeline finished!\n")
	})

	task.OnError(func(err error) {
		fmt.Printf("\n❌ Error: %v\n", err)
	})

	// Simulate conversation flow
	go func() {
		time.Sleep(100 * time.Millisecond)

		// Bot starts speaking
		log.Println("[Main] Bot starts speaking...")
		task.QueueFrame(frames.NewTTSStartedFrame())

		time.Sleep(100 * time.Millisecond)

		// User interrupts with 1 word (should not interrupt - need 3 words)
		log.Println("\n[Main] User says: 'Hey' (1 word)")
		task.QueueFrame(frames.NewUserStartedSpeakingFrame())
		task.QueueFrame(frames.NewTextFrame("Hey"))
		task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

		time.Sleep(200 * time.Millisecond)

		// User interrupts with 4 words (should interrupt!)
		log.Println("\n[Main] User says: 'Wait hold on please' (4 words)")
		task.QueueFrame(frames.NewUserStartedSpeakingFrame())
		task.QueueFrame(frames.NewTextFrame("Wait hold on please"))
		task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

		time.Sleep(200 * time.Millisecond)

		// Bot stops speaking (after interruption)
		log.Println("\n[Main] Bot stops speaking")
		task.QueueFrame(frames.NewTTSStoppedFrame())

		time.Sleep(200 * time.Millisecond)

		// User speaks again (bot not speaking, no interruption needed)
		log.Println("\n[Main] User says: 'Hello there friend' (3 words, but bot not speaking)")
		task.QueueFrame(frames.NewUserStartedSpeakingFrame())
		task.QueueFrame(frames.NewTextFrame("Hello there friend"))
		task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

		time.Sleep(200 * time.Millisecond)

		// End pipeline
		log.Println("\n[Main] Sending EndFrame")
		task.QueueFrame(frames.NewEndFrame())
	}()

	// Run the pipeline
	ctx := context.Background()
	if err := task.Run(ctx); err != nil {
		log.Fatalf("Pipeline error: %v", err)
	}

	fmt.Println("=================================================")
	fmt.Println("Example execution completed!")
	fmt.Println("=================================================")
}
