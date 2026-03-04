package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
)

func main() {
	fmt.Println("=================================================")
	fmt.Println("StrawGo - Advanced Example")
	fmt.Println("Demonstrating Interruptions and Bidirectional Flow")
	fmt.Println("=================================================\n")

	// Create processors
	messages := []string{
		"Processing message one",
		"Processing message two",
		"Processing message three",
		"This message will be interrupted",
		"This won't be shown",
	}

	generator := processors.NewTextGeneratorProcessor(messages)
	passthrough := processors.NewPassthroughProcessor("Monitor", true)
	lowercase := processors.NewTextTransformProcessor("Lowercase", strings.ToLower)
	printer := processors.NewTextPrinterProcessor()

	// Build pipeline
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		generator,
		passthrough,
		lowercase,
		printer,
	})

	// Create task
	task := pipeline.NewPipelineTask(pipe)

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

	// Simulate an interruption after a delay
	go func() {
		time.Sleep(800 * time.Millisecond)
		log.Println("\n[Main] ⚠️  Simulating user interruption!")
		if err := task.QueueFrame(frames.NewInterruptionFrame()); err != nil {
			log.Printf("[Main] Error queuing interruption frame: %v", err)
		}

		// Queue a new message after interruption
		time.Sleep(200 * time.Millisecond)
		log.Println("[Main] Queuing post-interruption message")
		if err := task.QueueFrame(frames.NewTextFrame("User interrupted and said something new")); err != nil {
			log.Printf("[Main] Error queuing text frame: %v", err)
		}

		// End after another delay
		time.Sleep(500 * time.Millisecond)
		log.Println("[Main] Sending EndFrame")
		if err := task.QueueFrame(frames.NewEndFrame()); err != nil {
			log.Printf("[Main] Error queuing end frame: %v", err)
		}
	}()

	// Run the pipeline
	ctx := context.Background()
	if err := task.Run(ctx); err != nil {
		log.Fatalf("Pipeline error: %v", err)
	}

	fmt.Println("=================================================")
	fmt.Println("Advanced pipeline execution completed!")
	fmt.Println("=================================================")
}
