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
	fmt.Println("StrawGo - A Go-based Pipecat Replacement")
	fmt.Println("Example: Text Flow Pipeline")
	fmt.Println("=================================================\n")

	// Create processors
	messages := []string{
		"Hello, World!",
		"This is StrawGo",
		"A real-time conversational AI pipeline framework",
		"Built in Go",
		"Inspired by Pipecat",
	}

	generator := processors.NewTextGeneratorProcessor(messages)
	uppercase := processors.NewTextTransformProcessor("Uppercase", strings.ToUpper)
	printer := processors.NewTextPrinterProcessor()

	// Build pipeline
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		generator,
		uppercase,
		printer,
	})

	// Create task
	task := pipeline.NewPipelineTask(pipe)

	// Set up event handlers
	task.OnStarted(func() {
		fmt.Println("\n✅ Pipeline started successfully!\n")
	})

	task.OnFinished(func() {
		fmt.Println("\n✅ Pipeline finished successfully!\n")
	})

	task.OnError(func(err error) {
		fmt.Printf("\n❌ Pipeline error: %v\n", err)
	})

	// Queue an end frame after a delay
	go func() {
		time.Sleep(2 * time.Second)
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
	fmt.Println("Pipeline execution completed!")
	fmt.Println("=================================================")
}
