package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/interruptions"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services"
)

// This example demonstrates:
// 1. LLMUserAggregator accumulating user transcriptions
// 2. LLMAssistantAggregator accumulating bot responses
// 3. Intelligent interruption handling based on MinWordsStrategy
// 4. Context persistence across multiple conversation turns

func main() {
	fmt.Println("=================================================================")
	fmt.Println("   StrawGo - LLM Aggregators with Intelligent Interruptions")
	fmt.Println("=================================================================\n")

	// Create shared LLM context
	llmContext := services.NewLLMContext(`You are a helpful voice assistant.
Keep your responses brief and conversational.`)

	fmt.Println("✓ Created LLM context with system prompt")

	// Create aggregators
	userAgg := aggregators.NewLLMUserAggregator(llmContext, nil)
	assistantAgg := aggregators.NewLLMAssistantAggregator(llmContext, nil)

	fmt.Println("✓ Created UserAggregator and AssistantAggregator")

	// Create a simple LLM simulator
	llmSimulator := NewSimpleLLMSimulator()

	// Build pipeline with aggregators
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		userAgg,
		llmSimulator,
		assistantAgg,
	})

	fmt.Println("✓ Built pipeline: UserAgg → LLM → AssistantAgg")

	// Configure interruption strategies
	config := &pipeline.PipelineTaskConfig{
		AllowInterruptions: true,
		InterruptionStrategies: []interruptions.InterruptionStrategy{
			interruptions.NewMinWordsInterruptionStrategy(3), // Interrupt after 3 words
		},
	}

	fmt.Println("✓ Configured interruptions (MinWords strategy: 3 words)")

	// Create pipeline task
	task := pipeline.NewPipelineTaskWithConfig(pipe, config)

	fmt.Println("✓ Created pipeline task\n")

	// Set up event handlers
	task.OnStarted(func() {
		fmt.Println("=================================================================")
		fmt.Println("                    PIPELINE STARTED")
		fmt.Println("=================================================================\n")
	})

	task.OnFinished(func() {
		fmt.Println("\n=================================================================")
		fmt.Println("                   PIPELINE FINISHED")
		fmt.Println("=================================================================")
	})

	task.OnError(func(err error) {
		fmt.Printf("\n❌ Pipeline error: %v\n", err)
	})

	// Simulate conversation flow
	go func() {
		time.Sleep(200 * time.Millisecond)

		fmt.Println("📞 Starting conversation simulation...\n")
		fmt.Println("-----------------------------------------------------------------")

		// SCENARIO 1: Normal conversation (no interruption)
		fmt.Println("\n🎤 SCENARIO 1: Normal Conversation")
		fmt.Println("User speaks while bot is silent")
		fmt.Println("-----------------------------------------------------------------")

		task.QueueFrame(frames.NewTranscriptionFrame("Hello", false))
		time.Sleep(50 * time.Millisecond)
		task.QueueFrame(frames.NewTranscriptionFrame("Hello there", true))

		time.Sleep(500 * time.Millisecond)

		// Bot starts speaking (LLM will simulate this)
		fmt.Println("\n🤖 Bot starts speaking...")
		task.QueueFrame(frames.NewTTSStartedFrame())

		time.Sleep(300 * time.Millisecond)

		// SCENARIO 2: User tries to interrupt with 1 word (fails)
		fmt.Println("\n🎤 SCENARIO 2: Weak Interruption Attempt")
		fmt.Println("User says 'Hey' (1 word) - NOT ENOUGH")
		fmt.Println("-----------------------------------------------------------------")

		task.QueueFrame(frames.NewTranscriptionFrame("Hey", true))

		time.Sleep(300 * time.Millisecond)

		// SCENARIO 3: User interrupts with 3+ words (succeeds)
		fmt.Println("\n🎤 SCENARIO 3: Successful Interruption")
		fmt.Println("User says 'Wait hold on' (3 words) - INTERRUPTION!")
		fmt.Println("-----------------------------------------------------------------")

		task.QueueFrame(frames.NewTranscriptionFrame("Wait hold on", true))

		time.Sleep(500 * time.Millisecond)

		// Bot should have stopped
		task.QueueFrame(frames.NewTTSStoppedFrame())
		fmt.Println("\n🤖 Bot stopped speaking (interrupted)")

		time.Sleep(300 * time.Millisecond)

		// SCENARIO 4: Bot starts speaking again
		fmt.Println("\n🤖 Bot starts speaking again...")
		task.QueueFrame(frames.NewTTSStartedFrame())

		time.Sleep(300 * time.Millisecond)

		// SCENARIO 5: User interrupts with 5 words
		fmt.Println("\n🎤 SCENARIO 4: Another Successful Interruption")
		fmt.Println("User says 'Actually I have a question' (5 words)")
		fmt.Println("-----------------------------------------------------------------")

		task.QueueFrame(frames.NewTranscriptionFrame("Actually I have a question", true))

		time.Sleep(500 * time.Millisecond)

		task.QueueFrame(frames.NewTTSStoppedFrame())
		fmt.Println("\n🤖 Bot stopped speaking (interrupted)")

		time.Sleep(300 * time.Millisecond)

		// SCENARIO 6: Final user message (no bot speaking)
		fmt.Println("\n🎤 SCENARIO 5: Normal Turn")
		fmt.Println("User speaks while bot is silent")
		fmt.Println("-----------------------------------------------------------------")

		task.QueueFrame(frames.NewTranscriptionFrame("Thank you", true))

		time.Sleep(500 * time.Millisecond)

		// Show final context state
		fmt.Println("\n=================================================================")
		fmt.Println("                  FINAL CONTEXT STATE")
		fmt.Println("=================================================================")
		fmt.Printf("Total messages in context: %d\n\n", len(llmContext.Messages))

		for i, msg := range llmContext.Messages {
			fmt.Printf("%d. [%s] %s\n", i+1, msg.Role, msg.Content)
		}

		fmt.Println("\n=================================================================")

		// End pipeline
		time.Sleep(300 * time.Millisecond)
		task.QueueFrame(frames.NewEndFrame())
	}()

	// Run the pipeline
	ctx := context.Background()
	if err := task.Run(ctx); err != nil {
		log.Fatalf("Pipeline error: %v", err)
	}

	fmt.Println("\n✅ Example completed successfully!")
	fmt.Println("\nKey Takeaways:")
	fmt.Println("1. UserAggregator accumulated transcriptions")
	fmt.Println("2. Interruptions triggered only when 3+ words spoken")
	fmt.Println("3. Input discarded when interruption conditions not met")
	fmt.Println("4. Context persisted across all conversation turns")
	fmt.Println("5. AssistantAggregator updated context after each response")
}

// SimpleLLMSimulator simulates an LLM for demonstration purposes
type SimpleLLMSimulator struct {
	*processors.BaseProcessor
	responseCounter int
}

func NewSimpleLLMSimulator() *SimpleLLMSimulator {
	s := &SimpleLLMSimulator{
		responseCounter: 0,
	}
	s.BaseProcessor = processors.NewBaseProcessor("SimpleLLMSimulator", s)
	return s
}

func (s *SimpleLLMSimulator) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	// Handle LLMContextFrame (from UserAggregator)
	if contextFrame, ok := frame.(*frames.LLMContextFrame); ok {
		if llmContext, ok := contextFrame.Context.(*services.LLMContext); ok {
			// Get last user message
			var userMessage string
			for i := len(llmContext.Messages) - 1; i >= 0; i-- {
				if llmContext.Messages[i].Role == "user" {
					userMessage = llmContext.Messages[i].Content
					break
				}
			}

			log.Printf("\n[LLM] Received user message: '%s'", userMessage)

			// Simulate LLM processing
			time.Sleep(100 * time.Millisecond)

			// Generate simulated response
			s.responseCounter++
			var response string
			switch s.responseCounter {
			case 1:
				response = "Hello! How can I help you today?"
			case 2:
				response = "Sure, I understand. What can I do for you?"
			case 3:
				response = "Of course! I'm here to answer your question."
			default:
				response = "I'm here to help!"
			}

			log.Printf("[LLM] Generated response: '%s'\n", response)

			// Emit LLMFullResponseStartFrame
			s.PushFrame(frames.NewLLMFullResponseStartFrame(), frames.Downstream)

			// Emit response as TextFrame chunks (simulating streaming)
			words := splitIntoWords(response)
			for _, word := range words {
				textFrame := frames.NewTextFrame(word + " ")
				s.PushFrame(textFrame, frames.Downstream)
				time.Sleep(50 * time.Millisecond) // Simulate streaming delay
			}

			// Emit LLMFullResponseEndFrame
			s.PushFrame(frames.NewLLMFullResponseEndFrame(), frames.Downstream)
		}
		return nil
	}

	// Pass through all other frames
	return s.PushFrame(frame, direction)
}

// Helper function to split text into words
func splitIntoWords(text string) []string {
	words := []string{}
	currentWord := ""

	for _, char := range text {
		if char == ' ' || char == ',' || char == '.' || char == '!' || char == '?' {
			if currentWord != "" {
				words = append(words, currentWord)
				currentWord = ""
			}
			if char != ' ' {
				words = append(words, string(char))
			}
		} else {
			currentWord += string(char)
		}
	}

	if currentWord != "" {
		words = append(words, currentWord)
	}

	return words
}
