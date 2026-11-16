# Quickstart

In this quickstart you build the smallest Strawgo bot that talks back: a text chat in your terminal. It needs one API key and no phone, microphone or speaker. Then you give it a tool (a Go function the model can call) in 15 lines.

## Before you begin

- Go 1.25.9 or newer, and an empty Go module (`go mod init example.com/mybot`). You add Strawgo to it after you write the program. See [Installation](installation.md).
- An OpenAI API key, exported as `OPENAI_API_KEY`. Any server that speaks the OpenAI Chat Completions API also works; see [Use another LLM server](#use-another-llm-server).

A few terms first. A frame is one unit of data moving through your bot, such as a piece of text. A processor receives frames, acts on them, and passes them on. A pipeline is a chain of processors. A task runs one pipeline from start to finish. [Core concepts](core-concepts.md) explains each in depth.

## Try the bundled example

If you cloned the repository, a slightly fuller version of this bot is already there as `examples/text_flow.go`. It also reads `OPENAI_MODEL` to pick the model and `SHOW_METRICS` to log metrics. The file starts with `//go:build ignore`, which keeps it out of `go build ./...` so that it does not clash with the other example programs. `go run` with the file name still runs it:

```sh
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
go run examples/text_flow.go
```

The rest of this page builds the bot in your own module, so you can change it.

## Step 1: a text chat bot

Create `main.go` in your module and paste this program:

```go
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
)

type console struct {
	*processors.BaseProcessor
	replied chan struct{}
	printed bool
}

func newConsole() *console {
	c := &console{replied: make(chan struct{}, 1)}
	c.BaseProcessor = processors.NewBaseProcessor("Console", c)
	return c
}

func (c *console) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.LLMTextFrame:
		fmt.Print(f.Text)
		c.printed = true
	case *frames.LLMFullResponseEndFrame:
		if c.printed {
			fmt.Println()
			c.printed = false
			c.done()
		}
	}
	return c.PushFrame(frame, direction)
}

func (c *console) done() {
	select {
	case c.replied <- struct{}{}:
	default:
	}
}

func main() {
	logger.SetLevel(logger.WARN)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("set OPENAI_API_KEY")
	}
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       apiKey,
		BaseURL:      os.Getenv("OPENAI_BASE_URL"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a concise, friendly assistant.",
	})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	out := newConsole()
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), llmSvc, out, pair.Assistant()})
	task := pipeline.NewPipelineTask(pipe)
	task.OnError(func(err error) {
		fmt.Fprintln(os.Stderr, "error:", err)
		out.done()
	})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		in := bufio.NewScanner(os.Stdin)
		for fmt.Print("> "); in.Scan(); fmt.Print("> ") {
			text := strings.TrimSpace(in.Text())
			if text == "" {
				continue
			}
			msg := []llm.Message{llm.User(text)}
			if err := task.QueueFrame(frames.NewLLMMessagesAppendFrame(msg, true)); err != nil {
				return
			}
			select {
			case <-out.replied:
			case <-task.Done():
				return
			}
		}
		_ = task.StopWhenDone()
	}()

	if err := task.Run(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nconversation had %d messages\n", conversation.Len())
}
```

Fetch Strawgo from `main`, add its indirect dependencies, and run the program. Run the first command only now that `main.go` exists, because `go mod tidy` drops modules that no code imports yet ([why](installation.md#add-strawgo-to-your-module)):

```sh
go get github.com/square-key-labs/strawgo-ai@latest && go mod tidy
go run .
```

You see a `>` prompt. Type a message and press Enter:

```text
> hi
Hello! How can I help?
>
```

Press Ctrl+D to end the conversation cleanly, or Ctrl+C to quit at once. On exit the program prints how many messages the conversation holds.

### How it works

**The console processor.** `console` is your own processor. Embedding `*processors.BaseProcessor` gives it the queues, goroutines and linking that every processor needs; `newConsole` passes the struct itself to `processors.NewBaseProcessor` so the base calls your `HandleFrame`.

**`HandleFrame`.** Strawgo calls it once for every frame that reaches the processor. The LLM streams its answer as many small `LLMTextFrame`s, so the console prints each piece as it arrives. `LLMFullResponseEndFrame` marks the end of one answer; if anything was printed, the console ends the line and signals `replied`. The last line, `c.PushFrame(frame, direction)`, passes every frame on to the next processor. A processor must forward frames it does not consume, or the pipeline stalls.

**`done`.** It sends on `replied` without blocking. The channel has room for one signal, so a second signal before the first is read is simply dropped.

**The LLM service.** `openai.NewLLMService` returns a processor that sends the conversation to the model and turns the streamed reply into frames. `BaseURL` is empty unless you set `OPENAI_BASE_URL`, and an empty value means OpenAI's own endpoint. `SystemPrompt` is the instruction the model sees before every conversation.

**The conversation context.** `llm.NewContext()` holds the conversation: the messages so far and the tools the model may call. `aggregators.NewContextAggregatorPair` returns two processors that own it. The user aggregator (`pair.User()`) adds the user's messages and asks the LLM to answer. The assistant aggregator (`pair.Assistant()`) adds the bot's replies once they have passed through the rest of the pipeline. Only these two write to the context while the bot runs.

**The pipeline.** `pipeline.NewPipeline` links the four processors in order: user aggregator, LLM, console, assistant aggregator. Frames flow from left to right ("downstream").

**The task.** `pipeline.NewPipelineTask` wraps the pipeline and runs it. `task.OnError` is called for every error that reaches the top of the pipeline. If the LLM request fails, the program prints the error and calls `out.done()` so that the input loop does not wait forever for a reply.

**The input loop.** It runs in its own goroutine, because `task.Run` blocks. For each line you type, it queues an `LLMMessagesAppendFrame` on the task. That frame tells the user aggregator to append a user message (`llm.User(text)`) and, because the second argument is `true`, to run the LLM. The loop then waits until the reply has printed, or until the task has finished. When standard input closes (Ctrl+D), `task.StopWhenDone()` queues an end frame, which lets the pipeline finish its work and stop.

**Running.** `task.Run(ctx)` starts every processor, runs until the task ends, and then stops them. Ctrl+C cancels `ctx`, which cancels the task. `conversation.Len()` returns the number of messages in the context. The system prompt is not one of them: the LLM service adds it to each request from its own settings.

## Step 2: add a tool

A tool, also called a function call, lets the model ask your program to do something: look up data, book a slot, hang up. You describe the tool with a JSON schema, and you register a Go handler that runs when the model calls it.

Add these lines right after `conversation := llm.NewContext()`:

```go
conversation.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{
	Name:        "get_weather",
	Description: "Current weather for a city.",
	Properties: map[string]any{
		"city": map[string]any{"type": "string", "description": "City name, e.g. Paris"},
	},
	Required: []string{"city"},
}}})
llmSvc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
	city, _ := p.Arguments["city"].(string)
	if city == "" {
		return llm.FunctionResult{}, fmt.Errorf("city is required")
	}
	return llm.FunctionResult{Value: map[string]any{"city": city, "temperature_c": 21, "conditions": "partly cloudy"}}, nil
})
```

Run the program again and ask about the weather:

```text
> what is the weather in Paris?
It is 21 degrees and partly cloudy in Paris.
>
```

The model's wording varies, but the facts come from your handler.

### How it works

**`conversation.SetTools`.** It tells the model which tools exist. Each `llm.FunctionSchema` has a name, a description the model reads to decide when to call it, and its parameters: `Properties` is a JSON schema object per parameter, and `Required` lists the ones the model must fill in. You set tools here, before the task runs. While a pipeline runs, change tools by queueing a `frames.NewLLMSetToolsFrame` instead, so that only the aggregators write to the context.

**`llmSvc.RegisterFunction`.** It connects a tool name to a Go handler. The handler gets a `context.Context` and an `llm.FunctionCallParams`, whose `Arguments` map holds the model's arguments decoded from JSON.

**The handler's result.** `llm.FunctionResult{Value: ...}` is sent back to the model as the tool result. Any value that encodes to JSON works. Leaving `RunLLM` nil means "run the LLM again with the result", so the model answers the user in words. If the handler returns an error, the model receives `{"error": "city is required"}` as the result instead.

**What happens in the pipeline.** The model's first response is a tool call, not text, so the console prints nothing and keeps waiting. The LLM service runs your handler, the assistant aggregator records the call and its result in the context, and the LLM runs again. The second response is text, which the console prints.

<details>
<summary>The complete program with the tool</summary>

```go
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"

	"github.com/square-key-labs/strawgo-ai/src/frames"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/logger"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
)

type console struct {
	*processors.BaseProcessor
	replied chan struct{}
	printed bool
}

func newConsole() *console {
	c := &console{replied: make(chan struct{}, 1)}
	c.BaseProcessor = processors.NewBaseProcessor("Console", c)
	return c
}

func (c *console) HandleFrame(ctx context.Context, frame frames.Frame, direction frames.FrameDirection) error {
	switch f := frame.(type) {
	case *frames.LLMTextFrame:
		fmt.Print(f.Text)
		c.printed = true
	case *frames.LLMFullResponseEndFrame:
		if c.printed {
			fmt.Println()
			c.printed = false
			c.done()
		}
	}
	return c.PushFrame(frame, direction)
}

func (c *console) done() {
	select {
	case c.replied <- struct{}{}:
	default:
	}
}

func main() {
	logger.SetLevel(logger.WARN)
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("set OPENAI_API_KEY")
	}
	llmSvc := openai.NewLLMService(openai.LLMConfig{
		APIKey:       apiKey,
		BaseURL:      os.Getenv("OPENAI_BASE_URL"),
		Model:        "gpt-4.1-mini",
		SystemPrompt: "You are a concise, friendly assistant.",
	})

	conversation := llm.NewContext()
	conversation.SetTools(llm.Tools{Functions: []llm.FunctionSchema{{
		Name:        "get_weather",
		Description: "Current weather for a city.",
		Properties: map[string]any{
			"city": map[string]any{"type": "string", "description": "City name, e.g. Paris"},
		},
		Required: []string{"city"},
	}}})
	llmSvc.RegisterFunction("get_weather", func(ctx context.Context, p llm.FunctionCallParams) (llm.FunctionResult, error) {
		city, _ := p.Arguments["city"].(string)
		if city == "" {
			return llm.FunctionResult{}, fmt.Errorf("city is required")
		}
		return llm.FunctionResult{Value: map[string]any{"city": city, "temperature_c": 21, "conditions": "partly cloudy"}}, nil
	})
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	out := newConsole()
	pipe := pipeline.NewPipeline([]processors.FrameProcessor{pair.User(), llmSvc, out, pair.Assistant()})
	task := pipeline.NewPipelineTask(pipe)
	task.OnError(func(err error) {
		fmt.Fprintln(os.Stderr, "error:", err)
		out.done()
	})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		in := bufio.NewScanner(os.Stdin)
		for fmt.Print("> "); in.Scan(); fmt.Print("> ") {
			text := strings.TrimSpace(in.Text())
			if text == "" {
				continue
			}
			msg := []llm.Message{llm.User(text)}
			if err := task.QueueFrame(frames.NewLLMMessagesAppendFrame(msg, true)); err != nil {
				return
			}
			select {
			case <-out.replied:
			case <-task.Done():
				return
			}
		}
		_ = task.StopWhenDone()
	}()

	if err := task.Run(ctx); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nconversation had %d messages\n", conversation.Len())
}
```

</details>

## Use another LLM server

The OpenAI service works with any server that implements the OpenAI Chat Completions API. Set `OPENAI_BASE_URL` to the server's `/v1` URL and, if needed, change the `Model` string in the program. For a local [Ollama](https://ollama.com) server, the URL is `http://localhost:11434/v1`. `OPENAI_API_KEY` must still be non-empty for the program to start, but Ollama ignores its value. Strawgo also has dedicated packages for many vendors, including `ollama`; see [LLM providers](../providers/llm.md).

## Next steps

- [Give the bot a voice on a phone line](first-phone-bot.md)
- [Learn the core concepts behind this program](core-concepts.md)
- [Go further with function calling](../guides/function-calling.md)
- [Manage the conversation context](../guides/conversation-context.md)
