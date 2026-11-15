# Flows

`src/flows` structures a conversation as a graph of nodes. Each node sets the system instruction, adds task messages to the context, and exposes a set of functions to the LLM. A function's handler can return the next node, which moves the conversation along. This is the Go counterpart of Pipecat Flows.

Status: the package compiles and has no tests. It has not been run against a live LLM.

## Concepts

| Type | Purpose |
| --- | --- |
| `flows.Node` | `Name`, `RoleMessage` (system instruction for the node), `RoleMessages`, `TaskMessages`, `Functions`, `PreActions`, `PostActions`, `ContextStrategy`, `WaitForUser` |
| `flows.Function` | `Name`, `Description`, `Properties`, `Required`, `Handler`, `Next` (static transition), `CancelOnInterruption`, `Timeout` |
| `flows.Handler` | `func(ctx, args flows.Args, m *flows.Manager) (result any, next *flows.Node, err error)` |
| `flows.Typed[T]` | builds a `Function` whose JSON schema is derived from struct `T` (`json`, `desc` and `enum` tags; non-pointer fields without `omitempty` are required) |
| `flows.Manager` | owns the current node, the registered functions and shared `State` |
| `flows.Processor` | pipeline processor the manager uses to see function-call results; place it after the LLM service and before the assistant aggregator |
| Actions | `flows.TTSSay{Text}`, `flows.EndConversation{Text}`, `flows.FunctionAction{Fn}`, `flows.Custom{Type, Params}` with `Manager.RegisterAction` |
| `flows.ContextStrategy` | `Append` (default), `Reset`, `ResetWithSummary` (needs `SummaryPrompt`; summarised with the flow's LLM through `RunInference`) |
| `flows.State`, `flows.Key[T]` | typed shared state across nodes |

How a node is entered (`Manager.SetNode`, or `Initialize` with the first node):

1. pre-actions run;
2. the node's functions plus the global functions are registered on the LLM service (functions no longer present are unregistered);
3. an `LLMUpdateSettingsFrame` (role message), an `LLMMessagesAppendFrame` or `LLMMessagesUpdateFrame` (depending on the context strategy) and an `LLMSetToolsFrame` are queued; all three survive interruptions, and the context aggregators apply them, so the single-writer rule holds;
4. unless `WaitForUser` is set, an `LLMRunFrame` is queued so the bot speaks first;
5. post-actions run; when `WaitForUser` is set they are deferred until the bot next stops speaking.

Text in role and task messages may reference state with `{{ key }}` (dotted paths allowed); a missing key is an error when the node is entered.

When a function returns a next node (or has `Next` set), its result is sent with `RunLLM` false and the transition happens once every function call from the same LLM response has completed. Returning `flows.NoResponse` as the next node suppresses the LLM rerun without transitioning. A handler error, or a call to a function that the current node does not offer, returns an error result to the LLM.

## Example

```go
package main

import (
	"context"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/flows"
	"github.com/square-key-labs/strawgo-ai/src/llm"
	"github.com/square-key-labs/strawgo-ai/src/pipeline"
	"github.com/square-key-labs/strawgo-ai/src/processors"
	"github.com/square-key-labs/strawgo-ai/src/processors/aggregators"
	"github.com/square-key-labs/strawgo-ai/src/services/cartesia"
	"github.com/square-key-labs/strawgo-ai/src/services/deepgram"
	"github.com/square-key-labs/strawgo-ai/src/services/openai"
	"github.com/square-key-labs/strawgo-ai/src/transports"
)

type nameArgs struct {
	Name string `json:"name" desc:"The caller's first name"`
}

var callerName flows.Key[string] = "caller_name"

func buildFlow() *flows.Node {
	goodbye := &flows.Node{
		Name:         "goodbye",
		TaskMessages: []llm.Message{llm.Developer("Thank {{ caller_name }} in one sentence and say goodbye.")},
		PostActions:  []flows.Action{flows.EndConversation{}},
	}
	return &flows.Node{
		Name:         "greeting",
		RoleMessage:  "You are a friendly receptionist on a phone call. Keep answers short.",
		TaskMessages: []llm.Message{llm.Developer("Greet the caller and ask for their first name.")},
		Functions: []flows.Function{
			flows.Typed("record_name", "Store the caller's first name once they have said it.",
				func(ctx context.Context, args nameArgs, m *flows.Manager) (any, *flows.Node, error) {
					callerName.Set(m.State(), args.Name)
					return map[string]any{"saved": args.Name}, goodbye, nil
				}),
		},
	}
}

func runCall(ctx context.Context, s *transports.Session) error {
	stt := deepgram.NewSTTService(deepgram.STTConfig{APIKey: os.Getenv("DEEPGRAM_API_KEY"), Model: "nova-3", Language: "en"})
	llmSvc := openai.NewLLMService(openai.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY"), Model: "gpt-4.1-mini"})
	tts := cartesia.NewTTSService(cartesia.TTSConfig{APIKey: os.Getenv("CARTESIA_API_KEY"), VoiceID: os.Getenv("CARTESIA_VOICE_ID"), Model: "sonic-2", Language: "en"})

	conversation := llm.NewContext()
	pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})
	flowProc := flows.NewProcessor()

	pipe := pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, flowProc, tts, s.Output(), pair.Assistant(),
	})
	task := pipeline.NewPipelineTask(pipe)

	manager, err := flows.NewManager(flows.Config{
		LLM:       llmSvc,
		Task:      task,
		Context:   conversation,
		Processor: flowProc,
	})
	if err != nil {
		return err
	}
	defer manager.Close()
	if err := manager.Initialize(ctx, buildFlow()); err != nil {
		return err
	}
	return task.Run(ctx)
}

func main() {
	_ = runCall
}
```

Serve `runCall` with `transports.NewServer` as in the [README quickstart](../README.md#quickstart). `Initialize` may be called before `task.Run`: the frames it queues are held by the task until the pipeline starts.

`flows.Config.LLM` accepts anything with `RegisterFunction`, `UnregisterFunction` and `RunInference`, which `*llmservice.Service` provides. `GlobalFunctions` are offered in every node; `Actions` registers custom action handlers by type name.

## JSON flow definitions

`flows.LoadJSON(data, flows.Registry{Handlers, Actions})` builds a `*flows.Flow` from a Pipecat-Flows-style JSON document. Function `handler` and action `handler` fields name entries in the registry; `transition_to` is a node name or a `{field, cases, default}` branch on an argument value; `transition_only` functions only transition. `flow.Initial()` returns the starting node, `flow.GlobalFunctions()` the global functions, `flow.Mermaid()` a Mermaid diagram (`flows.Mermaid(node)` does the same for Go-defined nodes).

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/square-key-labs/strawgo-ai/src/flows"
)

const flowJSON = `{
  "initial_node": "greeting",
  "nodes": {
    "greeting": {
      "role_message": "You are a friendly receptionist on a phone call.",
      "task_messages": [{"role": "developer", "content": "Greet the caller and ask for their first name."}],
      "functions": [{
        "name": "record_name",
        "description": "Store the caller's first name.",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "handler": "record_name",
        "transition_to": "goodbye"
      }]
    },
    "goodbye": {
      "task_messages": [{"role": "developer", "content": "Thank the caller by name and say goodbye."}],
      "post_actions": [{"type": "end_conversation"}]
    }
  }
}`

func main() {
	flow, err := flows.LoadJSON([]byte(flowJSON), flows.Registry{
		Handlers: map[string]flows.Handler{
			"record_name": func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				m.State().Set("caller_name", args["name"])
				return map[string]any{"saved": args["name"]}, nil, nil
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(flow.Initial().Name)
	fmt.Println(flow.Mermaid())
}
```

Pass `flow.Initial()` to `Manager.Initialize` and `flow.GlobalFunctions()` to `flows.Config.GlobalFunctions`.
