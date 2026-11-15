# Flows reference

This page documents every exported type and function in the `flows` package: nodes, functions, actions, context strategies, state, the manager, the pipeline processor, and the JSON flow format.

A **flow** is a conversation scripted as a graph of **nodes**. Each node gives the LLM its instructions and a set of **functions** (tools the model can call); a function's result can move the conversation to the next node. **Actions** are things that happen when entering or leaving a node, such as speaking a fixed sentence or ending the call. The **context** is the conversation history the LLM reads.

Import path: `github.com/square-key-labs/strawgo-ai/src/flows`.

For a walkthrough, read the [flows guide](../guides/flows.md). This page is the lookup.

## Quick example

This function defines a two-node booking flow with a typed function, and `startFlow` wires it to a running pipeline.

```go
type bookingArgs struct {
	Date      string `json:"date" desc:"Requested date, YYYY-MM-DD"`
	PartySize int    `json:"party_size" desc:"Number of guests"`
}

func bookingFlow() *flows.Node {
	confirm := &flows.Node{
		Name:         "confirm",
		TaskMessages: []llm.Message{llm.Developer("Confirm the table for {{ booking.party_size }} on {{ booking.date }}, then say goodbye.")},
		PostActions:  []flows.Action{flows.EndConversation{Text: "Thanks for calling. Goodbye!"}},
	}
	return &flows.Node{
		Name:         "collect",
		RoleMessage:  "You are a friendly restaurant host on the phone. Keep answers short.",
		TaskMessages: []llm.Message{llm.Developer("Ask for the date and the number of guests, then call book_table.")},
		Functions: []flows.Function{
			flows.Typed("book_table", "Book a table once you have the date and party size.",
				func(ctx context.Context, args bookingArgs, m *flows.Manager) (any, *flows.Node, error) {
					m.State().Set("booking", args)
					return map[string]any{"status": "booked"}, confirm, nil
				}),
		},
	}
}

func startFlow(ctx context.Context, svc *llmservice.Service, task *pipeline.PipelineTask, conversation *llm.Context, proc *flows.Processor) (*flows.Manager, error) {
	m, err := flows.NewManager(flows.Config{LLM: svc, Task: task, Context: conversation, Processor: proc})
	if err != nil {
		return nil, err
	}
	if err := m.Initialize(ctx, bookingFlow()); err != nil {
		m.Close()
		return nil, err
	}
	return m, nil
}
```

`proc` is a `flows.NewProcessor()` placed in the pipeline between the LLM and the assistant aggregator, and `conversation` is the context the aggregators share.

## Manager

```text
func NewManager(cfg Config) (*Manager, error)
func (m *Manager) Initialize(ctx context.Context, initial *Node) error
func (m *Manager) SetNode(ctx context.Context, node *Node) error
func (m *Manager) CurrentNode() string
func (m *Manager) State() *State
func (m *Manager) Context() *llm.Context
func (m *Manager) CurrentContext() []llm.Message
func (m *Manager) LLM() LLM
func (m *Manager) Task() Task
func (m *Manager) RegisterAction(actionType string, handler ActionHandler)
func (m *Manager) Close()
```

Besides `Initialize`, the method you use most is `SetNode`, for example to move the conversation on from outside a function handler:

```go
func escalate(ctx context.Context, m *flows.Manager) error {
	if m.CurrentNode() == "escalation" {
		return nil
	}
	return m.SetNode(ctx, &flows.Node{
		Name:         "escalation",
		TaskMessages: []llm.Message{llm.Developer("Apologise and tell the caller a person will call back today.")},
		PostActions:  []flows.Action{flows.EndConversation{}},
	})
}
```

### Config

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `LLM` | `LLM` | none (required) | Registers the node's functions and runs summaries. `*llmservice.Service` and `*pipeline.LLMSwitcher` satisfy it. |
| `Task` | `Task` | none (required) | Where node frames are queued. `*pipeline.PipelineTask` satisfies it. |
| `Context` | `*llm.Context` | none (required) | The shared conversation context. |
| `Processor` | `*Processor` | none (required) | The flow processor in the pipeline. One processor serves one manager. |
| `ContextStrategy` | `ContextStrategy` | `Append` | Default for nodes that do not set their own. |
| `GlobalFunctions` | `[]Function` | none | Functions available in every node. Names must be unique. |
| `Actions` | `map[string]ActionHandler` | none | Handlers for `Custom` actions, by type. |

`NewManager` returns an error for a missing required field, an invalid strategy, a global function without a name or without a handler and `Next`, a duplicate global name, or a processor already bound to another manager.

The two interfaces:

```text
type LLM interface {
	RegisterFunction(name string, handler llm.FunctionHandler, opts ...llm.FunctionOptions)
	UnregisterFunction(name string)
	RunInference(ctx context.Context, snapshot llm.Snapshot, delta llm.Settings) (string, error)
}

type Task interface {
	QueueFrames(list []frames.Frame) error
	OnFrameReachedDownstream(fn pipeline.FrameHandler, examples ...frames.Frame)
	OnFrameReachedUpstream(fn pipeline.FrameHandler, examples ...frames.Frame)
}
```

### Methods

| Method | Description |
| --- | --- |
| `Initialize` | Start the manager's worker and enter `initial` (nil enters nothing). Can be called before `task.Run`; frames are buffered. Returns `ErrAlreadyInitialized` the second time. |
| `SetNode` | Move to `node`. Blocks until the node is set. Called from inside a function handler, it records the node as that handler's transition instead. Returns `ErrNotInitialized` before `Initialize`, and an error for nil or `NoResponse`. |
| `CurrentNode` | The current node's name. |
| `State` | The flow's key-value state. |
| `Context`, `CurrentContext` | The context, and a snapshot of its messages. |
| `LLM`, `Task` | The configured values. |
| `RegisterAction` | Add or replace a `Custom` action handler. |
| `Close` | Stop the worker; pending calls get `ErrClosed`. The manager also stops when an `EndFrame`, `StopFrame` or `CancelFrame` reaches the end of the pipeline. |

Errors: `ErrNotInitialized`, `ErrAlreadyInitialized`, `ErrClosed`.

### What entering a node does

1. Pending transitions from the previous node are dropped.
2. `{{ key }}` placeholders in `RoleMessage`, `RoleMessages`, `TaskMessages` and `TTSSay` text are filled from `State`. Dotted paths (`{{ booking.date }}`) read into maps and structs (through their JSON form). A missing key is an error; `\{{` keeps the braces literally.
3. `PreActions` run.
4. The node's functions plus the global functions are registered on the LLM; functions from the previous node that are gone are unregistered.
5. The manager queues: an `LLMUpdateSettingsFrame` with `RoleMessage` as the system instruction (reaching inactive services too), then an `LLMMessagesAppendFrame` (strategy `Append`) or `LLMMessagesUpdateFrame` (`Reset`, `ResetWithSummary`) with `RoleMessages`, any summary, and `TaskMessages`, then an `LLMSetToolsFrame`. These frames survive interruptions.
6. Unless `WaitForUser` is set, an `LLMRunFrame` makes the bot respond now.
7. `PostActions` run now, or, with `WaitForUser`, after the bot next stops speaking.

## Node

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Name` | `string` | `node-N` | Name used in logs, `CurrentNode`, errors and Mermaid. |
| `RoleMessage` | `string` | none | Sent as the LLM's system instruction. Empty leaves the instruction unchanged. |
| `RoleMessages` | `[]llm.Message` | none | Messages placed before the task messages. |
| `TaskMessages` | `[]llm.Message` | none | What the bot should do in this node. |
| `Functions` | `[]Function` | none | Functions available in this node, in addition to the global ones. Names must not repeat or clash with globals. |
| `PreActions` | `[]Action` | none | Run before the node's messages are queued. |
| `PostActions` | `[]Action` | none | Run after them. |
| `ContextStrategy` | `*ContextStrategy` | the manager's | Override for this node. |
| `WaitForUser` | `bool` | false | Do not run the LLM on entry; wait for the user to speak. |

`NoResponse` is a sentinel node: return it from a handler to end the call's function round without a transition and without running the LLM.

This node greets the caller with fixed text, then waits for them to speak instead of running the LLM on entry:

```go
func welcomeNode() *flows.Node {
	return &flows.Node{
		Name:         "welcome",
		RoleMessage:  "You are the front desk of a dental clinic.",
		TaskMessages: []llm.Message{llm.Developer("Find out whether the caller wants to book, move or cancel an appointment.")},
		PreActions:   []flows.Action{flows.TTSSay{Text: "Hello, you've reached the clinic."}},
		WaitForUser:  true,
	}
}
```

## Function

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Name` | `string` | none (required) | Tool name the LLM calls. |
| `Description` | `string` | none | Tool description for the LLM. |
| `Properties` | `map[string]any` | none | JSON Schema properties of the arguments. |
| `Required` | `[]string` | none | Required argument names. |
| `Handler` | `Handler` | none | Your code. Either `Handler` or `Next` is required. |
| `Next` | `*Node` | none | Node to move to when the handler returns no node (or when there is no handler). |
| `CancelOnInterruption` | `bool` | false | Cancel the handler's `ctx` if the user interrupts. |
| `Timeout` | `time.Duration` | 0 (the LLM service's timeout) | Handler timeout. |

```text
type Handler func(ctx context.Context, args Args, m *Manager) (result any, next *Node, err error)
type Args map[string]any
func Typed[T any](name, description string, fn func(ctx context.Context, args T, m *Manager) (any, *Node, error)) Function
func Decode[T any](args Args) (T, error)
func (f Function) Schema() llm.FunctionSchema
```

Handler results:

| Return | Effect |
| --- | --- |
| `err != nil` | The LLM gets `{"status":"error","error":"<message>"}` and stays in the node. |
| `next == nil` (and no `Next`) | The result goes to the LLM, which responds. A nil result becomes `{"status":"acknowledged"}`. |
| `next` is a node | The result is recorded without running the LLM; once every call from the same response has finished and the context is updated, the manager moves to `next`. |
| `next == NoResponse` | The result is recorded; no transition, no LLM run. |

`Typed` builds the schema from `T`'s fields: the name from the `json` tag (or the field name), `desc` tag as the description, `enum` tag (comma-separated) as allowed values. Fields without `omitempty` and not pointers are required. Strings, booleans, integers, floats, slices, maps, nested structs and interfaces are supported; other types panic. `Decode` converts `Args` into `T` through JSON.

A function with a JSON Schema written by hand, a handler that stays in the node, and a timeout:

```go
func lookupFunction() flows.Function {
	return flows.Function{
		Name:        "lookup_patient",
		Description: "Find the patient by date of birth.",
		Properties:  map[string]any{"dob": map[string]any{"type": "string", "description": "YYYY-MM-DD"}},
		Required:    []string{"dob"},
		Timeout:     5 * time.Second,
		Handler: func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
			dob, _ := args["dob"].(string)
			return map[string]any{"found": dob != ""}, nil, nil
		},
	}
}
```

## Actions

```text
type Action interface {
	Execute(ctx context.Context, m *Manager) error
}
type ActionHandler func(ctx context.Context, action Custom, m *Manager) error
```

| Action | Fields | Effect |
| --- | --- | --- |
| `TTSSay` | `Text string`, `SkipContext bool` | Speak `Text` (queues a `TTSSpeakFrame`). With `SkipContext` the text is not added to the context. Empty text is an error. |
| `EndConversation` | `Text string`, `SkipContext bool` | Optionally speak `Text`, then queue an `EndFrame`. Actions after it do not run. |
| `FunctionAction` | `Fn func(ctx context.Context, m *Manager) error` | Run `Fn` once all frames queued before it have reached the end of the pipeline. Errors and panics are logged. |
| `Custom` | `Type string`, `Params map[string]any` | Call the handler registered for `Type`. An unregistered type fails when the node is entered. |

Actions run in order. After a `TTSSay` or `FunctionAction`, the next action of another kind waits until the speech or function has been delivered.

A custom action is a name in the node plus a handler registered on the manager:

```go
func registerCRMAction(m *flows.Manager) []flows.Action {
	m.RegisterAction("log_to_crm", func(ctx context.Context, a flows.Custom, m *flows.Manager) error {
		log.Printf("CRM: %v (node %s)", a.Params["note"], m.CurrentNode())
		return nil
	})
	return []flows.Action{flows.Custom{Type: "log_to_crm", Params: map[string]any{"note": "caller verified"}}}
}
```

Use the returned actions as a node's `PreActions` or `PostActions`.

## Context strategies

```text
type ContextStrategy struct {
	Kind           StrategyKind
	SummaryPrompt  string
	SummaryTimeout time.Duration
}
```

| `Kind` | Effect on entering a node |
| --- | --- |
| `Append` | Add the node's messages to the existing context. |
| `Reset` | Replace the context with the node's messages. |
| `ResetWithSummary` | Ask the LLM (via `RunInference`, with `SummaryPrompt` as the system instruction) to summarize the conversation, then replace the context with the node's messages plus a developer message holding the summary. Requires `SummaryPrompt`. If the summary fails or is empty, falls back to `Append` with a warning. |

`SummaryTimeout` defaults to 5 s (`DefaultSummaryTimeout`). `StrategyKind.String()` gives `append`, `reset` or `reset_with_summary`.

This node starts from a clean context that holds only a summary of the conversation so far:

```go
func paymentNode() *flows.Node {
	return &flows.Node{
		Name:         "payment",
		TaskMessages: []llm.Message{llm.Developer("Take the card payment for the booking.")},
		ContextStrategy: &flows.ContextStrategy{
			Kind:          flows.ResetWithSummary,
			SummaryPrompt: "Summarise the booking details the caller has confirmed.",
		},
	}
}
```

## State

```text
func NewState() *State
func (s *State) Get(key string) (any, bool)
func (s *State) Set(key string, value any)
func (s *State) Delete(key string)
func (s *State) Update(fn func(values map[string]any))
func (s *State) Snapshot() map[string]any
func Get[T any](s *State, key string) (T, bool)
type Key[T any] string
func (k Key[T]) Get(s *State) (T, bool)
func (k Key[T]) Set(s *State, value T)
```

`State` is safe for concurrent use. `Update` runs `fn` with the map under the lock. `Get[T]` returns false if the key is missing or holds another type. `Key[T]` gives you a typed key: `const Party flows.Key[int] = "party"`.

A typed key keeps the value's type in one place:

```go
const partySize flows.Key[int] = "party_size"

func rememberParty(m *flows.Manager, n int) int {
	partySize.Set(m.State(), n)
	got, _ := partySize.Get(m.State())
	return got
}
```

## Processor

```text
func NewProcessor() *Processor
```

Place it after the LLM and before the assistant aggregator. It groups the function calls of one LLM response, holds back the LLM re-run while a transition is pending, and starts the transition once the results are in the context. Without it, `NewManager` fails, and if it never receives a `StartFrame` the manager logs a warning on the first transition.

The processor goes between the LLM service and the TTS, before the assistant aggregator:

```go
func flowPipeline(s *transports.Session, stt, llmSvc, tts processors.FrameProcessor, pair *aggregators.ContextAggregatorPair, proc *flows.Processor) *pipeline.Pipeline {
	return pipeline.NewPipeline([]processors.FrameProcessor{
		s.Input(), stt, pair.User(), llmSvc, proc, tts, s.Output(), pair.Assistant(),
	})
}
```

## JSON flows

```text
func ParseConfig(data []byte) (*FlowConfig, error)
func LoadJSON(data []byte, reg Registry) (*Flow, error)
func (c *FlowConfig) Build(reg Registry) (*Flow, error)
```

`LoadJSON` is `ParseConfig` followed by `Build`. Unknown JSON fields are rejected. `Build` checks the whole flow and returns a `*ValidationError` listing every `Problem{Node, Function, Message}` it found, including unknown nodes, missing handlers and nodes unreachable from `initial_node`.

`Registry`: `Handlers map[string]Handler` (function handlers by name) and `Actions map[string]ActionHandler` (action handlers by name).

`Flow` methods: `Initial() *Node`, `Node(name string) (*Node, bool)`, `GlobalFunctions() []Function`, `Config() *FlowConfig`, `Mermaid() string`. Pass `flow.Initial()` to `Initialize` and `flow.GlobalFunctions()` to `Config.GlobalFunctions`.

| JSON key | Go field | Type | Meaning |
| --- | --- | --- | --- |
| `initial_node` | `FlowConfig.InitialNode` | string | Required. |
| `nodes` | `FlowConfig.Nodes` | object of `NodeConfig` | Node definitions by name. |
| `global_functions` | `FlowConfig.GlobalFunctions` | array of `FunctionConfig` | Functions in every node. |
| `role_message` | `NodeConfig.RoleMessage` | string | As `Node.RoleMessage`. |
| `task_messages` | `NodeConfig.TaskMessages` | array of `{role, content}` | Roles `system`, `developer`, `user`, `assistant`. |
| `functions` | `NodeConfig.Functions` | array | Node functions. |
| `pre_actions`, `post_actions` | `NodeConfig.PreActions`, `PostActions` | array of `ActionConfig` | Actions. |
| `context_strategy` | `NodeConfig.ContextStrategy` | `"append"`, `"reset"`, `"reset_with_summary"` | Node strategy. |
| `summary_prompt` | `NodeConfig.SummaryPrompt` | string | Required with `reset_with_summary`. |
| `respond_immediately` | `NodeConfig.RespondImmediately` | bool | `false` sets `WaitForUser`. |
| `name`, `description`, `properties`, `required` | `FunctionConfig` | | As `Function`. |
| `handler` | `FunctionConfig.Handler` | string | Registry handler name; defaults to the function name. |
| `transition_only` | `FunctionConfig.TransitionOnly` | bool | No handler: the function only moves to `transition_to`, which must be a node name. Needs a description. |
| `transition_to` | `FunctionConfig.TransitionTo` | string, or `{field, cases, default}` | A node name, or a branch on a field of the handler's result object: `cases` maps values to node names, `default` is used otherwise. A node the handler returns takes precedence. |
| `cancel_on_interruption` | `FunctionConfig.CancelOnInterruption` | bool | As `Function`. |
| `timeout_secs` | `FunctionConfig.TimeoutSecs` | number | As `Function.Timeout`; negative is an error. |
| `type` | `ActionConfig.Type` | string | `tts_say`, `end_conversation`, `function`, or a custom type. |
| `text` | `ActionConfig.Text` | string | Text for `tts_say` and `end_conversation`; also passed to custom actions in `Params["text"]`. |
| `append_text_to_context` | `ActionConfig.AppendTextToContext` | bool | `false` sets `SkipContext`. |
| `handler` | `ActionConfig.Handler` | string | Registry action for `function` (required) and custom types (defaults to the type). |
| any other key | `ActionConfig.Params` | any | Passed to the action handler in `Custom.Params`. |

`Transition` has `Node`, `Field`, `Cases`, `Default`; it marshals to a string when `Field` is empty.

This loads a flow from a file and maps the JSON handler names to Go functions:

```go
func loadFlow(path string) (*flows.Flow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return flows.LoadJSON(data, flows.Registry{
		Handlers: map[string]flows.Handler{
			"check_availability": func(ctx context.Context, args flows.Args, m *flows.Manager) (any, *flows.Node, error) {
				return map[string]any{"available": true}, nil, nil
			},
		},
	})
}
```

Initialize the manager with `flow.Initial()`, and pass `flow.GlobalFunctions()` as `Config.GlobalFunctions`.

## Diagrams

```text
func Mermaid(start *Node, globals ...Function) string
```

Returns a Mermaid `flowchart TD` of the nodes reachable from `start` through `Next` and JSON transitions. Paste it into a ```` ```mermaid ```` block to render it on GitHub. Transitions that a Go handler returns at run time cannot be seen and are not drawn.

This prints a diagram you can paste into a Mermaid block:

```go
func printDiagram(start *flows.Node) {
	fmt.Println(flows.Mermaid(start))
}
```

## Next steps

- [Flows guide](../guides/flows.md): build a flow step by step.
- [Appointment booking recipe](../cookbook/appointment-booking-flow.md): a complete flow program.
- [Function calling guide](../guides/function-calling.md): how function calls run underneath.
- [Aggregators reference](aggregators.md): the assistant aggregator the flow processor sits before.
