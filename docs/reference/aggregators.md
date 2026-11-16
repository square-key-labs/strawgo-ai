# Aggregators reference

This page documents the `processors/aggregators` package: the user and assistant context aggregators and their params, context summarization, idle detection, the DTMF aggregator, sentence aggregation, and LLM-judged user-turn completion.

An **aggregator** is a processor that collects pieces of something (transcript fragments, LLM tokens, keypad digits) and emits them as one unit. The **context** is the conversation history the LLM reads. A **turn** is one stretch of the user or the bot talking. An **interruption** (barge-in) is the user talking over the bot.

Import path: `github.com/square-key-labs/strawgo-ai/src/processors/aggregators`.

## Quick example

The usual setup is one context shared by a user aggregator (before the LLM) and an assistant aggregator (after the output transport). This function builds the pair with idle handling and automatic summarization that uses the same LLM service.

```go
func newContextPair(svc *llmservice.Service, systemPrompt string) *aggregators.ContextAggregatorPair {
	conversation := llm.NewContext(llm.System(systemPrompt))
	return aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{
		User: aggregators.UserAggregatorParams{
			UserIdleTimeout: 20 * time.Second,
			OnUserIdle:      func() { log.Print("user is quiet") },
		},
		Assistant: aggregators.AssistantAggregatorParams{
			EnableAutoContextSummarization: true,
			AutoSummarizationConfig:        aggregators.LLMAutoContextSummarizationConfig{MaxUnsummarizedMessages: 40},
			Summarize: func(ctx context.Context, snap llm.Snapshot) (string, error) {
				return svc.RunInference(ctx, snap, llm.Settings{})
			},
		},
	})
}
```

Place them like this: `transport input, STT, pair.User(), LLM, TTS, transport output, pair.Assistant()`.

## ContextAggregatorPair

```text
func NewContextAggregatorPair(c *llm.Context, params ContextAggregatorPairParams) *ContextAggregatorPair
func (p *ContextAggregatorPair) User() *LLMUserAggregator
func (p *ContextAggregatorPair) Assistant() *LLMAssistantAggregator
func (p *ContextAggregatorPair) Context() *llm.Context
```

A nil `c` creates an empty context. `ContextAggregatorPairParams` has two fields, `User UserAggregatorParams` and `Assistant AssistantAggregatorParams`.

Keep the pair's context to read the conversation after the call, for example to store a transcript:

```go
func printTranscript(pair *aggregators.ContextAggregatorPair) {
	for _, m := range pair.Context().Messages() {
		fmt.Printf("%s: %s\n", m.Role, m.Text())
	}
}
```

## User aggregator

```text
func NewLLMUserAggregator(c *llm.Context, strategies turns.UserTurnStrategies) *LLMUserAggregator
func NewLLMUserAggregatorWithParams(c *llm.Context, params UserAggregatorParams) *LLMUserAggregator
func (u *LLMUserAggregator) UserTurnActive() bool
func (u *LLMUserAggregator) Muted() bool
```

The user aggregator decides when a user turn starts and stops (using [turn strategies](turns.md)), collects the final transcriptions in between, adds them to the context as one user message, and pushes an `LLMContextFrame` downstream to run the LLM.

The callbacks are the simplest way to react to the caller's turns:

```go
func loggingUserParams() aggregators.UserAggregatorParams {
	return aggregators.UserAggregatorParams{
		OnUserTurnStarted: func() { log.Print("caller started a turn") },
		OnUserTurnStopped: func(text string) { log.Printf("caller said: %s", text) },
		OnUserMuteChanged: func(muted bool) { log.Printf("muted: %v", muted) },
	}
}
```

### UserAggregatorParams

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Strategies` | `turns.UserTurnStrategies` | empty: `turns.DefaultUserTurnStrategies()` | Start, stop and mute strategies. |
| `UserTurnStopTimeout` | `time.Duration` | 5 s (`DefaultUserTurnStopTimeout`) | Safety net: end an active turn when the user is not speaking and nothing (speech, transcription) has happened for this long, and no stop strategy has a pending deadline. Negative disables. |
| `UserIdleTimeout` | `time.Duration` | 0 (off) | Push `UserIdleTimeoutFrame` and call `OnUserIdle` when the user stays silent this long after the bot finishes. |
| `OnUserTurnStarted` | `func()` | none | Called when a user turn starts. |
| `OnUserTurnStopped` | `func(text string)` | none | Called with the committed text when a turn ends with text. |
| `OnUserIdle` | `func()` | none | Called on idle timeout. |
| `OnUserMuteChanged` | `func(muted bool)` | none | Called when mute strategies mute or unmute the user. |
| `EmptyUserTurn` | `*turns.EmptyUserTurnConfig` | nil: `turns.DefaultEmptyUserTurnConfig()` | What to do when a turn ends with no transcript. See [Empty user turns](#empty-user-turns). |

Which strategies apply, in order:

1. A non-empty `TurnStrategies` in the task config (`StartFrame`) always replaces the current set.
2. Otherwise the `Strategies` you passed.
3. If you passed none, a `ServiceMetadataFrame` from an STT service with its own turn detection (Deepgram Flux and others) replaces them.
4. Otherwise `turns.DefaultUserTurnStrategies()`.

### Behaviour

| Event | Action |
| --- | --- |
| Turn start (a start strategy fires) | If the strategy enables interruptions, the task allows them, and the bot is speaking or a response is pending, broadcast an `InterruptionFrame`. Call `OnUserTurnStarted`. |
| Final `TranscriptionFrame` | Append its text to the turn. Interim, eager and translation frames are not added. |
| Turn stop (a stop strategy fires, or the stop timeout) | Join the text; if non-empty, add `llm.User(text)` to the context, broadcast `LLMContextUserTurnFrame`, push `LLMContextFrame` downstream, call `OnUserTurnStopped`. If empty, follow `EmptyUserTurn`; when that does nothing, start the idle timer. |
| `LLMRunFrame` | Push `LLMContextFrame` downstream. |
| `LLMMessagesAppendFrame`, `LLMMessagesUpdateFrame`, `LLMMessagesTransformFrame`, `LLMSetToolChoiceFrame` | Apply to the context and consume. If the frame asks to run the LLM, push `LLMContextFrame`. |
| `LLMSetToolsFrame` | Apply and forward. |
| `LLMContextFrame` arriving upstream | Push it downstream (the assistant aggregator uses this to re-run the LLM after a tool call). |
| `LLMServiceMetadataFrame` | Follow the active LLM: empty-turn recovery is off while `IsRealtimeService` is true and back on when a non-realtime LLM announces itself; forward. |
| Mute on | Broadcast `UserMuteStartedFrame`; until unmuted, drop audio, transcription, backchannel, proposed-speaking and interruption frames. User and VAD started/stopped speaking frames are still forwarded and still update the speaking state, but do not start a turn. |
| Smart-turn metrics | Push a `MetricsFrame` with `TurnMetricsData`. |

### Empty user turns

A user turn can start on voice activity alone (a cough, line noise, a short "hmm") and end with no transcript. Nothing is written to the context for it and, on its own, the LLM does not run. If the turn interrupted the bot, the bot would stay silent with its reply cut off. `UserAggregatorParams.EmptyUserTurn` decides what happens instead. The two cases differ:

- Interrupted: the turn started while interruptions were enabled and the bot was not waiting for the user (it was thinking, speaking or running a function call, or had not spoken yet). By default the aggregator adds `llm.Developer(turns.DefaultEmptyUserTurnInterruptedPrompt)` to the context and pushes `LLMContextFrame`, so the bot asks the caller to repeat.
- Idle: the bot had finished and was waiting. By default the turn gets no answer, because answering what may be noise would be intrusive.

| Field of `turns.EmptyUserTurnConfig` | Default | Meaning |
| --- | --- | --- |
| `Disabled` | false | Leave every empty turn unanswered, and never resume. |
| `InterruptedPrompt` | `""`: `turns.DefaultEmptyUserTurnInterruptedPrompt` | Developer message for an empty turn that interrupted the bot. |
| `LeaveInterruptedUnanswered` | false | Do not prompt for an interrupted empty turn. |
| `IdlePrompt` | `""`: no answer | Developer message for an empty turn while the bot was idle. |
| `MaxConsecutiveRecoveries` | 0: 1 | How many empty turns in a row are answered (prompt or resume). The count resets when a turn has a transcript, and when a resumed reply plays to the end. |
| `ResumeInterruptedSpeech` | false | For an interrupted empty turn, have the bot continue what it was saying instead of prompting. |

No recovery runs when the bot is already going to answer: while the caller is muted, a function call is in progress, the bot is speaking, a response has been requested and not yet spoken, or the LLM was run during the turn (including a function result that asks to run it).

A recovery prompt is only there to get one answer. The aggregator removes the developer messages it added when the next user message is written to the context and when the session ends (`EndFrame` or `CancelFrame`), so they do not pile up in a noisy call or stay in a saved context.

A realtime (speech-to-speech) LLM service hears the caller's audio directly, so an empty transcript does not mean it missed anything. Every LLM service pushes an `LLMServiceMetadataFrame` upstream when it starts, and again when a `ServiceSwitcher` makes it the active service: `openai_realtime.LLMService` and `gemini.LiveService` with `IsRealtimeService` true, services built on `llmservice.Service` with it false. The switcher drops the announcement of an inactive branch. The user aggregator follows the latest announcement: while the active LLM is realtime it treats empty-turn recovery as if `Disabled` were set, and turns it back on when a cascaded LLM announces itself, for example after switching away from the realtime service. The aggregator follows the most recent announcement, and only `ServiceSwitcher` hides announcements from inactive services. If a realtime LLM and a text LLM sit in separate branches of a `ParallelPipeline`, either can announce last, so set `EmptyUserTurn.Disabled` yourself in that layout.

With `ResumeInterruptedSpeech`, the user aggregator asks the assistant aggregator of the same `ContextAggregatorPair` for the part of the interrupted reply the caller did not hear, and pushes it downstream as a `TTSSpeakFrame`. The remainder is the text the TTS had queued for the reply (`TTSQueuedTextFrame`) after the last `TTSTextFrame` that was played. It starts again at the beginning of the sentence that was cut, or at the next sentence when the cut fell between sentences; sentences are split the same way the TTS splits them (`tts.SplitSentences`), so abbreviations such as "Dr." do not end a sentence. The context keeps a single assistant message: when the resumed speech ends, or is itself interrupted, the interrupted message is replaced by the heard part up to that sentence boundary followed by what was spoken of the resumed text. A resume counts as a recovery.

Resume needs a TTS whose `TTSTextFrame`s follow the audio they describe:

| TTS built on `tts.Base` | Resume |
| --- | --- |
| HTTP providers (`tts.NewHTTP`) | yes; a sentence counts as heard once its audio has played |
| Streaming providers with `WordTimestamps` | yes; heard up to the last word played |
| Streaming providers without `WordTimestamps` | no; their sentence `TTSTextFrame` is sent before the sentence's audio, so the played text runs ahead of what was heard |

The aggregator falls back to the prompt behaviour above when it cannot work out the remainder:

- the TTS is a streaming provider without word timestamps, is not built on `tts.Base`, or there is no TTS;
- the LLM response had not finished when the bot was interrupted (the TTS had not received `LLMFullResponseEndFrame`), because the rest of the response, including any tool call, was never generated; running the LLM again is the only way to get it;
- a function call was in progress when the bot was interrupted, or a function result was waiting to be answered (its LLM run was deferred until the bot stopped speaking), so the prompt path runs the LLM and the result is voiced;
- the played text does not line up with the queued text;
- the aggregators were not created as a pair.

A turn with any final transcript never resumes. Only the interrupted reply is resumed: text the TTS had queued for a later context, such as a `TTSSpeakFrame` pushed behind the reply, is dropped by the interruption as usual and is not spoken by the resume.

```go
func resumingUserParams() aggregators.UserAggregatorParams {
	return aggregators.UserAggregatorParams{
		EmptyUserTurn: &turns.EmptyUserTurnConfig{
			ResumeInterruptedSpeech: true,
			IdlePrompt:              "The user may have said something, but it was not recognized. Briefly ask them to repeat it.",
		},
	}
}
```

## Assistant aggregator

```text
func NewLLMAssistantAggregator(c *llm.Context, params *AssistantAggregatorParams) *LLMAssistantAggregator
func DefaultAssistantAggregatorParams() *AssistantAggregatorParams
func (a *LLMAssistantAggregator) HasFunctionCallsInProgress() bool
func (a *LLMAssistantAggregator) Summarizer() *LLMContextSummarizer
func (a *LLMAssistantAggregator) Reset() error
```

The assistant aggregator records what the bot said, its thoughts, and its tool calls and results, and re-runs the LLM when a tool result asks for it. It goes after the output transport so that, when the bot is interrupted, it records the reply up to the last `TTSTextFrame` that was played (per word or per sentence, depending on the TTS).

`OnAssistantTurnStopped` reports each committed bot turn, and whether the caller cut it short:

```go
func loggingAssistantParams() aggregators.AssistantAggregatorParams {
	return aggregators.AssistantAggregatorParams{
		OnAssistantTurnStopped: func(text string, interrupted bool) {
			log.Printf("bot said (interrupted=%v): %s", interrupted, text)
		},
	}
}
```

### AssistantAggregatorParams

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `EnableAutoContextSummarization` | `bool` | false | Check after each committed turn and tool result whether to summarize. Needs `Summarize`. |
| `AutoSummarizationConfig` | `LLMAutoContextSummarizationConfig` | zero | When and how to summarize; see below. |
| `Summarize` | `SummaryFunc` | none | `func(ctx context.Context, snapshot llm.Snapshot) (string, error)`. Produces the summary. Without it there is no summarizer. |
| `SummarizationTimeout` | `time.Duration` | 120 s (`DefaultSummarizationTimeout`) | Limit for one `Summarize` call. |
| `OnAssistantTurnStopped` | `func(text string, interrupted bool)` | none | Called when an assistant turn is committed. |
| `OnAssistantThought` | `func(text string)` | none | Called when a thought ends. |
| `OnSummaryApplied` | `func(beforeCount, afterCount int)` | none | Called after a summary replaces messages. |
| `AddToolChangeMessages` | `bool` | false | On `LLMSetToolsFrame`, add a developer message naming the functions added and removed. `NewContextAggregatorPair` applies it to the user aggregator too; otherwise call `SetAddToolChangeMessages(true)` on it. The text comes from `ToolChangeMessage`. |
| `WhenIdleUserTurnTimeout` | `time.Duration` | 5 s (`DefaultUserTurnStopTimeout`) | For results with `llm.ResultDeliveryWhenIdle`: how long after the user stops speaking to assume their turn produced no reply and deliver the result. |

`DefaultAssistantAggregatorParams()` returns all zero values. `NewLLMAssistantAggregator` with nil params uses it.

### Behaviour

| Frame | Action |
| --- | --- |
| `LLMFullResponseStartFrame`, `TTSStartedFrame` | Start collecting a turn. |
| `TTSTextFrame` | Collect spoken text (preferred, because it reflects what was heard). |
| `TTSQueuedTextFrame` | Remember the text the TTS queued for a context, whether its playback is tracked, and whether the context is complete, to work out the unheard remainder after an interruption. Consumed. |
| `TTSStoppedFrame` | Commit a resumed reply (see [Empty user turns](#empty-user-turns)) when its context stops. |
| `LLMTextFrame`, `TextFrame`, `AggregatedTextFrame` | Collect text, used when no `TTSTextFrame` arrived. |
| `LLMFullResponseEndFrame`, `LLMAssistantPushAggregationFrame`, `EndFrame` | Commit: add `llm.Assistant(text)`, push `LLMContextFrame` and `LLMContextAssistantTimestampFrame` downstream, broadcast `LLMContextAssistantTurnFrame`, call `OnAssistantTurnStopped(text, false)`. |
| `InterruptionFrame`, `CancelFrame` | Commit what was collected so far with `interrupted` true. |
| `LLMThoughtStartFrame` … `LLMThoughtEndFrame` | Add an assistant `ThoughtPart` when `AppendToContext` was set. |
| `FunctionCallInProgressFrame` | Add the tool call and a tool result of `IN_PROGRESS`. |
| `FunctionCallResultFrame` | Replace the result. If the result asks to run the LLM, push `LLMContextFrame` upstream once no sibling call from the same response is pending, the user is not speaking, and the bot has stopped speaking. `Delivery` changes the timing: `interrupt` pushes `InterruptionTaskFrame` upstream when the bot is speaking or a reply is streaming, then runs the LLM when the `InterruptionFrame` arrives; `when_idle` also waits for the user's turn to be answered; `silent` never runs the LLM. Intermediate results (`Properties.IsFinal` false) are added as developer messages. |
| `FunctionCallCancelFrame` | Replace the result with the frame's `Result`, or `llm.ToolResultInterrupted` when it is empty. |
| `LLMRunFrame` | Push `LLMContextFrame` upstream. |
| `LLMMessagesAppendFrame` and the other context-edit frames | Apply; re-run the LLM if asked. |
| `LLMSummarizeContextFrame` | Summarize now, ignoring thresholds. |

## Context summarization

**Summarization** replaces older messages with one short summary so the context stays within the model's limits.

```text
func NewLLMContextSummarizer(config LLMAutoContextSummarizationConfig, summarize SummaryFunc) *LLMContextSummarizer
func (s *LLMContextSummarizer) ShouldAutoSummarize(snapshot llm.Snapshot) bool
func (s *LLMContextSummarizer) Summarize(ctx context.Context, c *llm.Context) (bool, error)
func (s *LLMContextSummarizer) Start(parent context.Context, snapshot llm.Snapshot, deliver func(SummaryResult)) bool
func (s *LLMContextSummarizer) Apply(c *llm.Context, res SummaryResult) bool
func (s *LLMContextSummarizer) SetTimeout(timeout time.Duration)
func (s *LLMContextSummarizer) InFlight() bool
func (s *LLMContextSummarizer) Cancel()
```

`LLMAutoContextSummarizationConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `MaxContextTokens` | `int` | 0 (no token limit) | Summarize when the estimated tokens of summarizable messages exceed this. |
| `MaxUnsummarizedMessages` | `int` | 0 (no count limit) | Summarize when the number of summarizable messages exceeds this. |
| `SummaryConfig` | `LLMContextSummaryConfig` | zero | What to keep and how to word the summary. |
| `EstimateTokens` | `func(llm.Message) int` | built-in estimate | Token estimate per message. |
| `ShouldSummarize` | `func(llm.Snapshot) bool` | none | Extra condition. With neither limit set, this alone decides; with no limits and no function, auto-summarization never fires. |

`LLMContextSummaryConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `TargetContextTokens` | `int` | 80% of `MaxContextTokens` | With a token limit, keep the most recent messages up to this many tokens. |
| `MinMessagesAfterSummary` | `int` | 0 | Always keep at least this many recent messages. |
| `SummarizationPrompt` | `string` | a built-in prompt asking to keep facts, decisions, commitments, preferences and open questions | System prompt sent to `Summarize`. |
| `SummaryMessageTemplate` | `string` | `"Conversation summary: " + summary` | Wording of the summary message. `%s` is replaced with the summary; without `%s` the summary is appended on a new line. |
| `Keep` | `func(llm.Message) bool` | none | Messages for which it returns true are never summarized. Tool calls and their results stay paired. |

Rules the summarizer always follows: leading system and developer messages are kept; it never cuts inside an unresolved tool call; the summary is inserted as a user message; if the context changed while the summary was being written, the summary is discarded. Only one summary runs at a time.

This configuration keeps every message that mentions an order number out of the summary, and summarizes once more than 30 messages can be summarized. Pass it as `AutoSummarizationConfig`, together with `EnableAutoContextSummarization` and `Summarize`, as in the quick example:

```go
func summaryConfig() aggregators.LLMAutoContextSummarizationConfig {
	return aggregators.LLMAutoContextSummarizationConfig{
		MaxUnsummarizedMessages: 30,
		SummaryConfig: aggregators.LLMContextSummaryConfig{
			MinMessagesAfterSummary: 6,
			Keep: func(m llm.Message) bool {
				return strings.Contains(m.Text(), "order #")
			},
		},
	}
}
```

To summarize right away, regardless of the limits, queue `frames.NewLLMSummarizeContextFrame()` on the task.

## Idle detection

```text
func NewUserIdleController(timeout time.Duration, onIdle func()) *UserIdleController
```

The user aggregator builds one from `UserIdleTimeout` and `OnUserIdle`; you rarely build it yourself. It starts its timer when the bot stops speaking (or when a user turn ended with no text) and cancels it on user speech, bot speech or function calls in progress. `UserIdleTimeoutUpdateFrame` changes the timeout at run time; 0 or negative turns it off. Methods: `ProcessFrame`, `UserTurnStarted`, `UserTurnStopped`, `WaitForUser`, `WaitingForUser`, `Timeout`, `Stop`.

A separate processor with the same purpose, `processors.UserIdleController`, is described in the [processors reference](processors.md#in-package-processors). The [turn-taking guide](../guides/turn-taking.md) shows how to re-prompt or hang up on idle.

This changes the idle timeout during a call, for example while the caller looks for a card. A timeout of 0 turns idle detection off:

```go
func waitLonger(task *pipeline.PipelineTask) error {
	return task.QueueFrame(frames.NewUserIdleTimeoutUpdateFrame(60 * time.Second))
}
```

## DTMF aggregator

```text
func NewDTMFAggregator(opts ...DTMFOption) *DTMFAggregator
```

Collects keypad digits from `InputDTMFFrame`s and pushes them downstream as one final `TranscriptionFrame`, so the LLM reads them like speech. Place it before the user aggregator.

| Option | Default | Effect |
| --- | --- | --- |
| `WithDTMFTimeout(d)` | 2 s (`DefaultDTMFTimeout`) | Flush after this long without a new digit. 0 or negative waits for the terminator. |
| `WithDTMFTerminationDigit(k)` | `frames.KeypadPound` (`#`) | Flush immediately when this key is pressed (it is included in the text). |
| `WithDTMFPrefix(prefix)` | `"DTMF: "` (`DefaultDTMFPrefix`) | Text placed before the digits. |

The first digit of a sequence broadcasts an `InterruptionFrame` when the task allows interruptions (`StartFrame.AllowInterruptions`). `InputDTMFFrame`s are forwarded too. `EndFrame` and `CancelFrame` flush pending digits.

This collects digits until the caller presses `*` or stops for three seconds:

```go
func newKeypad() *aggregators.DTMFAggregator {
	return aggregators.NewDTMFAggregator(
		aggregators.WithDTMFTimeout(3*time.Second),
		aggregators.WithDTMFTerminationDigit(frames.KeypadStar),
	)
}
```

Put it between the session input and the user aggregator, for example `s.Input(), newKeypad(), stt, pair.User(), ...`.

## Sentence aggregation

```text
func NewSentenceAggregator(mode ...TextAggregationMode) *SentenceAggregator
```

Buffers downstream `LLMTextFrame` and `TextFrame` text and emits one `TextFrame` per complete sentence. `TextAggregationModeSentence` (the default) buffers; `TextAggregationModeToken` passes text straight through. Frames with `SkipTTS` and all upstream frames pass unchanged. The buffer is flushed on `LLMFullResponseEndFrame` and `EndFrame`, and cleared on `InterruptionFrame`. The built-in TTS base already aggregates sentences, so you need this only for your own text consumers.

Text aggregator helpers for your own processors: `NewSimpleTextAggregator()` (sentences) and `NewWordTextAggregator()` (words) implement `TextAggregator` (`Aggregate`, `Flush`, `HandleInterruption`, `Reset`, `Type`). They work with `Aggregation{Text, Type, Spoken, Metadata}` values; `CollectAggregations` and `MergeAggregationChannels` help with their channel results.

This gives a processor of your own one `TextFrame` per sentence of the LLM's reply. Place the aggregator right before it:

```go
func sentenceStage(consumer processors.FrameProcessor) []processors.FrameProcessor {
	return []processors.FrameProcessor{aggregators.NewSentenceAggregator(), consumer}
}
```

## User-turn completion

```text
func NewUserTurnCompletionProcessor(ctx *llm.Context, cfg UserTurnCompletionConfig) *UserTurnCompletionProcessor
func DefaultUserTurnCompletionConfig() UserTurnCompletionConfig
func WrapSystemPromptWithTurnCompletion(systemPrompt string, cfg UserTurnCompletionConfig) string
```

This lets the LLM itself judge whether the user has finished. The system prompt tells the model to start every reply with a marker: `TurnCompleteMarker` (a check mark, U+2713) for complete, `TurnIncompleteShortMarker` (a white circle, U+25CB) for "will continue soon", `TurnIncompleteLongMarker` (a half-filled circle, U+25D0) for "needs more time". Place the processor right after the LLM.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Instructions` | `string` | `UserTurnCompletionInstructions` | Text that `WrapSystemPromptWithTurnCompletion` appends to your prompt. |
| `IncompleteShortTimeout` | `time.Duration` | 5 s | After the short-incomplete marker, wait this long, then re-prompt. |
| `IncompleteLongTimeout` | `time.Duration` | 10 s | After the long-incomplete marker, wait this long, then re-prompt. |
| `IncompleteShortPrompt` | `string` | built-in "encourage them to continue" prompt | System message appended (upstream, with `RunLLM`) on short timeout. |
| `IncompleteLongPrompt` | `string` | built-in "friendly check-in" prompt | System message appended on long timeout. |

Behaviour: the marker and any incomplete reply are pushed as `LLMTextFrame`s with `SkipTTS` set, so they reach the context but are not spoken; text after the complete marker is spoken. An interruption cancels the timer. If a response has no marker, the buffered text is pushed anyway with a warning.

The prompt and the processor must use the same config. Put the processor right after the LLM service:

```go
func turnCompletion(conversation *llm.Context, prompt string) (string, *aggregators.UserTurnCompletionProcessor) {
	cfg := aggregators.DefaultUserTurnCompletionConfig()
	cfg.IncompleteShortTimeout = 3 * time.Second
	return aggregators.WrapSystemPromptWithTurnCompletion(prompt, cfg), aggregators.NewUserTurnCompletionProcessor(conversation, cfg)
}
```

Use the returned prompt as the LLM's system prompt.

## Next steps

- [Turns reference](turns.md): the strategies the user aggregator runs.
- [Conversation context guide](../guides/conversation-context.md): messages, prompts and summarization in practice.
- [Function calling guide](../guides/function-calling.md): how results flow back through the assistant aggregator.
- [LLM reference](llm.md): `llm.Context` and messages.
