# Turns reference

This page documents how Strawgo decides who is talking: the user-turn start, stop and mute strategies in `src/turns`, the VAD analyzer and processor in `audio/vad`, and the smart-turn analyzers in `audio/turn`.

A **turn** is one stretch of the user or the bot talking. **VAD** (voice activity detection) decides from audio alone whether someone is speaking. **Smart turn** is a model that listens to the end of an utterance and predicts whether the speaker has finished. An **interruption** (barge-in) is the user talking over the bot. **Muting** the user means ignoring their speech for a while.

Import paths: `github.com/square-key-labs/strawgo-ai/src/turns` and its `user_start`, `user_stop`, `user_mute` subpackages; `.../src/audio/vad`; `.../src/audio/turn`.

For when to use which strategy, read the [turn-taking guide](../guides/turn-taking.md). This page is the lookup.

## Quick example

This function returns a strategy set that starts a turn on VAD or transcription, needs three words to interrupt the bot, ends the turn 800 ms after the user stops, and mutes the user until the bot has finished its greeting.

```go
func phoneStrategies() turns.UserTurnStrategies {
	return turns.UserTurnStrategies{
		StartStrategies: []user_start.UserTurnStartStrategy{
			user_start.NewVADUserTurnStartStrategy(false),
			user_start.NewMinWordsUserTurnStartStrategy(3, true),
		},
		StopStrategies: []user_stop.UserTurnStopStrategy{
			user_stop.NewSpeechTimeoutUserTurnStopStrategy(800 * time.Millisecond),
		},
		MuteStrategies: []user_mute.UserMuteStrategy{
			user_mute.NewMuteUntilFirstBotCompleteUserMuteStrategy(),
		},
	}
}
```

Pass it as `UserAggregatorParams.Strategies` or `PipelineTaskConfig.TurnStrategies`. See [aggregators](aggregators.md#user-aggregator) for which one wins.

## UserTurnStrategies

```text
type UserTurnStrategies struct {
	StartStrategies []user_start.UserTurnStartStrategy
	StopStrategies  []user_stop.UserTurnStopStrategy
	MuteStrategies  []user_mute.UserMuteStrategy
}
func (s UserTurnStrategies) Empty() bool
func DefaultUserTurnStrategies() UserTurnStrategies
```

`DefaultUserTurnStrategies()` returns:

| Kind | Strategies |
| --- | --- |
| Start | `NewVADUserTurnStartStrategy(true)`, `NewTranscriptionUserTurnStartStrategy(true)` |
| Stop | `NewSpeechTimeoutUserTurnStopStrategy(600 * time.Millisecond)` |
| Mute | none |

How the user aggregator runs them on every frame it sees:

1. Each start strategy is asked `ShouldStart(frame)`. The first one that says yes, when no turn is active, starts a turn. Its `EnableInterruptions()` decides whether the start may interrupt the bot.
2. Each stop strategy is asked `ShouldStop(frame)`. If any says yes and the user is still speaking, the turn ends as soon as they stop; otherwise it ends now.
3. Each mute strategy is asked `ShouldMute(frame)`. If any says yes, the user is muted.
4. Strategies are `Reset()` when a turn starts (start strategies) or ends (stop strategies).

Strategies identify frames by type name, so they work on `UserStartedSpeakingFrame` and `VADUserStartedSpeakingFrame` alike.

Start from the defaults and change one part, here adding a mute strategy:

```go
func defaultsWithFunctionMute() turns.UserTurnStrategies {
	s := turns.DefaultUserTurnStrategies()
	s.MuteStrategies = append(s.MuteStrategies, user_mute.NewFunctionCallUserMuteStrategy())
	return s
}
```

## EmptyUserTurnConfig

```text
const DefaultEmptyUserTurnInterruptedPrompt string
type EmptyUserTurnConfig struct {
	Disabled                   bool
	InterruptedPrompt          string
	LeaveInterruptedUnanswered bool
	IdlePrompt                 string
	MaxConsecutiveRecoveries   int
	ResumeInterruptedSpeech    bool
}
func DefaultEmptyUserTurnConfig() EmptyUserTurnConfig
func (c EmptyUserTurnConfig) Prompt(interruptedBot bool) string
func (c EmptyUserTurnConfig) MaxRecoveries() int
```

Decides what the user aggregator does with a turn that ends with no transcript, such as a cough that started a turn through VAD. Pass it as `UserAggregatorParams.EmptyUserTurn`; nil means `DefaultEmptyUserTurnConfig()`, which asks the caller to repeat when the turn interrupted the bot and leaves it unanswered otherwise. `ResumeInterruptedSpeech` makes the bot continue its interrupted reply instead. Recovery is off while the active LLM announces itself as realtime with an `LLMServiceMetadataFrame`. See [Empty user turns](aggregators.md#empty-user-turns).

## Start strategies

```text
type UserTurnStartStrategy interface {
	ShouldStart(frame any) bool
	EnableInterruptions() bool
	Reset()
}
```

| Constructor | Starts a turn on | Notes |
| --- | --- | --- |
| `NewVADUserTurnStartStrategy(enableInterruptions bool)` | `UserStartedSpeakingFrame` or `VADUserStartedSpeakingFrame` | Fastest; any noise VAD calls speech starts a turn. |
| `NewTranscriptionUserTurnStartStrategy(enableInterruptions bool)` | A `TranscriptionFrame` or `InterimTranscriptionFrame` with non-empty text | Works without VAD. |
| `NewMinWordsUserTurnStartStrategy(minWords int, enableInterruptions bool)` | Transcribed words: 1 while the bot is quiet, `minWords` while the bot is speaking | Stops short backchannels ("uh-huh") from interrupting. `minWords` below 1 becomes 1. Words are counted across final transcriptions of one utterance; interim text counts toward the total. When a final transcription falls short, the aggregator discards that text. |

`AggregationResetter` (`ConsumeAggregationReset() bool`) is the optional interface the min-words strategy uses to ask for that discard.

Without a VAD in the pipeline, start turns on transcriptions only:

```go
func transcriptionOnlyStart() []user_start.UserTurnStartStrategy {
	return []user_start.UserTurnStartStrategy{
		user_start.NewTranscriptionUserTurnStartStrategy(true),
	}
}
```

## Stop strategies

```text
type UserTurnStopStrategy interface {
	ShouldStop(frame any) bool
	Reset()
}
```

Optional interfaces: `TurnLifecycle` (`HandleUserTurnStarted`, `HandleUserTurnStopped`; called instead of `Reset` at turn boundaries), `Scheduler` (`NextDeadline() (time.Time, bool)`; the aggregator wakes up at that time to ask again), `TurnMetricsReporter` (`TakeTurnMetrics() (turn.TurnMetrics, bool)`; reported as `TurnMetricsData`), `Speculator` (speculative replies, see [EagerUserTurnStopStrategy](#eageruserturnstopstrategy)).

This stop strategy waits one second after the caller stops, and caps the wait for the STT's final transcription at 300 ms:

```go
func patientStop() user_stop.UserTurnStopStrategy {
	s := user_stop.NewSpeechTimeoutUserTurnStopStrategy(time.Second)
	s.SetTTFSP99Latency(300 * time.Millisecond)
	return s
}
```

### SpeechTimeoutUserTurnStopStrategy

```text
func NewSpeechTimeoutUserTurnStopStrategy(userSpeechTimeout time.Duration) *SpeechTimeoutUserTurnStopStrategy
func (s *SpeechTimeoutUserTurnStopStrategy) SetWaitForTranscript(wait bool)
func (s *SpeechTimeoutUserTurnStopStrategy) SetTTFSP99Latency(d time.Duration)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `userSpeechTimeout` | 600 ms (`DefaultUserSpeechTimeout`) in the defaults | Silence required after the user stops speaking. Negative becomes 0. |
| `SetWaitForTranscript(wait)` | true | When true, the turn ends only once some final transcription has text. |
| `SetTTFSP99Latency(d)` | from `STTMetadataFrame` | Maximum wait for the final transcript after speech stops. Setting it pins the value. 0 means do not wait. |

The turn ends when all of these hold: the user is not speaking, `userSpeechTimeout` has passed since VAD said they stopped, the STT has delivered a final transcription or its P99 latency has passed, and (by default) there is transcribed text. Without VAD frames, the timeout starts when a final transcription arrives. Speech that resumes cancels the pending stop.

When a `SpeechControlParamsFrame` with non-nil `TurnParams` arrives (the VAD processor sends one when a turn analyzer is attached), the turn also waits for a `UserStoppedSpeakingFrame`, which that VAD processor sends only once the analyzer predicts the turn is complete or its `StopSecs` of silence pass. A `VADUserStoppedSpeakingFrame` at a pause then starts the timers but does not end the turn on its own. See [Turn-taking: Smart Turn](../guides/turn-taking.md#smart-turn).

### TurnAnalyzerUserTurnStopStrategy

```text
func NewTurnAnalyzerUserTurnStopStrategy(analyzer turn.TurnAnalyzer) *TurnAnalyzerUserTurnStopStrategy
```

On each final `TranscriptionFrame`, calls `analyzer.AnalyzeEndOfTurn()` and stops the turn when it returns `TurnComplete`. Inference metrics are reported. `Reset` clears the analyzer. The analyzer must be receiving audio: in this repository only `VADInputProcessor` with `WithTurnAnalyzer` feeds it.

### EagerUserTurnStopStrategy

```text
func NewEagerUserTurnStopStrategy(policy EagerMatchPolicy, speculationTimeout time.Duration) *EagerUserTurnStopStrategy
func (s *EagerUserTurnStopStrategy) MatchPolicy() EagerMatchPolicy

type EagerMatchPolicy interface {
	Matches(eager, final string) bool
}

type Speculator interface {
	TakeSpeculation() (start *UserTurnSpeculation, cancel bool)
	ConfirmsSpeculation() bool
	SpeculationDeadline() (time.Time, bool)
}
```

Answers a provider's eager end of turn speculatively. Deepgram Flux recommends it when `FluxConfig.EnableEagerEndOfTurn` is set (see [Turn-taking: speculative replies](../guides/turn-taking.md#speculative-replies-eager-end-of-turn)).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `policy` | `NormalizedMatch{}` when nil | Decides whether the final transcript still matches the eager one. `NormalizedMatch` ignores case, punctuation, apostrophes and extra spaces. `ExactMatch` requires identical text. |
| `speculationTimeout` | 5 s (`DefaultSpeculationTimeout`) when 0 or less | A speculation that is neither confirmed nor withdrawn within this time is withdrawn. |

Behaviour:

- An `EagerTranscriptionFrame` with text starts a speculation. The user aggregator then sends a speculative `LLMContextFrame` built from a copy of the context plus the eager text. The LLM service holds the reply.
- A `ProposedUserStoppedSpeakingFrame` ends the turn. If the final `TranscriptionFrame`s received since the eager one match its text, the speculation is confirmed: the user aggregator writes the final transcript, does not run the LLM again, and broadcasts `UserTurnSpeculationConfirmedFrame`, which releases the held reply. Otherwise the speculation is withdrawn and the LLM runs on the final transcript.
- `UserTurnResponseCancelFrame`, an `InterruptionFrame`, a `UserStartedSpeakingFrame`, `VADUserStartedSpeakingFrame` or `ProposedUserStartedSpeakingFrame` during a speculation withdraw it. On the speaking frames the user aggregator broadcasts `UserTurnResponseCancelFrame`, which makes the LLM service discard the held reply and stop generating. The `InterruptionFrame` discards the held reply on its own.
- At turn stop, a matching speculation is still withdrawn if the context changed since it started (a tool result, appended messages), so the reply is regenerated from the current context.
- An eager transcription that reaches the strategy up to 100 ms before the turn-start frame is kept and reported when the turn starts. An older one is dropped, so a late eager transcription from the previous turn does not start a speculation in the next.
- The turn ending any other way (stop timeout, an empty transcript) withdraws an unresolved speculation.

`Speculator` is the interface the user aggregator looks for. `TakeSpeculation` reports a speculation to start or one to withdraw since the last call. `ConfirmsSpeculation` is read when the turn ends. `SpeculationDeadline` is when the aggregator should wake the strategy so it can time a speculation out.

## Mute strategies

```text
type UserMuteStrategy interface {
	ShouldMute(frame any) bool
	Reset()
}
```

| Constructor | Mutes the user |
| --- | --- |
| `NewAlwaysUserMuteStrategy()` | Always. |
| `NewFirstSpeechUserMuteStrategy()` | While the bot speaks its first utterance. |
| `NewMuteUntilFirstBotCompleteUserMuteStrategy()` | From the start of the call until the bot finishes its first utterance. An `ErrorFrame` before the bot has spoken unmutes. |
| `NewFunctionCallUserMuteStrategy()` | While any function call is running (from `FunctionCallsStartedFrame` until each call's result or cancel). |

While muted, the user aggregator drops `AudioFrame`, `TranscriptionFrame`, `InterimTranscriptionFrame`, `EagerTranscriptionFrame`, `UserBackchannelFrame`, the `ProposedUser...` frames and `InterruptionFrame`. It still forwards `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`, `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame` and uses them to track whether the caller is speaking (and for idle detection), but they do not start a turn. It broadcasts `UserMuteStartedFrame` and `UserMuteStoppedFrame` on changes.

This mutes the caller until the bot's greeting has finished, and again while any function call runs:

```go
func greetingAndToolMute() []user_mute.UserMuteStrategy {
	return []user_mute.UserMuteStrategy{
		user_mute.NewMuteUntilFirstBotCompleteUserMuteStrategy(),
		user_mute.NewFunctionCallUserMuteStrategy(),
	}
}
```

## VAD

This builds a VAD processor that needs a little more confidence than the default before it reports speech:

```go
func strictVAD(sock string) (*vad.VADInputProcessor, func(), error) {
	params := vad.DefaultVADParams()
	params.Confidence = 0.8
	analyzer, err := vad.NewSileroVADAnalyzer(16000, params, sock)
	if err != nil {
		return nil, nil, err
	}
	cleanup := func() { _ = analyzer.Cleanup() }
	return vad.NewVADInputProcessor(analyzer), cleanup, nil
}
```

### VAD parameters

```text
type VADParams struct {
	Confidence     float32
	StartSecs      float32
	StopSecs       float32
	MinVolume      float32
	HoldConfidence float32
}
func DefaultVADParams() VADParams
```

| Field | Default | Meaning |
| --- | --- | --- |
| `Confidence` | 0.7 | Voice confidence (0 to 1) needed to count a chunk as speech. |
| `StartSecs` | 0.2 | Seconds of speech before the state becomes speaking. |
| `StopSecs` | 0.2 | Seconds of non-speech before the state becomes quiet. |
| `MinVolume` | 0.1 | Smoothed volume below which confidence is treated as 0. |
| `HoldConfidence` | 0 (same as `Confidence`) | Lower threshold used while already speaking, so speech does not flicker off. Values above `Confidence` are ignored. |

`p.Merge(update)` copies each positive field of `update`. `ParamsFromFrame` and `ToFrameParams` convert to and from `frames.VADParams`. `VADParamsUpdateFrame` changes the parameters at run time.

`VADState` values: `VADStateQuiet`, `VADStateStarting`, `VADStateSpeaking`, `VADStateStopping`.

### Analyzers

```text
type VADAnalyzer interface {
	SetSampleRate(sampleRate int) error
	NumFramesRequired() int
	VoiceConfidence(buffer []byte) float32
	AnalyzeAudio(buffer []byte) (VADState, error)
	Restart()
}
func NewSileroVADAnalyzer(sampleRate int, params VADParams, sockPath string) (*SileroVADAnalyzer, error)
func NewBaseVADAnalyzer(sampleRate int, params VADParams) *BaseVADAnalyzer
```

`SileroVADAnalyzer` runs the Silero model in the `onnx-worker` sidecar over the Unix socket at `sockPath`. It supports 8000 and 16000 Hz and analyzes 256 or 512 samples per call. `Cleanup()` closes the connection. `BaseVADAnalyzer` holds the shared state machine; build your own analyzer on its `ProcessAudio(buffer, voiceConfidence, numFramesRequired)`. Install the sidecar as described in [installation](../getting-started/installation.md).

### VADInputProcessor

```text
func NewVADInputProcessor(analyzer VADAnalyzer, opts ...Option) *VADInputProcessor
func NewVADInputProcessorWithTurn(analyzer VADAnalyzer, turnAnalyzer turn.TurnAnalyzer, opts ...Option) *VADInputProcessor
func (p *VADInputProcessor) GetCurrentState() VADState
```

Place it right after the transport input. It forwards audio and emits speaking frames.

| Option | Default | Effect |
| --- | --- | --- |
| `WithSpeechActivityPeriod(d)` | 200 ms (`DefaultSpeechActivityPeriod`) | How often to broadcast `UserSpeakingFrame` during speech. |
| `WithAudioIdleTimeout(d)` | 1 s (`DefaultAudioIdleTimeout`) | If audio stops arriving while the user is speaking, force a stop after this long. 0 disables. |
| `WithTurnAnalyzer(a)` | none | Use smart turn to decide when the user has stopped. |

| Situation | Frames emitted |
| --- | --- |
| VAD speaking starts | `UserStartedSpeakingFrame` downstream, `VADUserStartedSpeakingFrame` both ways. |
| VAD speaking stops, no turn analyzer | `UserStoppedSpeakingFrame` downstream, `VADUserStoppedSpeakingFrame` both ways. |
| VAD speaking stops, with turn analyzer | `VADUserStoppedSpeakingFrame` at once; `UserStoppedSpeakingFrame` only when the analyzer predicts the turn is complete or `StopSecs` of silence pass. |
| `StartFrame`, parameter change | `SpeechControlParamsFrame` both ways; `TurnParams` is set when a turn analyzer is attached. |

If the analyzer rejects the input sample rate, the processor resamples to 16 kHz for it.

`transports.NewVADAnalyzerHook` is a lighter alternative that runs a VAD analyzer inside the transport and emits only the VAD frames; see [transports](transports.md#transportparams).

## Smart turn

```text
type TurnAnalyzer interface {
	AppendAudio(buffer []byte, isSpeech bool) EndOfTurnState
	AnalyzeEndOfTurn() (EndOfTurnState, *TurnMetrics, error)
	SpeechTriggered() bool
	SetSampleRate(sampleRate int)
	Clear()
}
```

`EndOfTurnState` is `TurnIncomplete` or `TurnComplete`. `TurnMetrics` has `IsComplete bool`, `Probability float64`, `InferenceTimeMs float64`, `TotalTimeMs float64`. The model works at 16 kHz (`ModelSampleRate`); audio is resampled for it.

### Parameters

`SmartTurnParams` (and the identical `TurnAnalyzerParams` for `NewBaseTurnAnalyzer`):

| Field | `DefaultSmartTurnParams()` | Meaning |
| --- | --- | --- |
| `StopSecs` | 3.0 | Silence after speech that ends the turn without asking the model. Also the cap on one inference: if the model has not answered within this, the turn ends. |
| `PreSpeechMs` | 500 (`DefaultTurnAnalyzerParams()`: 0) | Audio from before speech started that is sent to the model. |
| `MaxDurationSecs` | 8.0 | Longest audio window kept; 0 or negative becomes 8. |

A nil `Params` uses `DefaultSmartTurnParams()`.

### Analyzers

| Constructor | Config | Backend |
| --- | --- | --- |
| `NewOnnxSmartTurn(config OnnxSmartTurnConfig) (*OnnxSmartTurn, error)` | `SockPath string`, `Params *SmartTurnParams`, `Breaker BreakerConfig` | The local `onnx-worker` sidecar. A probability above 0.5 means complete. Each request times out after 5 s (`DefaultOnnxTurnTimeout`). `Close()` releases the connection. |
| `NewHTTPSmartTurn(config HTTPSmartTurnConfig) *HTTPSmartTurn` | see below | Any HTTP endpoint that serves the smart-turn model. |
| `NewFalSmartTurn(config FalSmartTurnConfig) *HTTPSmartTurn` | `URL` (default `DefaultFalSmartTurnURL`), `APIKey`, `Params`, `Breaker`, `HTTPClient` | fal.ai's hosted model, as an `HTTPSmartTurn`. |

`HTTPSmartTurnConfig`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `URL` | `string` | `https://fal.run/fal-ai/smart-turn/raw` | Endpoint. |
| `APIKey` | `string` | none | Sent as `Authorization: Key <APIKey>`. |
| `Headers` | `map[string]string` | none | Extra headers. |
| `Format` | `HTTPSmartTurnFormat` | `SmartTurnFormatNPY` | Request body: NumPy float32 array, or `SmartTurnFormatJSON`. |
| `Params` | `*SmartTurnParams` | `DefaultSmartTurnParams()` | Analyzer parameters. |
| `TimeoutSecs` | `int` | 0 (only the `StopSecs` cap) | Per-request timeout. |
| `Breaker` | `BreakerConfig` | see below | Failure handling. |
| `HTTPClient` | `*http.Client` | a new `http.Client` | Client. |

`BreakerConfig` (circuit breaker; when open, the analyzer skips the model and ends turns on silence):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Disabled` | `bool` | false | Turn the breaker off. |
| `FailureThreshold` | `int` | 3 | Consecutive failures that open it. |
| `Cooldown` | `time.Duration` | 5 s | First open period; doubles on each re-open. |
| `MaxCooldown` | `time.Duration` | 1 min | Cap on the open period. |

Analyzers also implement `ContextTurnAnalyzer` (`AnalyzeEndOfTurnContext(ctx)`) and `BreakerOpen() bool`. `turn.AnalyzeEndOfTurn(ctx, a)` uses the context variant when available. `OnnxTurnClient` (`NewOnnxTurnClient(sockPath)`) is the raw sidecar client.

```go
func newSmartVAD(sock string) (*vad.VADInputProcessor, func(), error) {
	analyzer, err := vad.NewSileroVADAnalyzer(16000, vad.DefaultVADParams(), sock)
	if err != nil {
		return nil, nil, err
	}
	smart, err := turn.NewOnnxSmartTurn(turn.OnnxSmartTurnConfig{SockPath: sock, Params: turn.DefaultSmartTurnParams()})
	if err != nil {
		analyzer.Cleanup()
		return nil, nil, err
	}
	cleanup := func() {
		smart.Close()
		analyzer.Cleanup()
	}
	return vad.NewVADInputProcessorWithTurn(analyzer, smart), cleanup, nil
}
```

## Next steps

- [Turn-taking guide](../guides/turn-taking.md): choosing and tuning strategies.
- [Aggregators reference](aggregators.md): the user aggregator that runs the strategies.
- [Frames reference](frames.md#user-and-bot-speaking): the speaking frames.
- [Deployment guide](../guides/deployment.md): running the `onnx-worker` sidecar.
