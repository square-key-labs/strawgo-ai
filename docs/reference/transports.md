# Transports reference

This page documents the `transports` package (the WebSocket server, client and session used for telephony and custom clients) and the `transports/webrtc` package (browser calls). It lists every config field with its default.

A **transport** moves audio and messages between the caller and the pipeline. A **session** is one connected call. A **serializer** translates between a carrier's wire format and frames; see [serializers](serializers.md). A **frame** is one typed message in the pipeline.

Import paths: `github.com/square-key-labs/strawgo-ai/src/transports`, `.../src/transports/webrtc`.

> [!NOTE]
> There is no Daily or LiveKit transport and no video. WebRTC negotiates G.711 (PCMU/PCMA) by default; Opus is only available when you build with the `opus` tag, and that path is unverified.

## Quick example

This function builds a Twilio-style WebSocket server. Each call gets its own session, and the handler builds a pipeline with the session's input first and output near the end.

```go
func newServer(build func(s *transports.Session) []processors.FrameProcessor) (*transports.Server, error) {
	return transports.NewServer(transports.ServerConfig{
		Addr: ":8080",
		Path: "/media",
		NewSerializer: func(*http.Request) serializers.FrameSerializer {
			return serializers.NewTwilioFrameSerializer(serializers.TwilioConfig{
				AccountSID: os.Getenv("TWILIO_ACCOUNT_SID"),
				AuthToken:  os.Getenv("TWILIO_AUTH_TOKEN"),
			})
		},
		Params: transports.TransportParams{SessionTimeout: 10 * time.Minute},
		Handler: func(ctx context.Context, s *transports.Session) error {
			task := pipeline.NewPipelineTask(pipeline.NewPipeline(build(s)))
			return task.Run(ctx)
		},
	})
}
```

A typical `build` returns `s.Input()`, STT, the user aggregator, the LLM, TTS, `s.Output()`, and the assistant aggregator, in that order. See [your first phone bot](../getting-started/first-phone-bot.md) for a full program.

## ServerConfig

```text
func NewServer(cfg ServerConfig) (*Server, error)
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Addr` | `string` | none | Listen address for `ListenAndServe`, for example `":8080"`. |
| `Path` | `string` | `"/ws"` | URL path the server answers on when it runs its own HTTP server. |
| `NewSerializer` | `func(r *http.Request) serializers.FrameSerializer` | none (required) | Builds a fresh serializer per connection. Returning nil rejects the connection. |
| `Handler` | `SessionHandler` | none (required) | `func(ctx context.Context, s *Session) error`. Runs once per call; the call ends when it returns. |
| `Params` | `TransportParams` | see below | Shared settings for every session. |
| `CheckOrigin` | `func(r *http.Request) bool` | allow every origin | WebSocket origin check. |

`NewServer` returns an error when `NewSerializer` or `Handler` is nil.

## Server methods

| Method | Signature | Description |
| --- | --- | --- |
| `ListenAndServe` | `func (s *Server) ListenAndServe(ctx context.Context) error` | Listen on `Addr` and serve until `ctx` is done. |
| `Serve` | `func (s *Server) Serve(ctx context.Context, ln net.Listener) error` | Serve on your own listener. |
| `ServeHTTP` | `func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request)` | Mount the server in your own `http.ServeMux`. `Path` is not checked here; your mux decides. |
| `ActiveSessions` | `func (s *Server) ActiveSessions() int` | Calls in progress. |
| `ShutdownContext` | `func (s *Server) ShutdownContext(ctx context.Context) error` | Graceful drain: new connections get HTTP 503, running calls finish on their own. When `ctx` expires, the remaining calls are closed and `ctx.Err()` is returned. |
| `Shutdown` | `func (s *Server) Shutdown()` | Cancel every handler's `ctx` and close every session now. |

What happens per connection:

1. The request is upgraded to a WebSocket. `NewSerializer(r)` builds the serializer.
2. If the serializer implements `serializers.Handshaker`, the server reads messages until the handshake completes (for example Twilio's `start` event), within `HandshakeTimeout`. The result is available as `s.CallInfo()`.
3. `Handler(ctx, s)` runs. `ctx` is the server's base context, cancelled by `Shutdown`. A panic in the handler is recovered and logged.
4. When the handler returns, the session is closed and the server waits for its goroutines.

When the `ctx` given to `ListenAndServe` or `Serve` is done, the server stops its HTTP listener (allowing `CloseTimeout`), calls `Shutdown`, and waits up to `HangupTimeout + CloseTimeout` for handlers to return. For a drain that lets calls finish, call `ShutdownContext` first. See the [deployment guide](../guides/deployment.md).

This mounts the server in your own mux next to a health check, and drains calls for up to five minutes when `ctx` is cancelled:

```go
func serveWithDrain(ctx context.Context, srv *transports.Server) error {
	mux := http.NewServeMux()
	mux.Handle("/media", srv)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "ok, %d calls\n", srv.ActiveSessions())
	})
	httpSrv := &http.Server{Addr: ":8080", Handler: mux}
	go func() {
		<-ctx.Done()
		drain, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		_ = srv.ShutdownContext(drain)
		_ = httpSrv.Shutdown(drain)
	}()
	if err := httpSrv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
```

## TransportParams

These apply to `ServerConfig.Params` and `ClientConfig.Params`. Zero durations get the defaults shown.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AudioInSampleRate` | `int` | `StartFrame.AudioInSampleRate` (16000 with the default task config) | 16-bit PCM caller audio is resampled to this rate before it enters the pipeline. |
| `AudioOutSampleRate` | `int` | the serializer's wire rate, else `StartFrame.AudioOutSampleRate` | Rate of audio sent to the caller, used only when the serializer does not declare a wire rate. |
| `AudioOutChunk` | `time.Duration` | 20 ms (`DefaultAudioOutChunk`); values under 1 ms use the default | Size of each outgoing audio chunk. Audio is paced in real time, one chunk per period. |
| `DisableAudioInPassthrough` | `bool` | false | Do not push caller `AudioFrame`s downstream. The analyzer and filter still see them. |
| `AudioInAnalyzer` | `AudioInputAnalyzer` | none | Analyzes each caller audio frame (as 16-bit PCM) and may emit frames downstream. One value is shared by every session; prefer `NewAudioInAnalyzer`. |
| `NewAudioInAnalyzer` | `func() AudioInputAnalyzer` | none | Builds an analyzer per session. Overrides `AudioInAnalyzer`. |
| `NewOutputMixer` | `func() mixer.Mixer` | none | Builds an output mixer per session. Overrides `OutputMixer`. |
| `NewInputFilter` | `func() filter.Filter` | none | Builds an input filter per session. Overrides `InputFilter`. |
| `AudioOutMixer` | `AudioMixer` (`mixer.Mixer`) | none | Mixer used when `OutputMixer` is nil. |
| `OutputMixer` | `mixer.Mixer` | none | Mixes background audio into everything sent to the caller, including silence. Controlled by `MixerEnableFrame` and `MixerUpdateSettingsFrame`. |
| `InputFilter` | `filter.Filter` | none | Filters caller audio (for example a noise gate) before analysis and before it enters the pipeline. Controlled by `FilterEnableFrame` and `FilterUpdateSettingsFrame`. |
| `AudioInNative` | `bool` | false | Ask the serializer to deliver caller audio in its wire codec (for example mu-law) without decoding. Needs a `NativeAudioSerializer`. |
| `AudioOutNative` | `bool` | false | Send TTS audio that is already in the wire codec and rate straight to the wire, without transcoding. Ignored when a mixer is set or audio is multi-channel. |
| `AudioOutEndSilence` | `time.Duration` | 0 | On `EndFrame`, send this much silence after the last audio (when no playback acknowledgement is used). |
| `InterruptionFade` | `time.Duration` | 0 (cut) | On interruption, fade the audio out over this long instead of cutting it. |
| `BotStopTimeout` | `time.Duration` | 3 s (`DefaultBotStopTimeout`) | If no new audio arrives this long after the last chunk was due, the bot is treated as having stopped speaking. |
| `PlaybackAck` | `bool` | false | After the bot's audio is sent, ask the carrier to confirm playback (for example a Twilio `mark`) and emit `BotStoppedSpeakingFrame` only when it is confirmed. Needs a `PlaybackAckSerializer`. |
| `PlaybackAckTimeout` | `time.Duration` | 1 s (`DefaultPlaybackAckTimeout`) | How long to wait for that confirmation. |
| `SessionTimeout` | `time.Duration` | 0 (off) | Maximum call length, counted from the `StartFrame`. When it expires: `OnSessionTimeout` if set, otherwise an `EndTaskFrame("session timeout")` ends the call gracefully. |
| `WriteTimeout` | `time.Duration` | 10 s (`DefaultWriteTimeout`) | Deadline for each WebSocket write. |
| `PingInterval` | `time.Duration` | 20 s (`DefaultPingInterval`); negative disables | WebSocket ping interval. A connection with no reads for 2 × ping + write timeout is treated as gone. |
| `HandshakeTimeout` | `time.Duration` | 10 s (`DefaultHandshakeTimeout`) | Limit for reading HTTP headers and for the serializer handshake. |
| `CloseTimeout` | `time.Duration` | 2 s (`DefaultCloseTimeout`) | Limit for the WebSocket close and for stopping the HTTP server. |
| `HangupTimeout` | `time.Duration` | 10 s (`DefaultHangupTimeout`) | Limit for the carrier's REST hang-up call and for each REST DTMF send. |
| `OnClientConnected` | `func(*Session)` | none | Called once the pipeline's `StartFrame` reaches the input, right after `ClientConnectedFrame` is pushed. |
| `OnClientDisconnected` | `func(*Session)` | none | Called when the caller hangs up or the socket drops. If set, it replaces the default action, which is to push `CancelTaskFrame("client disconnected")` upstream. |
| `OnSessionTimeout` | `func(*Session)` | none | Called when `SessionTimeout` expires. If set, it replaces the default `EndTaskFrame`. |

`AudioInputAnalyzer` is `interface{ AnalyzeAudio(frame *frames.AudioFrame, emit func(frames.Frame)) }`. `NewVADAnalyzerHook(analyzer vad.VADAnalyzer) *VADAnalyzerHook` adapts a VAD analyzer to it: it emits `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`. For full turn handling use `vad.VADInputProcessor` in the pipeline instead; see [turns](turns.md#vad).

Typical phone-call settings: a maximum call length, a short fade on interruption, playback confirmation from the carrier, and a log line per call:

```go
func phoneParams() transports.TransportParams {
	return transports.TransportParams{
		SessionTimeout:   15 * time.Minute,
		InterruptionFade: 30 * time.Millisecond,
		PlaybackAck:      true,
		OnClientConnected: func(s *transports.Session) {
			log.Printf("%s connected: call %s", s.ID(), s.CallInfo().CallID)
		},
	}
}
```

## Session

| Method | Signature | Description |
| --- | --- | --- |
| `Input` | `func (s *Session) Input() *InputProcessor` | Put first in the pipeline. Pushes caller audio and messages. |
| `Output` | `func (s *Session) Output() *OutputProcessor` | Put after TTS and before the assistant aggregator. Sends audio and messages to the caller. |
| `ID` | `func (s *Session) ID() string` | `"ws-N"`, unique in the process. |
| `CallInfo` | `func (s *Session) CallInfo() serializers.CallInfo` | Call details from the handshake (empty for `Dial`). |
| `Request` | `func (s *Session) Request() *http.Request` | The upgrade request (nil for `Dial`). Use it to read query parameters. |
| `Serializer` | `func (s *Session) Serializer() serializers.FrameSerializer` | The session's serializer. |
| `Done` | `func (s *Session) Done() <-chan struct{}` | Closed when the session closes. |
| `Close` | `func (s *Session) Close() error` | Close the socket now. Safe to call more than once. |
| `Wait` | `func (s *Session) Wait()` | Wait for the session's read and keepalive goroutines to exit. |

What the session does with frames:

| Frame | Where | Action |
| --- | --- | --- |
| `StartFrame` | input | Push `ClientConnectedFrame`, call `OnClientConnected`, start `SessionTimeout`. Caller messages wait until this point. |
| Caller audio | input | Resample, filter, analyze, then push `AudioFrame` downstream. |
| Caller `InterruptionFrame`, `InputTransportMessageFrame` | input | Broadcast in both directions. |
| Carrier "stop" or hang-up | input | Push `ClientDisconnectedFrame`, then `OnClientDisconnected` or `CancelTaskFrame`. |
| `TTSAudioFrame` | output | Convert to the wire format and send paced; emits `BotStartedSpeakingFrame`, `BotSpeakingFrame` (every 200 ms) and `BotStoppedSpeakingFrame`. |
| `InterruptionFrame` | output | Clear queued audio, send the carrier's clear message, end bot speech. |
| `OutputTransportMessageFrame`, `OutputDTMFFrame` | output | Sent in order after queued audio. DTMF goes out as a serializer message (`DTMFSerializer`), else through the carrier's REST API (`DTMFSender`, Telnyx and Plivo with credentials), else as in-band tones in the audio. |
| `OutputTransportMessageUrgentFrame`, `OutputDTMFUrgentFrame` | output | Sent at once. |
| `MixerEnableFrame`, `MixerUpdateSettingsFrame` | output | Passed to the output mixer. |
| `FilterEnableFrame`, `FilterUpdateSettingsFrame` | input | Passed to the input filter. |
| `EndFrame` | output | Wait for queued audio to play, send the serializer's end message, call `Hangup` if the serializer is a `Hangupper` (within `HangupTimeout`), close the socket. |
| `CancelFrame` | output | Stop sending at once, then the same end, hang-up and close as `EndFrame`. |

This reads a query parameter from the upgrade request, for example `wss://YOUR_HOST/media?lang=de`. `Request()` is nil for sessions from `Dial`:

```go
func languageOf(s *transports.Session) string {
	if r := s.Request(); r != nil {
		if lang := r.URL.Query().Get("lang"); lang != "" {
			return lang
		}
	}
	return "en"
}
```

## Client: ClientConfig and Dial

```text
func Dial(ctx context.Context, cfg ClientConfig) (*Session, error)
```

`Dial` connects out to a WebSocket server and returns a started `Session`. Build your pipeline around `Input()` and `Output()` exactly as for the server.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `URL` | `string` | none | `ws://` or `wss://` URL. |
| `Header` | `http.Header` | none | Headers for the opening request, for example auth. |
| `Serializer` | `serializers.FrameSerializer` | none (required) | Wire format. |
| `Params` | `TransportParams` | as above | Session settings. |
| `Dialer` | `*websocket.Dialer` | `websocket.DefaultDialer` | Gorilla WebSocket dialer. |

`Dial` performs no serializer handshake, so `CallInfo()` is empty. See the [WebSocket client guide](../guides/websocket-client.md).

This dials a media server that speaks the protobuf wire format, with a bearer token:

```go
func dialMediaServer(ctx context.Context, url, token string) (*transports.Session, error) {
	return transports.Dial(ctx, transports.ClientConfig{
		URL:        url,
		Header:     http.Header{"Authorization": []string{"Bearer " + token}},
		Serializer: serializers.NewProtobufFrameSerializer(),
	})
}
```

## WebRTC

```text
func NewHandler(cfg Config) (*Handler, error)
```

The handler is an `http.Handler` that implements a small signalling API: POST an SDP offer, get an answer; PATCH extra ICE candidates. Each new peer connection runs your `SessionHandler` once.

### webrtc.Config

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `ICEServers` | `[]webrtc.ICEServer` (Pion) | none | STUN and TURN servers. Change later with `UpdateICEServers`. |
| `Handler` | `SessionHandler` | none (required) | `func(ctx context.Context, s *Session) error`, run once per peer. |
| `Params` | `Params` | see below | Session settings. |
| `Codecs` | `[]Codec` | `DefaultCodecs()`: PCMU, PCMA (Opus first with the `opus` build tag) | Preference order; the first one the offer contains is used. |
| `SettingEngine` | `*webrtc.SettingEngine` (Pion) | none | Advanced Pion settings, for example a fixed UDP port range. |
| `ConnectionMode` | `ConnectionMode` | `MultipleConnections` | `SingleConnection` allows one peer at a time; a second new offer is rejected. |
| `GatherTimeout` | `time.Duration` | 10 s (`DefaultGatherTimeout`) | How long to gather ICE candidates before answering. |

### webrtc.Params

> [!WARNING]
> Unlike `TransportParams`, `webrtc.Params` has no per-session factories (`NewAudioInAnalyzer`, `NewOutputMixer`, `NewInputFilter`). An `AudioInAnalyzer` or `AudioOutMixer` set here is one instance shared by every peer of the handler, and both keep per-call state. With more than one concurrent peer, create stateful processors such as `vad.NewVADInputProcessor` inside your session handler instead, and leave these fields empty.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `AudioInSampleRate` | `int` | `StartFrame.AudioInSampleRate` | Caller audio is resampled to this rate. |
| `DisableAudioInPassthrough` | `bool` | false | Do not push caller audio downstream. |
| `AudioInAnalyzer` | `transports.AudioInputAnalyzer` | none | Analyzes caller audio; may emit frames. |
| `AudioOutMixer` | `transports.AudioMixer` | none | Mixes into outgoing audio. |
| `DisableAudioOutSilence` | `bool` | false | By default silence is sent while the bot is quiet; set true to send nothing. |
| `InterruptionFade` | `time.Duration` | 0 (cut) | Fade-out length on interruption. |
| `BotStopTimeout` | `time.Duration` | 3 s | As for `TransportParams`. |
| `ConnectTimeout` | `time.Duration` | 60 s (`DefaultConnectTimeout`) | Time allowed for the peer connection to connect. |
| `DisconnectTimeout` | `time.Duration` | 5 s (`DefaultDisconnectTimeout`) | Grace period after the connection reports "disconnected" before the session treats the peer as gone. |
| `DataChannelTimeout` | `time.Duration` | 10 s (`DefaultDataChannelTimeout`) | Outgoing messages queue until the data channel opens; they are discarded if it does not open in time. |
| `KeepAliveTimeout` | `time.Duration` | 0 (off) | Treat the peer as gone when no data-channel message arrives for this long. Messages starting with `ping` count. |
| `SessionTimeout` | `time.Duration` | 0 (off) | Maximum call length from connect. Default action is `EndTaskFrame("session timeout")`. |
| `CloseTimeout` | `time.Duration` | 250 ms (`DefaultCloseTimeout`) | Time allowed to close the peer connection. |
| `OnClientConnected`, `OnClientDisconnected`, `OnSessionTimeout` | `func(*Session)` | none | Same meaning as in `TransportParams`. |

### Handler methods

| Method | Description |
| --- | --- |
| `ServeHTTP` | POST: decode an `OfferRequest`, reply with an `Answer` as JSON. PATCH: decode a `PatchRequest`, reply `{"status":"success"}`. Other methods get 405. Bodies are limited to 1 MiB. |
| `HandleOffer(ctx, req OfferRequest, r *http.Request) (Answer, error)` | The POST logic, for your own routing. An offer with an existing `pc_id` renegotiates that session. |
| `HandleCandidates(ctx, req PatchRequest) error` | The PATCH logic. |
| `Session(pcID string) *Session` | Look up a running session. |
| `UpdateICEServers(servers)` | Replace the ICE servers for new sessions. |
| `Shutdown(ctx) error` | Reject new offers, close every session, and wait for handlers until `ctx` is done. |

Error mapping in `ServeHTTP`: `ErrBadRequest` and `ErrNoSupportedCodec` give 400, `ErrNotFound` and `ErrSessionClosed` give 404, `ErrShuttingDown` gives 503, anything else 500.

Wire types:

| Type | JSON fields |
| --- | --- |
| `OfferRequest` | `sdp`, `type` (must be `"offer"`), `pc_id` (optional), `restart_pc`, `request_data` or `requestData` (any JSON, available as `Session.RequestData()`) |
| `Answer` | `sdp`, `type`, `pc_id` |
| `PatchRequest` | `pc_id`, `candidates` |
| `ICECandidate` | `candidate`, `sdp_mid`, `sdp_mline_index` |

### webrtc.Session

| Method | Description |
| --- | --- |
| `Input()`, `Output()` | Pipeline processors, used like the WebSocket session's. |
| `ID()` | The `pc_id`. |
| `Codec()` | The negotiated `Codec`. |
| `Request()`, `RequestData()` | The offer's HTTP request and its `request_data` JSON. |
| `SendMessage(ctx, msg any) error` | Send a message on the data channel. |
| `RequestRenegotiation(ctx) error` | Ask the browser to send a new offer. |
| `Disconnected()` | Closed when the peer is gone. |
| `Done()` | Closed when the session closes. |
| `Close() error`, `Wait()` | Close the session; wait for it to finish. |

`Codec` is an interface (`Capability`, `PayloadType`, `SampleRate`, `NewEncoder`, `NewDecoder`); `PCMU` and `PCMA` are the built-in values. See the [WebRTC guide](../guides/webrtc.md).

```go
func mountWebRTC(mux *http.ServeMux, run webrtc.SessionHandler) (*webrtc.Handler, error) {
	h, err := webrtc.NewHandler(webrtc.Config{
		Handler: run,
		Params:  webrtc.Params{SessionTimeout: 15 * time.Minute},
	})
	if err != nil {
		return nil, err
	}
	mux.Handle("/api/offer", h)
	return h, nil
}
```

## Next steps

- [Transports overview](../guides/transports-overview.md): how transports, sessions and serializers fit together.
- [Serializers reference](serializers.md): every carrier's wire format.
- [Deployment guide](../guides/deployment.md): graceful shutdown in production.
- [WebRTC guide](../guides/webrtc.md): a browser call end to end.
