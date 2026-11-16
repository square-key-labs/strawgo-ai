# Call control

Call control is what a phone bot does to the call itself rather than to the conversation: hand the caller over to a person or another number, hang up, or press keypad keys. This guide shows how to transfer a live call from a running pipeline, which carriers support it, and how to give the LLM ready-made `transfer_call`, `end_call` and `send_dtmf` tools with the `extensions/callcontrol` package.

A **transfer** here is a cold (blind) transfer: the bot hands the call to the destination and leaves; it does not stay on the line or introduce the caller. A **destination** is an E.164 phone number such as `+14155550100` or a SIP URI such as `sip:agent@pbx.example.com`.

## Before you begin

- You have a working phone bot over a WebSocket carrier. See [Your first phone bot](../getting-started/first-phone-bot.md) and [Telephony providers](telephony-providers.md).
- You know how function calling works. See [Function calling](function-calling.md).
- Your serializer has REST credentials where the carrier needs them (the same ones used for hang-up).

> [!WARNING]
> None of the transfers below has been run against a live carrier. The Twilio, Telnyx and Plivo requests are tested against a local mock server; the Genesys one against the message it writes. The request shapes come from the carriers' public documentation, not from a live call. Place a test call on your own account before you rely on a transfer in production.

## Carrier support

| Carrier | Transfer | How | What you configure |
| --- | --- | --- | --- |
| Twilio | implemented | `POST /2010-04-01/Accounts/{AccountSID}/Calls/{CallSID}.json` with a `Twiml` parameter: `<Dial>` with a `<Number>` or `<Sip>`, an optional `<Say>` announcement, `callerId` and `action`. The TwiML must stay under 4000 characters after escaping, so a long announcement is refused | `AuthToken`, or `APIKeySID` and `APIKeySecret`; optionally `TransferActionURL` and `TransferSayLanguage` |
| Telnyx | implemented | `POST /v2/calls/{call_control_id}/actions/transfer` with `to` and an optional `from` (caller ID). Announcements are not sent | `APIKey` |
| Plivo | implemented, needs your own XML endpoint | `POST /v1/Account/{AuthID}/Call/{CallID}/` with `legs=aleg`, `aleg_url` and `aleg_method=GET`. Plivo then fetches `aleg_url`, which is your `TransferURL` plus `destination`, `caller_id` and `announcement` query parameters; your endpoint answers with the `<Dial>` XML | `AuthID`, `AuthToken`, `TransferURL` |
| Genesys AudioHook | implemented as a hand-off | AudioHook cannot dial. The destination goes into the `transfer_to` output variable of the `disconnect` message (action `transfer`) sent when the pipeline ends; your Architect flow routes on it | Nothing; read `transfer_to` in the flow |
| Exotel | not a transfer | The Voicebot applet has no transfer message. End the stream with `callcontrol.EndCall`; Exotel moves to the next applet in your call flow, so put a Connect applet with the agent's number there. This behaviour is from Exotel's documentation and has not been checked on a live call | Your Exotel call flow |
| Asterisk | not implemented | The `chan_websocket` protocol in this repository has no transfer command. Redirect the channel through ARI or the dialplan from your own code | |
| Jambonz | not implemented | The `listen` socket here only sends `disconnect` and `killAudio`. Transfer through the jambonz REST API or the next verb in your application | |
| Vonage | not implemented | The serializer has no REST client. Transfer the call through the Vonage Voice API from your own code | |

`TransferRequest.Announcement` is spoken by the carrier itself only on Twilio (in `TransferSayLanguage` when set), and passed on to your endpoint on Plivo. It is limited to 1000 bytes. To have the bot's own voice say something first, queue a `TTSSpeakFrame` before the transfer frame, or set `callcontrol.Config.TransferMessage`.

## What a successful transfer means

Strawgo sees only the carrier's answer to the transfer request. A success means the carrier **accepted the command**, not that anyone answered. Once accepted, the bot leaves the call and never learns whether the destination was busy or did not answer. What happens then depends on the carrier:

| Carrier | Success means | If the destination does not answer |
| --- | --- | --- |
| Twilio | Twilio replaced the call's instructions with the `<Dial>` TwiML; the media stream to the bot ends | Without `TransferActionURL`, the TwiML runs out after `<Dial>` and Twilio ends the call: the caller is disconnected. With `TransferActionURL`, Twilio requests that URL when the `<Dial>` ends and runs the TwiML your webhook returns, for example a voicemail or a route back to a bot |
| Telnyx | Telnyx accepted the `transfer` command and is calling the destination | Telnyx reports the failed leg through your call-control webhooks. The original call stays up, but the bot has left it, so the caller hears silence until your webhook handler acts on it (for example by hanging up or transferring again) |
| Plivo | Plivo accepted the request and will fetch your `TransferURL` | Your XML decides. Give its `<Dial>` whatever fallback you need |
| Genesys AudioHook | The `transfer_to` output variable is set; the `disconnect` is sent when the pipeline ends | Your Architect flow decides |

The behaviour after acceptance is taken from the carriers' documentation and has not been checked on a live call. Strawgo does not ship a webhook consumer for any of these; handle them in your own application.

## Transfer from your own code

Queue a `frames.TransferCallFrame` on the task. The output transport waits until the speech queued before it has been sent, then calls the serializer's `Transferer`. When the request returns it:

1. calls the frame's `OnComplete` with the result,
2. pushes a `TransferCallResultFrame` upstream,
3. on success (the carrier accepted the command), pushes an `EndTaskFrame` upstream so the pipeline ends. The serializer remembers the transfer and skips its hang-up, so the caller is not cut off.

The request uses `TransportParams.HangupTimeout`. Errors are passed through `logger.RedactError` before they are logged or reported. The errors you can get:

| Error | Meaning | The call afterwards |
| --- | --- | --- |
| a carrier error response, such as `status 400: ...` | The carrier refused the request | Still with the bot; the pipeline keeps running and the hang-up at the end still happens |
| `serializers.ErrTransferUnsupported` | The serializer cannot transfer, or lacks the credentials or `TransferURL` it needs | Still with the bot |
| `serializers.ErrTransferInProgress` | Another transfer on this call has not returned yet | Unchanged; the first transfer decides |
| `serializers.ErrTransferOutcomeUnknown` | No answer from the carrier (timeout or network error) | Unknown. Strawgo neither retries nor hangs up, because the transfer may have gone through |
| `serializers.ErrCallEnded` | The call had already ended, was already transferred, or the frame was dropped because the pipeline was ending | Gone |
| `serializers.ErrTransferCancelled` | You called `Cancel` on the frame before the transport reached it | Still with the bot; no request was sent |

If the pipeline ends while a transfer request is in flight, the hang-up waits for it: an accepted transfer skips the hang-up, a refused one hangs up as usual, an unknown outcome leaves the call alone.

`OnComplete` runs exactly once for every frame that reaches the output transport, including frames dropped because the call is ending or the pipeline is cancelled. A frame lost before it reaches the output, for example in a processor you wrote, never completes, so always wait with a deadline. When the deadline passes, call `Cancel`. It returns true when the transport has not started the transfer, and then never will; if it returns false the request is already running, so wait for `OnComplete`:

```go
func transferToSupport(ctx context.Context, task *pipeline.PipelineTask) error {
	if err := task.QueueFrame(frames.NewTTSSpeakFrame("Let me put you through to support.")); err != nil {
		return err
	}
	done := make(chan error, 1)
	f := frames.NewTransferCallFrame("+14155550100")
	f.CallerID = "+14155550199"
	f.OnComplete = func(err error) { done <- err }
	if err := task.QueueFrame(f); err != nil {
		return err
	}
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()
	select {
	case err := <-done:
		return err
	case <-timer.C:
	case <-ctx.Done():
	}
	if f.Cancel() {
		return errors.New("transfer did not start in time and was cancelled")
	}
	return <-done
}
```

Validate destinations that come from outside your code with `serializers.ValidateTransferRequest`. It accepts only E.164 numbers and `sip:user@host[:port][;params]` or `sips:` URIs with a DNS name or IPv4 host and a port from 1 to 65535 (no IPv6), and an E.164 caller ID. Every built-in `Transferer` calls it too.

For Plivo, set `TransferURL` to an endpoint you host. This one returns the XML for the requested number:

```go
func newPlivoSerializer(*http.Request) serializers.FrameSerializer {
	return serializers.NewPlivoFrameSerializer(serializers.PlivoConfig{
		AuthID:      os.Getenv("PLIVO_AUTH_ID"),
		AuthToken:   os.Getenv("PLIVO_AUTH_TOKEN"),
		TransferURL: "https://bot.example.com/plivo/transfer",
	})
}

func plivoTransferXML(w http.ResponseWriter, r *http.Request) {
	dest := r.URL.Query().Get("destination")
	if !serializers.IsE164(dest) {
		http.Error(w, "bad destination", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/xml")
	fmt.Fprintf(w, `<Response><Dial><Number>%s</Number></Dial></Response>`, dest)
}
```

Plivo fetches that URL without your credentials. Check that the number is one you allow before you dial it, as above, or sign the URL.

## Give the LLM call-control tools

`callcontrol.Register` registers up to three functions on your LLM service and returns their schemas for the context's tools:

| Tool | Arguments | What it does |
| --- | --- | --- |
| `transfer_call` | `destination`: a destination **name** | Looks the name up in your allow-list (or resolver), queues `Config.TransferMessage` if set, queues a `TransferCallFrame` and waits for the result. When the carrier accepts the transfer it returns `{"status":"transfer_accepted"}`, the LLM does not run again and the pipeline ends. |
| `end_call` | `goodbye`: optional closing line | Asks `Config.EndCallGuard` first, if set. Then queues the line (or `Config.GoodbyeMessage`) as a `TTSSpeakFrame`, then an `EndFrame`, so the call ends after the bot has finished speaking. |
| `send_dtmf` | `digits`: keys `0`-`9`, `*`, `#` | Validates the keys and queues an `OutputDTMFFrame`, sent after the speech queued before it. Returns `{"status":"queued"}`: the tool cannot see whether the carrier delivered the keys. At most `Config.MaxDTMFDigits` (default 20) per call. |

Every failure comes back to the model as `{"status":"failed","error":"..."}` with a short reason it can repeat to the caller: an unknown destination (with the allowed names), a transfer the carrier refused, another transfer in progress, a call that already ended, an unconfirmed transfer (the model is told not to retry), a guard that refused to end the call, or invalid digits. Carrier error bodies are logged, never passed to the model.

If the transfer has not started within `Config.TransferTimeout` (default 30 seconds, counted from when the tool queues it, so long queued speech counts), the tool cancels the frame and reports that the caller is still with the bot; the transport then never transfers it. If the request has already started, the tool waits for its result instead.

`Config.EndCallGuard` is a `func(ctx context.Context) error`. Return an error to refuse `end_call`, for example while a required disclosure has not been read yet; its message is passed to the model, so write it for the model and keep secrets out of it.

```go
func registerCallControl(llmSvc *llmservice.Service, task *pipeline.PipelineTask, conversation *llm.Context, own []llm.FunctionSchema, guard func(context.Context) error) error {
	schemas, err := callcontrol.Register(llmSvc, task, callcontrol.Config{
		Destinations: []callcontrol.Destination{
			{Name: "sales", Description: "new orders and pricing", Target: "+14155550100", CallerID: "+14155550199"},
			{Name: "support", Description: "problems with an existing order", Target: "sip:support@pbx.example.com"},
		},
		TransferMessage: "One moment, I'm transferring you now.",
		EndCall:         true,
		EndCallGuard:    guard,
		GoodbyeMessage:  "Thanks for calling. Goodbye!",
		SendDTMF:        true,
	})
	if err != nil {
		return err
	}
	conversation.SetTools(llm.Tools{Functions: append(own, schemas...)})
	return nil
}
```

Pass the `*pipeline.PipelineTask` as the second argument: the tools queue their frames on it, at the start of the pipeline, so they reach the output after anything the bot is already saying.

### How the allow-list protects you

The model never sees or chooses a phone number. `transfer_call` takes a destination name; the schema lists the allowed names as an `enum` and the tool looks the name up in `Config.Destinations` (case-insensitive). A name that is not there is refused, so a caller who says "transfer me to +1 900 ..." cannot make the bot dial it. Names are 1 to 64 letters, digits, spaces, `.`, `_` or `-`, with at most four digits, so a name can never be a phone number or a SIP URI. Every `Name`, `Target`, `CallerID` and `Announcement` is validated when you call `Register`, and `Register` fails on a bad entry.

If the destinations are only known at call time, set `Config.Resolver` instead of, or as well as, `Destinations`. It receives the name the model chose, but only if that name passes the same name rule; anything else, such as `4155550100` or `sip:x@example.com`, is refused without calling the resolver. It returns the destination and whether it is allowed. Return only destinations your application approves; never turn the name into a number. Whatever the resolver returns is validated again before it is dialled.

```go
func agentResolver(roster map[string]string) callcontrol.Resolver {
	return func(_ context.Context, name string) (callcontrol.Destination, bool) {
		sip, ok := roster[name]
		if !ok {
			return callcontrol.Destination{}, false
		}
		return callcontrol.Destination{Name: name, Target: sip}, true
	}
}
```

### Ending the call on Exotel and Genesys

`callcontrol.EndCall(task, closingLine)` is the same helper `end_call` uses: an optional `TTSSpeakFrame`, then an `EndFrame`. What the end of the pipeline means depends on the carrier. Twilio, Telnyx and Plivo hang up through REST, Asterisk and Jambonz send their hang-up message. Exotel just closes the stream, and the call continues to the next applet of your Exotel flow, which is how you reach an agent there. Genesys sends `disconnect` with action `transfer` and your output variables, and the Architect flow decides what happens next.

## Next steps

- [Serializers reference](../reference/serializers.md): the `Transferer` interface and each carrier's request.
- [Frames reference](../reference/frames.md#call-control): `TransferCallFrame` and `TransferCallResultFrame`.
- [Function calling](function-calling.md): result delivery, interruptions and timeouts for your own tools.
- [IVR navigation](ivr-navigation.md): navigate a phone menu with DTMF before talking to a person.
