# Cookbook

Each recipe here is a complete, runnable program for one realistic use case, with the commands to run it, a walk-through of the interesting parts, and short variations. Pick the recipe closest to what you are building and change it from there.

> [!NOTE]
> Every recipe compiles against the code in this repository. None of the provider integrations they use has been verified against the live vendor API; see the [provider matrix](../providers/README.md).

All recipes follow the same layout: **What you build**, **Before you begin**, **The code**, **Run it**, **How it works**, **Variations** and **Next steps**. The phone recipes use Twilio; set up a number and a tunnel once with [Your first phone bot](../getting-started/first-phone-bot.md).

| Recipe | What it shows | Providers and keys |
| --- | --- | --- |
| [Customer support bot](customer-support-bot.md) | Function calling with an order lookup tool and an end-call tool that hangs up | Twilio, Deepgram, OpenAI, Cartesia: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` |
| [Appointment booking flow](appointment-booking-flow.md) | A four-node `flows` conversation with typed functions and shared state | Twilio, Deepgram, OpenAI, Cartesia: same keys as above |
| [Outbound call with voicemail](outbound-call-with-voicemail.md) | Placing a call and leaving a message or talking to a person, decided by the voicemail detector | Twilio (plus `TWILIO_FROM_NUMBER`, `PUBLIC_HOST`), Deepgram, OpenAI, Cartesia |
| [IVR navigator](ivr-navigator.md) | Pressing through a phone menu with DTMF, then talking to the human at the end | Twilio (plus `TWILIO_FROM_NUMBER`, `PUBLIC_HOST`), Deepgram, OpenAI, Cartesia; optional `ONNX_WORKER_SOCK` for VAD |
| [Call recording and transcript](call-recording-and-transcript.md) | A stereo WAV and a JSON transcript written for each call | Twilio, Deepgram, OpenAI, Cartesia |
| [Multilingual bot](multilingual-bot.md) | Switching STT language, LLM prompt and TTS voice mid-call with settings-update frames | Twilio, Deepgram, OpenAI, Cartesia with one voice per language: `CARTESIA_VOICE_EN`, `_ES`, `_FR`, `_DE` |
| [LLM failover](llm-failover.md) | `LLMSwitcher` with a failover strategy from OpenAI to Anthropic | Twilio, Deepgram, OpenAI, Anthropic (`ANTHROPIC_API_KEY`), Cartesia |
| [Browser voice widget](browser-voice-widget.md) | A WebRTC voice chat in a web page served by the same Go program | Deepgram, OpenAI, Cartesia; no telephony |
| [Background ambience](background-ambience.md) | Office noise under the bot's voice, swapped while a tool runs, with the soundfile mixer | Twilio, Deepgram, OpenAI, Cartesia; two 16-bit WAV files |
| [Metrics dashboard](metrics-dashboard.md) | A custom observer that aggregates latency, TTFB and usage per call into a JSON summary | Twilio, Deepgram, OpenAI, Cartesia |

Every recipe reads keys from environment variables; the library itself never does. To use other providers, swap the constructor lines using the [LLM](../providers/llm.md), [STT](../providers/stt.md) and [TTS](../providers/tts.md) provider pages.

## Next steps

- [Quickstart](../getting-started/quickstart.md): the smallest bot, if you have not run one yet.
- [Core concepts](../getting-started/core-concepts.md): frames, processors and pipelines explained.
- [Function calling guide](../guides/function-calling.md): the feature most recipes build on.
- [Writing a custom provider](../providers/custom.md): when the vendor you need is missing.
