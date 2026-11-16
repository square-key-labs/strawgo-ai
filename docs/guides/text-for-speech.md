# Text for speech

LLMs write for the eye: `9876543210`, `₹1,50,000`, `12/03/2026`, `OTP 482916`. Many voices read those badly, for example a phone number as "nine billion eight hundred seventy-six million...". This guide shows how to rewrite text into speakable words just before the TTS service sends it, so the bot reads phone numbers and OTPs digit by digit, and money, dates and percentages the way a person says them.

The transforms live in `github.com/square-key-labs/strawgo-ai/src/utils/text/transforms`. They are a port of Pipecat's text transforms and `VoiceFormatter`, with two locales added for Indian telephony: `en-IN` and `hi-IN`.

Nothing is enabled by default. A TTS service that you do not give a formatter sends text exactly as before.

## Before you begin

- A working bot with a TTS service, for example from [Your first phone bot](../getting-started/first-phone-bot.md).
- Terms used on this page:
  - A **transform** rewrites a piece of text and returns the new text.
  - A **formatter** is a chain of transforms applied in a fixed order.
  - A **locale** is a table of words and rules for one language and region, such as `en-IN`.
  - **Digit by digit** means `482916` becomes `4 8 2 9 1 6` (or, in Hindi, `चार आठ दो नौ एक छह`).

## Add a formatter to a TTS service

Every built-in TTS service embeds `*tts.Base`, which has `AddFilters`. Call it when you build the service:

```go
package main

import (
	"log"
	"os"

	"github.com/square-key-labs/strawgo-ai/src/services/sarvam"
	"github.com/square-key-labs/strawgo-ai/src/services/tts"
	"github.com/square-key-labs/strawgo-ai/src/utils/text/transforms"
)

func newVoice() *sarvam.TTSService {
	formatter, err := transforms.NewVoiceFormatter(transforms.DefaultVoiceFormatterOptions(transforms.EnIN))
	if err != nil {
		log.Fatal(err)
	}
	svc := sarvam.NewTTSService(sarvam.TTSConfig{APIKey: os.Getenv("SARVAM_API_KEY"), Language: "en-IN"})
	svc.AddFilters(tts.MarkdownFilter{}, formatter)
	return svc
}
```

With that formatter, the LLM sentence

```text
Your OTP is 482916. Pay ₹1,50,000 by 12/03/2026 or call +91 98765 43210, or mail help@bank.in for a 5% discount.
```

is sent to the voice as

```text
Your O T P is 4 8 2 9 1 6. Pay one lakh fifty thousand rupees by 12th March twenty twenty-six or call 9 1 9 8 7 6 5 4 3 2 1 0, or mail help at bank dot in for a five percent discount.
```

If you write your own provider on `tts.Base`, pass the formatter in `tts.Options.Filters` instead. The formatter is a `tts.TextFilter`, so it goes anywhere a filter goes. Put `tts.MarkdownFilter{}` first so that `**` and links are gone before the transforms look at the text. `AddFilters` is safe to call while the pipeline runs; the new filter applies from the next piece of text.

A formatter keeps a little state between sentences (see [Streaming](#streaming)). Each TTS service gets its own copy of that state: `AddFilters` and `tts.Options.Filters` take a fresh copy of a formatter, so one formatter value can be passed to several services. Code that calls `Filter` on a formatter directly from more than one stream should call `Fork()` once per stream.

The formatter only changes what the voice receives. The `TTSQueuedTextFrame` and `TTSTextFrame` the base emits, and so the conversation context, keep the text as the LLM wrote it. The exception is a provider that reports word timestamps: its words come back from the vendor, so the context holds the spoken form.

## Pick a locale

| Locale | `LocaleFor` tags | Numbers | Money | Phone numbers | Dates |
| --- | --- | --- | --- | --- | --- |
| `transforms.EnUS` | `en`, `en-US` | "one thousand, two hundred and thirty-four" | `$ € £ ¥`, and the rupee forms of `en-IN` read with US grouping ("one hundred and fifty thousand rupees") | US, 10 digits with optional `+1` | `MM/DD/YYYY` and ISO, "December 3rd, twenty twenty-six" |
| `transforms.EnIN` | `en-IN` | lakh and crore: "one lakh fifty thousand" | `₹`, `Rs`, `Rs.`, `Re.`, `INR`, `/-` after the amount, and the scales lakh, crore, `L`, `cr` | 10-digit mobiles, `+91`, leading `0`, landlines such as `080-2345 6789`, `1800` numbers, then the US pattern | `DD/MM/YYYY`, `DD-MM-YYYY`, `DD.MM.YYYY` and ISO, "12th March twenty twenty-six" |
| `transforms.HiIN` | `hi`, `hi-IN` | Devanagari words with lakh and crore: "एक लाख पचास हज़ार" | the same symbols, read as रुपये and पैसे; scale words लाख, करोड़, हज़ार | as `en-IN`, read as Hindi digit words | `DD/MM/YYYY`, "बारह मार्च दो हज़ार छब्बीस" |

`transforms.LocaleFor(tag)` returns the locale for a tag such as the TTS service's language setting. A nil locale means `EnUS`.

`hi-IN` covers numbers, money, percentages, dates, times and digit-by-digit reading. Units are not translated, and email addresses are read with the English words "at" and "dot".

A `Locale` is a plain struct of tables: the words for 0 to 99, the scale words, currencies, phone patterns, code keywords, month names and date layout. To add a language, build a new `Locale` value with those tables filled in and pass it wherever a locale is taken.

## The transforms

`DefaultVoiceFormatterOptions(locale)` turns on the same transforms Pipecat's `VoiceFormatter` turns on by default, plus code spelling. The zero `VoiceFormatterOptions{}` turns on nothing.

| Option | Transform | Pipecat name | Example (`en-IN` unless noted) | Default |
| --- | --- | --- | --- | --- |
| `EmailToSpeech` | `EmailToSpeech` | `email_to_speech` | `help@bank.in` → "help at bank dot in" | on |
| `ExpandPhoneNumbers` | `ExpandPhoneNumbers(locale)` | `expand_phone_numbers` | `+91 98765 43210` → "9 1 9 8 7 6 5 4 3 2 1 0" | on |
| `CodePatterns` | `SpellMatches(locale, patterns...)` | none | your own id formats, digit by digit | none |
| `SpellCodes` | `SpellCodes(locale)` | none | `OTP is 482916` → "OTP is 4 8 2 9 1 6" | on |
| `NormalizeDates` | `NormalizeDates(locale)` | `normalize_dates` | `12/03/2026` → "12th March twenty twenty-six"; in `en-US` "December 3rd, twenty twenty-six" | on |
| `NormalizeTimes` | `NormalizeTimes(locale)` | none | `10:05 PM` → "ten oh five P M", `5 PM` → "five P M", `at 18:00` → "six P M", `00:00` → "twelve A M", `09:00` → "nine o'clock"; in `hi-IN` `10:30 बजे` → "दस बजकर तीस मिनट" | on |
| `ExpandCurrency` | `ExpandCurrency(locale)` | `expand_currency` | `Rs. 2.5 lakh` → "two point five lakh rupees"; `$12.50` → "twelve dollars and fifty cents"; `₹10-12 lakh` → "ten to twelve lakh rupees"; `₹10 cr` → "ten crore rupees"; `-$5` and `$-5` → "minus five dollars" | on |
| `ExpandPercentages` | `ExpandPercentages(locale)` | `expand_percentages` | `3.5%` → "three point five percent", `-5%` → "minus five percent" | on |
| `ExpandUnits` | `ExpandUnits(locale)` | `expand_units` | `5km` → "5 kilometers", `1km` → "1 kilometer" | on |
| `NormalizeAcronyms` | `NormalizeAcronyms` | `normalize_acronyms` | `API` → "A P I" | on |
| `ExpandNumbers`, `NumberDigitCutoff` | `ExpandNumbers(locale, cutoff)` | `expand_numbers` | `42` → "forty-two"; `-5` → "minus five"; `007` → "0 0 7" but `05 March` → "five March"; with cutoff 2025, `2026` → "2 0 2 6" | off |
| `CustomReplacements` | `ReplaceText(rules...)` | `replace_text` | ``Replacement{Pattern: `\bDr\.`, Replace: "Doctor"}`` turns `Dr.` into "Doctor" | none |

A money range is read as one amount only when it is clearly a range: two amounts joined by a hyphen or en dash with no spaces (`₹10-12 lakh`, `$10-12`), or joined by "to" when the second amount has its own currency symbol or a scale word follows (`$10 to $12`, `₹10 to 12 lakh`). `₹500 to 9876543210`, `$20 to 5 friends` and `₹999 - 10% off` are read as one amount followed by the rest of the sentence.

A clock time is read when it has `AM` or `PM`, when the hour is written with a leading zero (`09:00`), when a word such as "at", "by", "from", "to", "until", "before" or "after" (in `hi-IN` also सुबह, शाम, रात, दोपहर) comes before it, or when "o'clock" or बजे comes after it. Other `H:MM` text such as scores, ratios and verses (`21:19`, `16:10`, `John 3:16`) is left as written. A 24-hour time is read as a 12-hour time with "P M" or "A M" (`at 18:00` → "six P M", `00:00` → "twelve A M"). A following "o'clock" or बजे is not repeated.

The formatter applies them in the order of the table, which is Pipecat's order with the code transforms added after phone numbers and times after dates. It leaves web addresses (`https://...`, `www....`) exactly as written. Email runs before phone numbers and acronyms, so a digit-only domain or an all-caps address is still read as an address. Units run before acronyms, so `MB` becomes "megabytes" and not "M B". Phone numbers and codes run before plain numbers, so they are read digit by digit and not as quantities.

The rules follow Pipecat: `ExpandNumbers` reads decimals digit by digit after "point", keeping trailing zeros (`1.0` → "one point zero"); a number above `NumberDigitCutoff` is read digit by digit, and `0` means no cutoff. Currency drops sub-cent digits and reads a scaled amount as one number (`$2.50 million` → "two point five million dollars"). A unit takes the singular only for a bare `1`. Beyond Pipecat, `ExpandNumbers` reads a number with a leading zero digit by digit, and leaves alone numbers next to `:` or `/` (`10:30:00`, `24/7`, `3/4`) and dotted runs such as `v2.5.1` and `192.168.1.1`. `CustomReplacements` uses Go `regexp` syntax, so a group is `$1`, not `\1`. An invalid pattern makes `NewVoiceFormatter` return an error.

Each transform can also be used on its own, and `transforms.NewChain(t1, t2, ...)` composes any set of them in the order given. A `*Chain` is both a `Transform` and a `tts.TextFilter`.

## Read ids digit by digit

`ExpandPhoneNumbers` spells numbers that look like phone numbers for the locale, and groups of four digits such as card and Aadhaar numbers (`9123 4567 8901`). It skips a match right after a currency symbol or right before a currency or scale word, so `$2500000000` and `250 500 1000 rupees` are read as money. `SpellCodes` spells an id that follows a keyword: OTP, verification code, PIN, pin code, account number, order id, booking id, PNR, reference number, and others in the locale's `CodeKeywords` table, including Devanagari keywords such as ओटीपी for `hi-IN`. Fillers such as "is", "number", "no.", ":" and "#" may sit between the keyword and the id, and the abbreviations `Ref.`, `A/c.` and `Acct.` count as keywords (`Ref. 12345` → "Ref. 1 2 3 4 5"). Any other full stop after the keyword ends the search, so in "Please enter the code. 2024 was a good year." the year is not spelled. The id must contain a digit and at least three letters or digits, so "order 25 items" and "the code is ready" are left alone. Letters are spelled as capitals (`ORD-4521-XY` → "O R D 4 5 2 1 X Y").

When your ids have a known shape, mark them with `CodePatterns`. A pattern with a group spells only the group:

```go
opts := transforms.DefaultVoiceFormatterOptions(transforms.EnIN)
opts.CodePatterns = []*regexp.Regexp{
	regexp.MustCompile(`\bCUST-\d{6}\b`),
	regexp.MustCompile(`member (\d{8})`),
}
formatter, err := transforms.NewVoiceFormatter(opts)
```

## Streaming

The base aggregates LLM tokens into sentences and calls the filter once per sentence, so a phone number, an amount or an id inside one sentence always reaches the transforms whole, and no text is transformed twice. What the base hands the filter has not been transformed yet.

The sentence splitter treats `Rs.` followed by a space as the end of a sentence, so `Rs. 2.5 lakh` arrives as `Rs.` and `2.5 lakh.`. A formatter with `ExpandCurrency` holds a trailing `Rs.`, `Re.` or `रु.` back and puts it in front of the next sentence when that sentence starts with an amount, so the voice says "two point five lakh rupees". When the next sentence does not start with an amount, the held piece is spoken as a word in front of it: `Prices in Rs.` then `Thank you.` is spoken as "Prices in" then "rupees. Thank you.". When the text before it ends with a digit, as in `500 Rs.`, nothing is held, and `500 Rs` is read as "five hundred rupees". Interruptions and the end of a turn call `Reset`, which drops held text, so held text never moves into another turn. A `TTSSpeakFrame` that arrives in the middle of a turn is filtered with its own fresh copy of the formatter, so it neither receives nor drops the turn's held `Rs.`.

Transforms leave their own output alone, so running the formatter on text it already produced changes nothing: spelled digits are not grouped into numbers again, and words are never turned back into digits.

Limits:

- With `AggregateTokens` (the `StreamTokens` option on Cartesia, ElevenLabs, Inworld, Neuphonic, Rime and Sarvam), the filter sees single LLM tokens, so a number split across tokens is not recognised. Use sentence aggregation with a formatter.
- Only the currency prefixes above are carried across sentences. Other abbreviations the splitter does not know, such as `No.` in `Account No. 1234567`, split the keyword from the id, so `SpellCodes` does not see them together. The id is still spelled if it looks like a phone number or matches a `CodePatterns` entry. Writing "account number" in the prompt avoids it.
- A sentence that is only `Rs.` sends nothing to the voice, so its `TTSQueuedTextFrame` and `TTSTextFrame` are not emitted; the amount in the next sentence is spoken with "rupees". A turn that ends with `Rs.` and no amount loses the `Rs.`.

Known wrong or unhandled readings, from an outside review of 46 sentences in the three locales:

- Years are not detected. With `ExpandNumbers`, `1999` is read as "one thousand, nine hundred and ninety-nine" (in `hi-IN` "एक हज़ार नौ सौ निन्यानवे"), not as a year, unless it is part of a full date. Set `NumberDigitCutoff` or leave `ExpandNumbers` off.
- Times with seconds (`10:30:00`), two-digit years (`3/4/24`), fractions (`3/4`) and dates in the other locale's order (`25/12/2024` in `en-US`, `12/25/2024` in `en-IN`) are left as written.
- `10-12 people` and `9-5` are read "ten-twelve" and "nine-five" with `ExpandNumbers`; only currency ranges are read with "to". A "to" range without a second symbol or a scale word, such as `₹5,000 to 6,000`, reads only the first amount as money.
- A clock time with no cue word (`lunch 13:30`) is left as written, and a score after a cue word (`by 21:19`) is read as a time.
- A US local number such as `555-0100` is not a phone number; with `ExpandNumbers` it becomes "five hundred and fifty-five-0 1 0 0".
- A comma as decimal separator (`€5,50`) is not understood.
- `NormalizeAcronyms` spells every all-caps word, so shouted text such as `PLEASE HOLD` becomes "P L E A S E H O L D".
- A lowercase `g` after a number is grams, so `4g data` becomes "4 grams data". `5G` is left alone.
- `hi-IN` has no unit table: `5 kg` and `6ft` are left as written, and `GB` is spelled "G B".
- IP addresses and version numbers are left as written rather than read part by part.
- Text in one language under another locale is read with that locale's words, for example English number words inside a Hindi sentence under `en-IN`. Pick the locale that matches the voice's language.
- In `en-US`, Indian phone formats such as `+91 98765 43210` and `080-2345 6789` are not phone numbers; with `ExpandNumbers` they are read as quantities. Use `EnIN` for Indian numbers.

## Pick transforms per vendor

Some vendors normalise text themselves. Running both can double-handle text: a vendor that already reads `₹500` correctly gains nothing from `ExpandCurrency`, and a vendor that groups digits may undo the spacing. The safe pattern is to always spell ids and phone numbers, which vendors often read as quantities, and to leave the rest to a vendor that normalises:

```go
opts := transforms.VoiceFormatterOptions{
	Locale:             transforms.EnIN,
	ExpandPhoneNumbers: true,
	SpellCodes:         true,
}
```

| Vendor in Strawgo | Its own normalisation switch | Suggested formatter |
| --- | --- | --- |
| Sarvam | The service always sends `enable_preprocessing: true` for every model except `bulbul:v2`, where it is false unless you set it in `Params`. | Phone numbers and codes with `bulbul:v3` and newer; the full default formatter with `bulbul:v2` and preprocessing off. Use `HiIN` or `EnIN` to match the `Language`. |
| ElevenLabs HTTP | `ApplyTextNormalization` | Phone numbers and codes when the vendor normalises; the default formatter otherwise. |
| Inworld websocket | `ApplyTextNormalization` | As ElevenLabs. |
| MiniMax | `Params` key `text_normalization_mode` | As ElevenLabs. |
| Speechify | `Params` key `text_normalization` | As ElevenLabs. |
| Piper, XTTS (self-hosted) | none | The default formatter with `ExpandNumbers: true`; local models read digits poorly. |
| Every other TTS service | none in Strawgo | The default formatter. |

Vendor behaviour changes and has not been verified live from Strawgo. Listen to a few test calls with your voice and locale before choosing.

## Pronunciations

`transforms.Pronunciations(words, format)` replaces whole words, case-insensitively and longest first, with the vendor's pronunciation markup. `transforms.SSMLPhoneme` writes an SSML `<phoneme>` tag; pass your own `PronunciationFormatter` for other markup. Words the formatter cannot handle are returned in a list and spoken as written. Add it last, so no later transform rewrites the markup:

```go
pronounce, unusable := transforms.Pronunciations(map[string]string{"Siobhan": "ʃəˈvɔːn"}, transforms.SSMLPhoneme)
if len(unusable) > 0 {
	log.Printf("spoken as written: %v", unusable)
}
svc.AddFilters(transforms.NewChain(formatter, pronounce))
```

A chain used as a filter passes each piece of text to the `Filter` and `Reset` of a formatter inside it, so the formatter still holds `Rs.` back across sentences when wrapped this way.

Only use it with a voice that reads the markup; others will speak the tags.

## Next steps

- [STT and TTS bases](../reference/stt-tts-bases.md): where `TextFilter` sits in the TTS base.
- [TTS providers](../providers/tts.md): each vendor's options, including the normalisation switches above.
