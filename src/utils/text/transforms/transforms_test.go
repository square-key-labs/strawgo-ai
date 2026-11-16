package transforms

import (
	"regexp"
	"testing"
)

type tcase struct {
	in, want string
}

func run(t *testing.T, name string, tr Transform, cases []tcase) {
	t.Helper()
	for _, c := range cases {
		if got := tr.Transform(c.in); got != c.want {
			t.Errorf("%s(%q) = %q, want %q", name, c.in, got, c.want)
		}
	}
}

func TestCardinal(t *testing.T) {
	tests := []struct {
		loc  *Locale
		n    uint64
		want string
	}{
		{EnUS, 0, "zero"},
		{EnUS, 42, "forty-two"},
		{EnUS, 101, "one hundred and one"},
		{EnUS, 1001, "one thousand and one"},
		{EnUS, 1234, "one thousand, two hundred and thirty-four"},
		{EnUS, 1000000, "one million"},
		{EnUS, 1000100, "one million, one hundred"},
		{EnIN, 150000, "one lakh fifty thousand"},
		{EnIN, 100005, "one lakh and five"},
		{EnIN, 12345678, "one crore twenty-three lakh forty-five thousand six hundred and seventy-eight"},
		{EnIN, 12340000000, "one thousand two hundred and thirty-four crore"},
		{HiIN, 0, "शून्य"},
		{HiIN, 99, "निन्यानवे"},
		{HiIN, 250, "दो सौ पचास"},
		{HiIN, 150000, "एक लाख पचास हज़ार"},
		{HiIN, 25000000, "दो करोड़ पचास लाख"},
		{HiIN, 2026, "दो हज़ार छब्बीस"},
	}
	for _, tt := range tests {
		if got := tt.loc.Cardinal(tt.n); got != tt.want {
			t.Errorf("%s Cardinal(%d) = %q, want %q", tt.loc.Tag, tt.n, got, tt.want)
		}
	}
}

func TestHindiNumberTableIsComplete(t *testing.T) {
	seen := map[string]int{}
	for i, w := range HiIN.Below100 {
		if w == "" {
			t.Fatalf("missing Hindi word for %d", i)
		}
		if j, dup := seen[w]; dup {
			t.Fatalf("Hindi word %q used for %d and %d", w, j, i)
		}
		seen[w] = i
	}
}

func TestExpandPhoneNumbers(t *testing.T) {
	run(t, "en-US", ExpandPhoneNumbers(EnUS), []tcase{
		{"Call 123-456-7890 now", "Call 1 2 3 4 5 6 7 8 9 0 now"},
		{"Call (123) 456-7890.", "Call 1 2 3 4 5 6 7 8 9 0."},
		{"Call +1 123 456 7890", "Call 1 1 2 3 4 5 6 7 8 9 0"},
		{"Call 9876543210", "Call 9 8 7 6 5 4 3 2 1 0"},
		{"Order 123456789012 shipped", "Order 123456789012 shipped"},
		{"Nothing to read here.", "Nothing to read here."},
	})
	run(t, "en-IN", ExpandPhoneNumbers(EnIN), []tcase{
		{"Call 9876543210", "Call 9 8 7 6 5 4 3 2 1 0"},
		{"Call +91 98765 43210 today", "Call 9 1 9 8 7 6 5 4 3 2 1 0 today"},
		{"Call +91-9876543210", "Call 9 1 9 8 7 6 5 4 3 2 1 0"},
		{"Call 09876543210", "Call 0 9 8 7 6 5 4 3 2 1 0"},
		{"Call 98765-43210", "Call 9 8 7 6 5 4 3 2 1 0"},
		{"Office 080-2345 6789", "Office 0 8 0 2 3 4 5 6 7 8 9"},
		{"Toll free 1800 123 4567", "Toll free 1 8 0 0 1 2 3 4 5 6 7"},
		{"Pay 50000 now", "Pay 50000 now"},
	})
	run(t, "hi-IN", ExpandPhoneNumbers(HiIN), []tcase{
		{"नंबर 9876543210 है", "नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है"},
	})
}

func TestSpellCodes(t *testing.T) {
	run(t, "en-US", SpellCodes(EnUS), []tcase{
		{"Your OTP is 482916.", "Your OTP is 4 8 2 9 1 6."},
		{"Your verification code: 482 916", "Your verification code: 4 8 2 9 1 6"},
		{"Order ID ORD-4521-XY has shipped", "Order ID O R D 4 5 2 1 X Y has shipped"},
		{"Your account number ending in 4321 is active", "Your account number ending in 4 3 2 1 is active"},
		{"Order #4521 is ready", "Order #4 5 2 1 is ready"},
		{"Use code SAVE20 at checkout", "Use code S A V E 2 0 at checkout"},
		{"Your PIN code is 560001", "Your PIN code is 5 6 0 0 0 1"},
		{"The code is ready", "The code is ready"},
		{"I will order 25 items", "I will order 25 items"},
		{"Use barcode 123456", "Use barcode 123456"},
		{"Your OTP is 4 8 2 9 1 6.", "Your OTP is 4 8 2 9 1 6."},
	})
	run(t, "hi-IN", SpellCodes(HiIN), []tcase{
		{"आपका OTP 482916 है", "आपका OTP चार आठ दो नौ एक छह है"},
		{"आपका ओटीपी 4821 है", "आपका ओटीपी चार आठ दो एक है"},
	})
}

func TestSpellMatches(t *testing.T) {
	tr := SpellMatches(EnUS, regexp.MustCompile(`\bCUST-\d{6}\b`), regexp.MustCompile(`member (\d{8})`))
	run(t, "patterns", tr, []tcase{
		{"Ticket CUST-004213 opened", "Ticket C U S T 0 0 4 2 1 3 opened"},
		{"for member 12345678 today", "for member 1 2 3 4 5 6 7 8 today"},
		{"for member 1234 today", "for member 1234 today"},
	})
}

func TestExpandCurrency(t *testing.T) {
	run(t, "en-US", ExpandCurrency(EnUS), []tcase{
		{"$12.50", "twelve dollars and fifty cents"},
		{"$1", "one dollar"},
		{"£1.01", "one pound and one penny"},
		{"¥500", "five hundred yen"},
		{"$1,000.50", "one thousand dollars and fifty cents"},
		{"$5.500", "five dollars and fifty cents"},
		{"$3.567", "three dollars and fifty-six cents"},
		{"We raised $2.50 million", "We raised two point five million dollars"},
		{"$2.05 million", "two point zero five million dollars"},
		{"They raised $2.5m", "They raised two point five million dollars"},
		{"$500k", "five hundred thousand dollars"},
		{"€3.25 Trillion", "three point two five trillion euros"},
		{"$1.5x", "$1.5x"},
		{"$5mo", "$5mo"},
		{"US$5", "US five dollars"},
		{"₹1,50,000", "one hundred and fifty thousand rupees"},
		{"No money here.", "No money here."},
	})
	run(t, "en-IN", ExpandCurrency(EnIN), []tcase{
		{"₹1,50,000", "one lakh fifty thousand rupees"},
		{"Rs. 2.5 lakh", "two point five lakh rupees"},
		{"Rs.500/- only", "five hundred rupees only"},
		{"INR 10.50", "ten rupees and fifty paise"},
		{"Re. 1", "one rupee"},
		{"₹5cr", "five crore rupees"},
		{"₹2.5L", "two point five lakh rupees"},
		{"Pay 500 Rs. today", "Pay five hundred rupees. today"},
		{"$12.50", "twelve dollars and fifty cents"},
		{"Mrs. 500", "Mrs. 500"},
	})
	run(t, "hi-IN", ExpandCurrency(HiIN), []tcase{
		{"₹1,50,000", "एक लाख पचास हज़ार रुपये"},
		{"Rs. 2.5 lakh", "दो दशमलव पाँच लाख रुपये"},
		{"₹2.5 लाख", "दो दशमलव पाँच लाख रुपये"},
		{"₹3 करोड़", "तीन करोड़ रुपये"},
		{"₹1", "एक रुपया"},
		{"₹10.50", "दस रुपये और पचास पैसे"},
		{"$12.50", "बारह डॉलर और पचास सेंट"},
	})
}

func TestNormalizeDates(t *testing.T) {
	run(t, "en-US", NormalizeDates(EnUS), []tcase{
		{"Due 12/03/2026", "Due December 3rd, twenty twenty-six"},
		{"Meeting on 2023-05-10", "Meeting on May 10th, twenty twenty-three"},
		{"Born on 1999-03-05", "Born on March 5th, nineteen ninety-nine"},
		{"Due 07/04/1905", "Due July 4th, nineteen oh-five"},
		{"Launch on 2005-01-02", "Launch on January 2nd, two thousand and five"},
		{"On 2023-05-11 and 2023-05-21", "On May 11th, twenty twenty-three and May 21st, twenty twenty-three"},
		{"Due 13/45/2023 now", "Due 13/45/2023 now"},
	})
	run(t, "en-IN", NormalizeDates(EnIN), []tcase{
		{"Due 12/03/2026", "Due 12th March twenty twenty-six"},
		{"Due 12.03.2026", "Due 12th March twenty twenty-six"},
		{"Due 31/02/2026", "Due 31/02/2026"},
	})
	run(t, "hi-IN", NormalizeDates(HiIN), []tcase{
		{"तारीख 12/03/2026", "तारीख बारह मार्च दो हज़ार छब्बीस"},
	})
}

func TestEmailToSpeech(t *testing.T) {
	run(t, "email", EmailToSpeech, []tcase{
		{"Contact user@example.com today", "Contact user at example dot com today"},
		{"Write to first.last-name+tag@my-mail.co.in", "Write to first dot last dash name plus tag at my dash mail dot co dot in"},
		{"No address here.", "No address here."},
	})
}

func TestExpandPercentages(t *testing.T) {
	run(t, "en-US", ExpandPercentages(EnUS), []tcase{
		{"50% off", "fifty percent off"},
		{"3.5% rate", "three point five percent rate"},
		{"1,500%", "one thousand, five hundred percent"},
		{"1,234.5%", "one thousand, two hundred and thirty-four point five percent"},
		{"2.50 %", "two point five percent"},
	})
	run(t, "hi-IN", ExpandPercentages(HiIN), []tcase{
		{"50% छूट", "पचास प्रतिशत छूट"},
	})
}

func TestExpandUnits(t *testing.T) {
	run(t, "en-US", ExpandUnits(EnUS), []tcase{
		{"Run 5km at 100kph", "Run 5 kilometers at 100 kilometers per hour"},
		{"Only 1km left", "Only 1 kilometer left"},
		{"1.0km away", "1.0 kilometers away"},
		{"100 MB file", "100 megabytes file"},
		{"Run 5m", "Run 5 meters"},
		{"Buy 2L of milk", "Buy 2 liters of milk"},
		{"1 in 5 people", "1 in 5 people"},
		{"We raised $5M", "We raised $5M"},
		{"Works on 5G networks", "Works on 5G networks"},
		{"3 GHz processor", "3 gigahertz processor"},
	})
	run(t, "hi-IN", ExpandUnits(HiIN), []tcase{
		{"5km दूर", "5km दूर"},
	})
}

func TestNormalizeAcronyms(t *testing.T) {
	run(t, "acronyms", NormalizeAcronyms, []tcase{
		{"Use the API or HTTP endpoint", "Use the A P I or H T T P endpoint"},
		{"NASA and I", "N A S A and I"},
		{"Nothing Here", "Nothing Here"},
	})
}

func TestExpandNumbers(t *testing.T) {
	run(t, "cutoff", ExpandNumbers(EnUS, DefaultDigitCutoff), []tcase{
		{"Room 42", "Room forty-two"},
		{"opens in 2026", "opens in 2 0 2 6"},
		{"2025", "two thousand and twenty-five"},
		{"3000.5 units", "3 0 0 0 point 5 units"},
		{"1.0", "one point zero"},
		{"0.50", "zero point five zero"},
		{"the 21st", "the 21st"},
		{"Call 9 8 7 6 now", "Call 9 8 7 6 now"},
		{"3 0 0 0 point 5 units", "3 0 0 0 point 5 units"},
		{"point 5 and 7 apples", "point five and seven apples"},
	})
	run(t, "no cutoff", ExpandNumbers(EnUS, 0), []tcase{
		{"9999", "nine thousand, nine hundred and ninety-nine"},
		{"3.5", "three point five"},
	})
	run(t, "en-IN", ExpandNumbers(EnIN, 0), []tcase{
		{"1,50,000 people", "one lakh fifty thousand people"},
	})
	run(t, "hi-IN", ExpandNumbers(HiIN, 99), []tcase{
		{"42 लोग", "बयालीस लोग"},
		{"2026", "दो शून्य दो छह"},
		{"2.5", "दो दशमलव पाँच"},
	})
}

func TestReplaceText(t *testing.T) {
	tr, err := ReplaceText(Replacement{`\bDr\.`, "Doctor"}, Replacement{`\bvs\b`, "versus"})
	if err != nil {
		t.Fatal(err)
	}
	run(t, "replace", tr, []tcase{{"Dr. Smith vs Dr. Jones", "Doctor Smith versus Doctor Jones"}})
	if _, err := ReplaceText(Replacement{`(`, ""}); err == nil {
		t.Fatal("invalid pattern accepted")
	}
}

func TestPronunciations(t *testing.T) {
	tr, unusable := Pronunciations(map[string]string{"Siobhan": "ʃəˈvɔːn", "New York": "nuː jɔːk", "New": "njuː", "Bad": ""}, SSMLPhoneme)
	if len(unusable) != 1 || unusable[0] != "Bad" {
		t.Fatalf("unusable = %v", unusable)
	}
	run(t, "pronunciations", tr, []tcase{
		{"Hi siobhan.", `Hi <phoneme alphabet="ipa" ph="ʃəˈvɔːn">siobhan</phoneme>.`},
		{"New Yorker", `<phoneme alphabet="ipa" ph="njuː">New</phoneme> Yorker`},
		{"Siobhan-like", "Siobhan-like"},
	})
}

func TestUnchangedText(t *testing.T) {
	for _, loc := range []*Locale{EnUS, EnIN, HiIN} {
		o := DefaultVoiceFormatterOptions(loc)
		o.ExpandNumbers = true
		f, err := NewVoiceFormatter(o)
		if err != nil {
			t.Fatal(err)
		}
		for _, in := range []string{"", "Hello, how can I help you today?", "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"} {
			if got := f.Transform(in); got != in {
				t.Errorf("%s: %q changed to %q", loc.Tag, in, got)
			}
		}
	}
}

func TestVoiceFormatter(t *testing.T) {
	tests := []struct {
		loc       *Locale
		in, want  string
		configure func(*VoiceFormatterOptions)
	}{
		{
			loc:  EnIN,
			in:   "Hi! Your OTP is 482916. Pay ₹1,50,000 by 12/03/2026 or call +91 98765 43210, or mail help@bank.in for a 5% discount.",
			want: "Hi! Your O T P is 4 8 2 9 1 6. Pay one lakh fifty thousand rupees by 12th March twenty twenty-six or call 9 1 9 8 7 6 5 4 3 2 1 0, or mail help at bank dot in for a five percent discount.",
		},
		{
			loc:  EnUS,
			in:   "Your balance is $42.50 and your 100 MB plan renews 2026-01-02 for USER@EXAMPLE.COM.",
			want: "Your balance is forty-two dollars and fifty cents and your 100 megabytes plan renews January 2nd, twenty twenty-six for U S E R at E X A M P L E dot C O M.",
		},
		{
			loc:  HiIN,
			in:   "आपका OTP 482916 है और बकाया ₹2.5 लाख है।",
			want: "आपका O T P चार आठ दो नौ एक छह है और बकाया दो दशमलव पाँच लाख रुपये है।",
		},
		{
			loc:  EnUS,
			in:   "Room 42 opens in 2026. Ask Dr. Lee.",
			want: "Room forty-two opens in 2 0 2 6. Ask Doctor Lee.",
			configure: func(o *VoiceFormatterOptions) {
				o.ExpandNumbers = true
				o.NumberDigitCutoff = DefaultDigitCutoff
				o.CustomReplacements = []Replacement{{`\bDr\.`, "Doctor"}}
			},
		},
		{
			loc:  EnIN,
			in:   "Your booking reference is AB12-CD34 and member 00123456 is linked.",
			want: "Your booking reference is A B 1 2 C D 3 4 and member 0 0 1 2 3 4 5 6 is linked.",
			configure: func(o *VoiceFormatterOptions) {
				o.CodePatterns = []*regexp.Regexp{regexp.MustCompile(`\b[A-Z]{2}\d{2}-[A-Z]{2}\d{2}\b`), regexp.MustCompile(`member (\d{8})`)}
			},
		},
	}
	for _, tt := range tests {
		o := DefaultVoiceFormatterOptions(tt.loc)
		if tt.configure != nil {
			tt.configure(&o)
		}
		f, err := NewVoiceFormatter(o)
		if err != nil {
			t.Fatal(err)
		}
		got := f.Transform(tt.in)
		if got != tt.want {
			t.Errorf("%s:\n got %q\nwant %q", tt.loc.Tag, got, tt.want)
		}
		if again := f.Transform(got); again != got {
			t.Errorf("%s: second pass changed %q to %q", tt.loc.Tag, got, again)
		}
	}
}

func TestZeroOptionsChangeNothing(t *testing.T) {
	f, err := NewVoiceFormatter(VoiceFormatterOptions{})
	if err != nil {
		t.Fatal(err)
	}
	in := "Call 9876543210 about $5 and 50% on 12/03/2026."
	if got := f.Filter(in); got != in {
		t.Fatalf("got %q", got)
	}
}

func TestChainCarriesSplitCurrencyPrefix(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultVoiceFormatterOptions(EnIN))
	if err != nil {
		t.Fatal(err)
	}
	steps := []tcase{
		{"That costs Rs.", "That costs "},
		{"2.5 lakh.", "two point five lakh rupees."},
		{"Rs.", ""},
		{"500 is due.", "five hundred rupees is due."},
		{"Pay 500 Rs.", "Pay five hundred rupees."},
		{"Next.", "Next."},
	}
	for _, s := range steps {
		if got := f.Filter(s.in); got != s.want {
			t.Fatalf("Filter(%q) = %q, want %q", s.in, got, s.want)
		}
	}
	f.Filter("Costs Rs.")
	f.Reset()
	if got := f.Filter("5 apples."); got != "5 apples." {
		t.Fatalf("held text survived Reset: %q", got)
	}
}

func TestLocaleFor(t *testing.T) {
	for tag, want := range map[string]*Locale{"en": EnUS, "en-US": EnUS, "en_IN": EnIN, "hi-IN": HiIN, "HI": HiIN} {
		if got, ok := LocaleFor(tag); !ok || got != want {
			t.Errorf("LocaleFor(%q) = %v, %v", tag, got, ok)
		}
	}
	if _, ok := LocaleFor("fr-FR"); ok {
		t.Error("unexpected locale for fr-FR")
	}
}
