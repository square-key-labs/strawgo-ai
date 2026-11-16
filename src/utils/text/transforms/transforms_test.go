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
		{HiIN, 44, "चौवालीस"},
		{HiIN, 79, "उनासी"},
		{HiIN, 87, "सतासी"},
		{HiIN, 88, "अठासी"},
		{HiIN, 95, "पचानवे"},
		{HiIN, 1999, "एक हज़ार नौ सौ निन्यानवे"},
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
		{"₹0.75", "पचहत्तर पैसे"},
		{"₹0.01", "एक पैसा"},
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
		{"जन्मतिथि 05/08/1999", "जन्मतिथि पाँच अगस्त उन्नीस सौ निन्यानवे"},
		{"1905-01-26", "छब्बीस जनवरी उन्नीस सौ पाँच"},
		{"01/01/2000", "एक जनवरी दो हज़ार"},
		{"01/01/1900", "एक जनवरी उन्नीस सौ"},
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

func TestReviewSentences(t *testing.T) {
	tests := []struct {
		loc     *Locale
		numbers bool
		in      string
		want    string
	}{
		{EnUS, false, "Your total is $1,234.56 today.", "Your total is one thousand, two hundred and thirty-four dollars and fifty-six cents today."},
		{EnUS, false, "That costs $5 or $5.5 or $0.99 or $1.005.", "That costs five dollars or five dollars and fifty cents or zero dollars and ninety-nine cents or one dollar."},
		{EnUS, false, "We raised $5 million and $2.5M and $3k.", "We raised five million dollars and two point five million dollars and three thousand dollars."},
		{EnUS, false, "The price dropped by -5% to 12.50% and 0.5%.", "The price dropped by minus five percent to twelve point five percent and zero point five percent."},
		{EnUS, false, "Between 10-12 people at 10:30 AM, or 10:05 PM.", "Between 10-12 people at ten thirty A M, or ten oh five P M."},
		{EnUS, false, "The temperature is -5 degrees.", "The temperature is -5 degrees."},
		{EnUS, false, "In 2024 and 1999 and 2005 and 1905.", "In 2024 and 1999 and 2005 and 1905."},
		{EnUS, false, "Version v2.5.1 and iOS 17.4.1 released.", "Version v2.5.1 and iOS 17.4.1 released."},
		{EnUS, false, "Server IP is 192.168.1.1 port 8080.", "Server I P is 192.168.1.1 port 8080."},
		{EnUS, false, "Visit https://example.com/path?id=42&x=5km now.", "Visit https://example.com/path?id=42&x=5km now."},
		{EnUS, false, "Call +1 (415) 555-0100 or 415-555-0100 before 5 PM.", "Call 1 4 1 5 5 5 5 0 1 0 0 or 4 1 5 5 5 5 0 1 0 0 before five P M."},
		{EnUS, false, "Call 4155550100 and pay 500.", "Call 4 1 5 5 5 5 0 1 0 0 and pay 500."},
		{EnUS, false, "You have 3 items, the 1st and 22nd, about 3/4 full.", "You have 3 items, the 1st and 22nd, about 3/4 full."},
		{EnUS, false, "Order number 1234567890123456 shipped.", "Order number 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 shipped."},
		{EnUS, false, "Your OTP is 004216. It expires in 10 minutes.", "Your O T P is 0 0 4 2 1 6. It expires in 10 minutes."},
		{EnUS, false, "Use promo code SAVE20 for 20% off.", "Use promo code S A V E 2 0 for twenty percent off."},
		{EnUS, false, "The code is valid for 30 days.", "The code is valid for 30 days."},
		{EnUS, false, "Please enter the code. 2024 was a good year.", "Please enter the code. 2024 was a good year."},
		{EnUS, false, "Your PIN code 560001 is serviceable.", "Your P I N code 5 6 0 0 0 1 is serviceable."},
		{EnUS, false, "Email john.doe+test@example.co.in. Thanks.", "Email john dot doe plus test at example dot co dot in. Thanks."},
		{EnUS, false, "Flight AI 0420 leaves at 0900 hours, agent 007.", "Flight A I 0420 leaves at 0900 hours, agent 007."},
		{EnUS, false, "Plans at 250 500 1000 rupees.", "Plans at 250 500 1000 rupees."},
		{EnUS, false, "Card 4111 1111 1111 1111 expires 12/26.", "Card 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 expires 12/26."},
		{EnUS, false, "Ratio is 1,000.50 and 1,50,000 and 12,34,567.", "Ratio is 1,000.50 and 1,50,000 and 12,34,567."},
		{EnUS, false, "Tracking ID: AB-1234/XY 5678 arrives soon.", "Tracking I D: A B 1 2 3 4 X Y 5 6 7 8 arrives soon."},
		{EnUS, false, "Account number ending in 4321 has $2500000000.", "Account number ending in 4 3 2 1 has two billion, five hundred million dollars."},
		{EnUS, false, "Your reference number is 12345 dated 12/25/2024.", "Your reference number is 1 2 3 4 5 dated December 25th, twenty twenty-four."},
		{EnUS, false, "Booking code 2024-12-25 is confirmed.", "Booking code 2 0 2 4 1 2 2 5 is confirmed."},
		{EnUS, false, "A 3.5% fee, 2x speed, top 10, room 404, 24/7 support.", "A three point five percent fee, 2x speed, top 10, room 404, 24/7 support."},
		{EnUS, false, "Rs. 500 and Rs 1,50,000 and ₹2.5 lakh and 500 Rs and Rs.499/- only.", "five hundred rupees and one hundred and fifty thousand rupees and two point five lakh rupees and five hundred rupees and four hundred and ninety-nine rupees only."},
		{EnUS, false, "INR 2,00,000 or ₹10 cr or ₹5L or Re. 1 or ₹0.50.", "two hundred thousand rupees or ten crore rupees or five lakh rupees or one rupee or zero rupees and fifty paise."},
		{EnUS, false, "Others 5 and hours 10 and offers 3.", "Others 5 and hours 10 and offers 3."},
		{EnUS, false, "Call 9876543210 or +91 98765 43210 or 080-2345 6789 or 1800 123 4567.", "Call 9 8 7 6 5 4 3 2 1 0 or +91 98765 43210 or 080-2345 6789 or 1 8 0 0 1 2 3 4 5 6 7."},
		{EnUS, false, "Date 25/12/2024 and 25.12.2024 and 01-02-2024.", "Date 25/12/2024 and 25.12.2024 and January 2nd, twenty twenty-four."},
		{EnUS, false, "आपका बकाया ₹1,50,000 है और OTP 123456 है।", "आपका बकाया one hundred and fifty thousand rupees है और O T P 1 2 3 4 5 6 है।"},
		{EnUS, false, "कुल राशि रु. 2.5 लाख है, 15% छूट, 25/12/2024 तक।", "कुल राशि two point five lakh rupees है, fifteen percent छूट, 25/12/2024 तक।"},
		{EnUS, false, "आपका order 3 दिन में आएगा, total Rs. 499 है, call करें 9876543210 पर।", "आपका order 3 दिन में आएगा, total four hundred and ninety-nine rupees है, call करें 9 8 7 6 5 4 3 2 1 0 पर।"},
		{EnUS, false, "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।", "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।"},
		{EnUS, false, "Salary between 80000 90000 rupees, range ₹10-12 lakh.", "Salary between 80000 90000 rupees, range ten to twelve lakh rupees."},
		{EnUS, false, "Aadhaar 9123 4567 8901 and PAN number ABCDE1234F.", "Aadhaar 9 1 2 3 4 5 6 7 8 9 0 1 and P A N number A B C D E 1 2 3 4 F."},
		{EnUS, true, "Your total is $1,234.56 today.", "Your total is one thousand, two hundred and thirty-four dollars and fifty-six cents today."},
		{EnUS, true, "That costs $5 or $5.5 or $0.99 or $1.005.", "That costs five dollars or five dollars and fifty cents or zero dollars and ninety-nine cents or one dollar."},
		{EnUS, true, "We raised $5 million and $2.5M and $3k.", "We raised five million dollars and two point five million dollars and three thousand dollars."},
		{EnUS, true, "The price dropped by -5% to 12.50% and 0.5%.", "The price dropped by minus five percent to twelve point five percent and zero point five percent."},
		{EnUS, true, "The temperature is -5 degrees.", "The temperature is minus five degrees."},
		{EnUS, true, "Version v2.5.1 and iOS 17.4.1 released.", "Version v2.5.1 and iOS 17.4.1 released."},
		{EnUS, true, "Server IP is 192.168.1.1 port 8080.", "Server I P is 192.168.1.1 port eight thousand and eighty."},
		{EnUS, true, "Visit https://example.com/path?id=42&x=5km now.", "Visit https://example.com/path?id=42&x=5km now."},
		{EnUS, true, "Call +1 (415) 555-0100 or 415-555-0100 before 5 PM.", "Call 1 4 1 5 5 5 5 0 1 0 0 or 4 1 5 5 5 5 0 1 0 0 before five P M."},
		{EnUS, true, "Call 4155550100 and pay 500.", "Call 4 1 5 5 5 5 0 1 0 0 and pay five hundred."},
		{EnUS, true, "You have 3 items, the 1st and 22nd, about 3/4 full.", "You have three items, the 1st and 22nd, about 3/4 full."},
		{EnUS, true, "Order number 1234567890123456 shipped.", "Order number 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 shipped."},
		{EnUS, true, "Your OTP is 004216. It expires in 10 minutes.", "Your O T P is 0 0 4 2 1 6. It expires in ten minutes."},
		{EnUS, true, "Use promo code SAVE20 for 20% off.", "Use promo code S A V E 2 0 for twenty percent off."},
		{EnUS, true, "The code is valid for 30 days.", "The code is valid for thirty days."},
		{EnUS, true, "Your PIN code 560001 is serviceable.", "Your P I N code 5 6 0 0 0 1 is serviceable."},
		{EnUS, true, "Email john.doe+test@example.co.in. Thanks.", "Email john dot doe plus test at example dot co dot in. Thanks."},
		{EnUS, true, "Flight AI 0420 leaves at 0900 hours, agent 007.", "Flight A I 0 4 2 0 leaves at 0 9 0 0 hours, agent 0 0 7."},
		{EnUS, true, "Plans at 250 500 1000 rupees.", "Plans at two hundred and fifty five hundred one thousand rupees."},
		{EnUS, true, "Card 4111 1111 1111 1111 expires 12/26.", "Card 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 expires 12/26."},
		{EnUS, true, "Tracking ID: AB-1234/XY 5678 arrives soon.", "Tracking I D: A B 1 2 3 4 X Y 5 6 7 8 arrives soon."},
		{EnUS, true, "Account number ending in 4321 has $2500000000.", "Account number ending in 4 3 2 1 has two billion, five hundred million dollars."},
		{EnUS, true, "Your reference number is 12345 dated 12/25/2024.", "Your reference number is 1 2 3 4 5 dated December 25th, twenty twenty-four."},
		{EnUS, true, "Booking code 2024-12-25 is confirmed.", "Booking code 2 0 2 4 1 2 2 5 is confirmed."},
		{EnUS, true, "A 3.5% fee, 2x speed, top 10, room 404, 24/7 support.", "A three point five percent fee, 2x speed, top ten, room four hundred and four, 24/7 support."},
		{EnUS, true, "Rs. 500 and Rs 1,50,000 and ₹2.5 lakh and 500 Rs and Rs.499/- only.", "five hundred rupees and one hundred and fifty thousand rupees and two point five lakh rupees and five hundred rupees and four hundred and ninety-nine rupees only."},
		{EnUS, true, "INR 2,00,000 or ₹10 cr or ₹5L or Re. 1 or ₹0.50.", "two hundred thousand rupees or ten crore rupees or five lakh rupees or one rupee or zero rupees and fifty paise."},
		{EnUS, true, "Others 5 and hours 10 and offers 3.", "Others five and hours ten and offers three."},
		{EnUS, true, "Date 25/12/2024 and 25.12.2024 and 01-02-2024.", "Date 25/12/2024 and 25.12.2024 and January 2nd, twenty twenty-four."},
		{EnUS, true, "आपका बकाया ₹1,50,000 है और OTP 123456 है।", "आपका बकाया one hundred and fifty thousand rupees है और O T P 1 2 3 4 5 6 है।"},
		{EnUS, true, "कुल राशि रु. 2.5 लाख है, 15% छूट, 25/12/2024 तक।", "कुल राशि two point five lakh rupees है, fifteen percent छूट, 25/12/2024 तक।"},
		{EnUS, true, "आपका order 3 दिन में आएगा, total Rs. 499 है, call करें 9876543210 पर।", "आपका order three दिन में आएगा, total four hundred and ninety-nine rupees है, call करें 9 8 7 6 5 4 3 2 1 0 पर।"},
		{EnUS, true, "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।", "कोड four thousand, eight hundred and twenty-one दर्ज करें। twenty-one और ninety-nine और one hundred और one thousand and one और seventy-nine और eighty-nine।"},
		{EnUS, true, "Salary between 80000 90000 rupees, range ₹10-12 lakh.", "Salary between eighty thousand ninety thousand rupees, range ten to twelve lakh rupees."},
		{EnUS, true, "Aadhaar 9123 4567 8901 and PAN number ABCDE1234F.", "Aadhaar 9 1 2 3 4 5 6 7 8 9 0 1 and P A N number A B C D E 1 2 3 4 F."},
		{EnIN, false, "Your total is $1,234.56 today.", "Your total is one thousand two hundred and thirty-four dollars and fifty-six cents today."},
		{EnIN, false, "That costs $5 or $5.5 or $0.99 or $1.005.", "That costs five dollars or five dollars and fifty cents or zero dollars and ninety-nine cents or one dollar."},
		{EnIN, false, "We raised $5 million and $2.5M and $3k.", "We raised five million dollars and two point five million dollars and three thousand dollars."},
		{EnIN, false, "The price dropped by -5% to 12.50% and 0.5%.", "The price dropped by minus five percent to twelve point five percent and zero point five percent."},
		{EnIN, false, "Between 10-12 people at 10:30 AM, or 10:05 PM.", "Between 10-12 people at ten thirty A M, or ten oh five P M."},
		{EnIN, false, "The temperature is -5 degrees.", "The temperature is -5 degrees."},
		{EnIN, false, "In 2024 and 1999 and 2005 and 1905.", "In 2024 and 1999 and 2005 and 1905."},
		{EnIN, false, "Version v2.5.1 and iOS 17.4.1 released.", "Version v2.5.1 and iOS 17.4.1 released."},
		{EnIN, false, "Server IP is 192.168.1.1 port 8080.", "Server I P is 192.168.1.1 port 8080."},
		{EnIN, false, "Visit https://example.com/path?id=42&x=5km now.", "Visit https://example.com/path?id=42&x=5km now."},
		{EnIN, false, "Call +1 (415) 555-0100 or 415-555-0100 before 5 PM.", "Call 1 4 1 5 5 5 5 0 1 0 0 or 4 1 5 5 5 5 0 1 0 0 before five P M."},
		{EnIN, false, "Call 4155550100 and pay 500.", "Call 4 1 5 5 5 5 0 1 0 0 and pay 500."},
		{EnIN, false, "You have 3 items, the 1st and 22nd, about 3/4 full.", "You have 3 items, the 1st and 22nd, about 3/4 full."},
		{EnIN, false, "Order number 1234567890123456 shipped.", "Order number 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 shipped."},
		{EnIN, false, "Your OTP is 004216. It expires in 10 minutes.", "Your O T P is 0 0 4 2 1 6. It expires in 10 minutes."},
		{EnIN, false, "Use promo code SAVE20 for 20% off.", "Use promo code S A V E 2 0 for twenty percent off."},
		{EnIN, false, "The code is valid for 30 days.", "The code is valid for 30 days."},
		{EnIN, false, "Please enter the code. 2024 was a good year.", "Please enter the code. 2024 was a good year."},
		{EnIN, false, "Your PIN code 560001 is serviceable.", "Your P I N code 5 6 0 0 0 1 is serviceable."},
		{EnIN, false, "Email john.doe+test@example.co.in. Thanks.", "Email john dot doe plus test at example dot co dot in. Thanks."},
		{EnIN, false, "Flight AI 0420 leaves at 0900 hours, agent 007.", "Flight A I 0420 leaves at 0900 hours, agent 007."},
		{EnIN, false, "Plans at 250 500 1000 rupees.", "Plans at 250 500 1000 rupees."},
		{EnIN, false, "Card 4111 1111 1111 1111 expires 12/26.", "Card 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 expires 12/26."},
		{EnIN, false, "Ratio is 1,000.50 and 1,50,000 and 12,34,567.", "Ratio is 1,000.50 and 1,50,000 and 12,34,567."},
		{EnIN, false, "Tracking ID: AB-1234/XY 5678 arrives soon.", "Tracking I D: A B 1 2 3 4 X Y 5 6 7 8 arrives soon."},
		{EnIN, false, "Account number ending in 4321 has $2500000000.", "Account number ending in 4 3 2 1 has two hundred and fifty crore dollars."},
		{EnIN, false, "Your reference number is 12345 dated 12/25/2024.", "Your reference number is 1 2 3 4 5 dated 12/25/2024."},
		{EnIN, false, "Booking code 2024-12-25 is confirmed.", "Booking code 2 0 2 4 1 2 2 5 is confirmed."},
		{EnIN, false, "A 3.5% fee, 2x speed, top 10, room 404, 24/7 support.", "A three point five percent fee, 2x speed, top 10, room 404, 24/7 support."},
		{EnIN, false, "Rs. 500 and Rs 1,50,000 and ₹2.5 lakh and 500 Rs and Rs.499/- only.", "five hundred rupees and one lakh fifty thousand rupees and two point five lakh rupees and five hundred rupees and four hundred and ninety-nine rupees only."},
		{EnIN, false, "INR 2,00,000 or ₹10 cr or ₹5L or Re. 1 or ₹0.50.", "two lakh rupees or ten crore rupees or five lakh rupees or one rupee or zero rupees and fifty paise."},
		{EnIN, false, "Others 5 and hours 10 and offers 3.", "Others 5 and hours 10 and offers 3."},
		{EnIN, false, "Call 9876543210 or +91 98765 43210 or 080-2345 6789 or 1800 123 4567.", "Call 9 8 7 6 5 4 3 2 1 0 or 9 1 9 8 7 6 5 4 3 2 1 0 or 0 8 0 2 3 4 5 6 7 8 9 or 1 8 0 0 1 2 3 4 5 6 7."},
		{EnIN, false, "Date 25/12/2024 and 25.12.2024 and 01-02-2024.", "Date 25th December twenty twenty-four and 25th December twenty twenty-four and 1st February twenty twenty-four."},
		{EnIN, false, "आपका बकाया ₹1,50,000 है और OTP 123456 है।", "आपका बकाया one lakh fifty thousand rupees है और O T P 1 2 3 4 5 6 है।"},
		{EnIN, false, "कुल राशि रु. 2.5 लाख है, 15% छूट, 25/12/2024 तक।", "कुल राशि two point five lakh rupees है, fifteen percent छूट, 25th December twenty twenty-four तक।"},
		{EnIN, false, "आपका order 3 दिन में आएगा, total Rs. 499 है, call करें 9876543210 पर।", "आपका order 3 दिन में आएगा, total four hundred and ninety-nine rupees है, call करें 9 8 7 6 5 4 3 2 1 0 पर।"},
		{EnIN, false, "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।", "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।"},
		{EnIN, false, "Salary between 80000 90000 rupees, range ₹10-12 lakh.", "Salary between 80000 90000 rupees, range ten to twelve lakh rupees."},
		{EnIN, false, "Aadhaar 9123 4567 8901 and PAN number ABCDE1234F.", "Aadhaar 9 1 2 3 4 5 6 7 8 9 0 1 and P A N number A B C D E 1 2 3 4 F."},
		{EnIN, true, "Your total is $1,234.56 today.", "Your total is one thousand two hundred and thirty-four dollars and fifty-six cents today."},
		{EnIN, true, "That costs $5 or $5.5 or $0.99 or $1.005.", "That costs five dollars or five dollars and fifty cents or zero dollars and ninety-nine cents or one dollar."},
		{EnIN, true, "We raised $5 million and $2.5M and $3k.", "We raised five million dollars and two point five million dollars and three thousand dollars."},
		{EnIN, true, "The price dropped by -5% to 12.50% and 0.5%.", "The price dropped by minus five percent to twelve point five percent and zero point five percent."},
		{EnIN, true, "The temperature is -5 degrees.", "The temperature is minus five degrees."},
		{EnIN, true, "Version v2.5.1 and iOS 17.4.1 released.", "Version v2.5.1 and iOS 17.4.1 released."},
		{EnIN, true, "Server IP is 192.168.1.1 port 8080.", "Server I P is 192.168.1.1 port eight thousand and eighty."},
		{EnIN, true, "Visit https://example.com/path?id=42&x=5km now.", "Visit https://example.com/path?id=42&x=5km now."},
		{EnIN, true, "Call +1 (415) 555-0100 or 415-555-0100 before 5 PM.", "Call 1 4 1 5 5 5 5 0 1 0 0 or 4 1 5 5 5 5 0 1 0 0 before five P M."},
		{EnIN, true, "Call 4155550100 and pay 500.", "Call 4 1 5 5 5 5 0 1 0 0 and pay five hundred."},
		{EnIN, true, "You have 3 items, the 1st and 22nd, about 3/4 full.", "You have three items, the 1st and 22nd, about 3/4 full."},
		{EnIN, true, "Order number 1234567890123456 shipped.", "Order number 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 shipped."},
		{EnIN, true, "Your OTP is 004216. It expires in 10 minutes.", "Your O T P is 0 0 4 2 1 6. It expires in ten minutes."},
		{EnIN, true, "Use promo code SAVE20 for 20% off.", "Use promo code S A V E 2 0 for twenty percent off."},
		{EnIN, true, "The code is valid for 30 days.", "The code is valid for thirty days."},
		{EnIN, true, "Your PIN code 560001 is serviceable.", "Your P I N code 5 6 0 0 0 1 is serviceable."},
		{EnIN, true, "Email john.doe+test@example.co.in. Thanks.", "Email john dot doe plus test at example dot co dot in. Thanks."},
		{EnIN, true, "Flight AI 0420 leaves at 0900 hours, agent 007.", "Flight A I 0 4 2 0 leaves at 0 9 0 0 hours, agent 0 0 7."},
		{EnIN, true, "Plans at 250 500 1000 rupees.", "Plans at two hundred and fifty five hundred one thousand rupees."},
		{EnIN, true, "Card 4111 1111 1111 1111 expires 12/26.", "Card 4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 expires 12/26."},
		{EnIN, true, "Tracking ID: AB-1234/XY 5678 arrives soon.", "Tracking I D: A B 1 2 3 4 X Y 5 6 7 8 arrives soon."},
		{EnIN, true, "Account number ending in 4321 has $2500000000.", "Account number ending in 4 3 2 1 has two hundred and fifty crore dollars."},
		{EnIN, true, "Your reference number is 12345 dated 12/25/2024.", "Your reference number is 1 2 3 4 5 dated 12/25/2024."},
		{EnIN, true, "Booking code 2024-12-25 is confirmed.", "Booking code 2 0 2 4 1 2 2 5 is confirmed."},
		{EnIN, true, "A 3.5% fee, 2x speed, top 10, room 404, 24/7 support.", "A three point five percent fee, 2x speed, top ten, room four hundred and four, 24/7 support."},
		{EnIN, true, "Rs. 500 and Rs 1,50,000 and ₹2.5 lakh and 500 Rs and Rs.499/- only.", "five hundred rupees and one lakh fifty thousand rupees and two point five lakh rupees and five hundred rupees and four hundred and ninety-nine rupees only."},
		{EnIN, true, "INR 2,00,000 or ₹10 cr or ₹5L or Re. 1 or ₹0.50.", "two lakh rupees or ten crore rupees or five lakh rupees or one rupee or zero rupees and fifty paise."},
		{EnIN, true, "Others 5 and hours 10 and offers 3.", "Others five and hours ten and offers three."},
		{EnIN, true, "Date 25/12/2024 and 25.12.2024 and 01-02-2024.", "Date 25th December twenty twenty-four and 25th December twenty twenty-four and 1st February twenty twenty-four."},
		{EnIN, true, "आपका बकाया ₹1,50,000 है और OTP 123456 है।", "आपका बकाया one lakh fifty thousand rupees है और O T P 1 2 3 4 5 6 है।"},
		{EnIN, true, "कुल राशि रु. 2.5 लाख है, 15% छूट, 25/12/2024 तक।", "कुल राशि two point five lakh rupees है, fifteen percent छूट, 25th December twenty twenty-four तक।"},
		{EnIN, true, "आपका order 3 दिन में आएगा, total Rs. 499 है, call करें 9876543210 पर।", "आपका order three दिन में आएगा, total four hundred and ninety-nine rupees है, call करें 9 8 7 6 5 4 3 2 1 0 पर।"},
		{EnIN, true, "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।", "कोड four thousand eight hundred and twenty-one दर्ज करें। twenty-one और ninety-nine और one hundred और one thousand and one और seventy-nine और eighty-nine।"},
		{EnIN, true, "Salary between 80000 90000 rupees, range ₹10-12 lakh.", "Salary between eighty thousand ninety thousand rupees, range ten to twelve lakh rupees."},
		{EnIN, true, "Aadhaar 9123 4567 8901 and PAN number ABCDE1234F.", "Aadhaar 9 1 2 3 4 5 6 7 8 9 0 1 and P A N number A B C D E 1 2 3 4 F."},
		{HiIN, false, "Your total is $1,234.56 today.", "Your total is एक हज़ार दो सौ चौंतीस डॉलर और छप्पन सेंट today."},
		{HiIN, false, "That costs $5 or $5.5 or $0.99 or $1.005.", "That costs पाँच डॉलर or पाँच डॉलर और पचास सेंट or निन्यानवे सेंट or एक डॉलर."},
		{HiIN, false, "We raised $5 million and $2.5M and $3k.", "We raised पाँच मिलियन डॉलर and दो दशमलव पाँच मिलियन डॉलर and तीन हज़ार डॉलर."},
		{HiIN, false, "The price dropped by -5% to 12.50% and 0.5%.", "The price dropped by माइनस पाँच प्रतिशत to बारह दशमलव पाँच प्रतिशत and शून्य दशमलव पाँच प्रतिशत."},
		{HiIN, false, "Between 10-12 people at 10:30 AM, or 10:05 PM.", "Between 10-12 people at सुबह साढ़े दस बजे, or रात दस बजकर पाँच मिनट."},
		{HiIN, false, "The temperature is -5 degrees.", "The temperature is -5 degrees."},
		{HiIN, false, "In 2024 and 1999 and 2005 and 1905.", "In 2024 and 1999 and 2005 and 1905."},
		{HiIN, false, "Version v2.5.1 and iOS 17.4.1 released.", "Version v2.5.1 and iOS 17.4.1 released."},
		{HiIN, false, "Server IP is 192.168.1.1 port 8080.", "Server I P is 192.168.1.1 port 8080."},
		{HiIN, false, "Visit https://example.com/path?id=42&x=5km now.", "Visit https://example.com/path?id=42&x=5km now."},
		{HiIN, false, "Call +1 (415) 555-0100 or 415-555-0100 before 5 PM.", "Call एक चार एक पाँच पाँच पाँच पाँच शून्य एक शून्य शून्य or चार एक पाँच पाँच पाँच पाँच शून्य एक शून्य शून्य before शाम पाँच बजे."},
		{HiIN, false, "Call 4155550100 and pay 500.", "Call चार एक पाँच पाँच पाँच पाँच शून्य एक शून्य शून्य and pay 500."},
		{HiIN, false, "You have 3 items, the 1st and 22nd, about 3/4 full.", "You have 3 items, the 1st and 22nd, about 3/4 full."},
		{HiIN, false, "Order number 1234567890123456 shipped.", "Order number एक दो तीन चार पाँच छह सात आठ नौ शून्य एक दो तीन चार पाँच छह shipped."},
		{HiIN, false, "Your OTP is 004216. It expires in 10 minutes.", "Your O T P is शून्य शून्य चार दो एक छह. It expires in 10 minutes."},
		{HiIN, false, "Use promo code SAVE20 for 20% off.", "Use promo code S A V E दो शून्य for बीस प्रतिशत off."},
		{HiIN, false, "The code is valid for 30 days.", "The code is valid for 30 days."},
		{HiIN, false, "Please enter the code. 2024 was a good year.", "Please enter the code. 2024 was a good year."},
		{HiIN, false, "Your PIN code 560001 is serviceable.", "Your P I N code पाँच छह शून्य शून्य शून्य एक is serviceable."},
		{HiIN, false, "Email john.doe+test@example.co.in. Thanks.", "Email john dot doe plus test at example dot co dot in. Thanks."},
		{HiIN, false, "Flight AI 0420 leaves at 0900 hours, agent 007.", "Flight A I 0420 leaves at 0900 hours, agent 007."},
		{HiIN, false, "Plans at 250 500 1000 rupees.", "Plans at 250 500 1000 rupees."},
		{HiIN, false, "Card 4111 1111 1111 1111 expires 12/26.", "Card चार एक एक एक एक एक एक एक एक एक एक एक एक एक एक एक expires 12/26."},
		{HiIN, false, "Ratio is 1,000.50 and 1,50,000 and 12,34,567.", "Ratio is 1,000.50 and 1,50,000 and 12,34,567."},
		{HiIN, false, "Tracking ID: AB-1234/XY 5678 arrives soon.", "Tracking I D: A B एक दो तीन चार X Y पाँच छह सात आठ arrives soon."},
		{HiIN, false, "Account number ending in 4321 has $2500000000.", "Account number ending in चार तीन दो एक has दो सौ पचास करोड़ डॉलर."},
		{HiIN, false, "Your reference number is 12345 dated 12/25/2024.", "Your reference number is एक दो तीन चार पाँच dated 12/25/2024."},
		{HiIN, false, "Booking code 2024-12-25 is confirmed.", "Booking code दो शून्य दो चार एक दो दो पाँच is confirmed."},
		{HiIN, false, "A 3.5% fee, 2x speed, top 10, room 404, 24/7 support.", "A तीन दशमलव पाँच प्रतिशत fee, 2x speed, top 10, room 404, 24/7 support."},
		{HiIN, false, "Rs. 500 and Rs 1,50,000 and ₹2.5 lakh and 500 Rs and Rs.499/- only.", "पाँच सौ रुपये and एक लाख पचास हज़ार रुपये and दो दशमलव पाँच लाख रुपये and पाँच सौ रुपये and चार सौ निन्यानवे रुपये only."},
		{HiIN, false, "INR 2,00,000 or ₹10 cr or ₹5L or Re. 1 or ₹0.50.", "दो लाख रुपये or दस करोड़ रुपये or पाँच लाख रुपये or एक रुपया or पचास पैसे."},
		{HiIN, false, "Others 5 and hours 10 and offers 3.", "Others 5 and hours 10 and offers 3."},
		{HiIN, false, "Call 9876543210 or +91 98765 43210 or 080-2345 6789 or 1800 123 4567.", "Call नौ आठ सात छह पाँच चार तीन दो एक शून्य or नौ एक नौ आठ सात छह पाँच चार तीन दो एक शून्य or शून्य आठ शून्य दो तीन चार पाँच छह सात आठ नौ or एक आठ शून्य शून्य एक दो तीन चार पाँच छह सात."},
		{HiIN, false, "Date 25/12/2024 and 25.12.2024 and 01-02-2024.", "Date पच्चीस दिसंबर दो हज़ार चौबीस and पच्चीस दिसंबर दो हज़ार चौबीस and एक फ़रवरी दो हज़ार चौबीस."},
		{HiIN, false, "आपका बकाया ₹1,50,000 है और OTP 123456 है।", "आपका बकाया एक लाख पचास हज़ार रुपये है और O T P एक दो तीन चार पाँच छह है।"},
		{HiIN, false, "कुल राशि रु. 2.5 लाख है, 15% छूट, 25/12/2024 तक।", "कुल राशि दो दशमलव पाँच लाख रुपये है, पंद्रह प्रतिशत छूट, पच्चीस दिसंबर दो हज़ार चौबीस तक।"},
		{HiIN, false, "आपका order 3 दिन में आएगा, total Rs. 499 है, call करें 9876543210 पर।", "आपका order 3 दिन में आएगा, total चार सौ निन्यानवे रुपये है, call करें नौ आठ सात छह पाँच चार तीन दो एक शून्य पर।"},
		{HiIN, false, "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।", "कोड चार आठ दो एक दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।"},
		{HiIN, false, "Salary between 80000 90000 rupees, range ₹10-12 lakh.", "Salary between 80000 90000 rupees, range दस से बारह लाख रुपये."},
		{HiIN, false, "Aadhaar 9123 4567 8901 and PAN number ABCDE1234F.", "Aadhaar नौ एक दो तीन चार पाँच छह सात आठ नौ शून्य एक and P A N number A B C D E एक दो तीन चार F."},
		{HiIN, true, "Your total is $1,234.56 today.", "Your total is एक हज़ार दो सौ चौंतीस डॉलर और छप्पन सेंट today."},
		{HiIN, true, "That costs $5 or $5.5 or $0.99 or $1.005.", "That costs पाँच डॉलर or पाँच डॉलर और पचास सेंट or निन्यानवे सेंट or एक डॉलर."},
		{HiIN, true, "We raised $5 million and $2.5M and $3k.", "We raised पाँच मिलियन डॉलर and दो दशमलव पाँच मिलियन डॉलर and तीन हज़ार डॉलर."},
		{HiIN, true, "The price dropped by -5% to 12.50% and 0.5%.", "The price dropped by माइनस पाँच प्रतिशत to बारह दशमलव पाँच प्रतिशत and शून्य दशमलव पाँच प्रतिशत."},
		{HiIN, true, "The temperature is -5 degrees.", "The temperature is माइनस पाँच degrees."},
		{HiIN, true, "Version v2.5.1 and iOS 17.4.1 released.", "Version v2.5.1 and iOS 17.4.1 released."},
		{HiIN, true, "Server IP is 192.168.1.1 port 8080.", "Server I P is 192.168.1.1 port आठ हज़ार अस्सी."},
		{HiIN, true, "Visit https://example.com/path?id=42&x=5km now.", "Visit https://example.com/path?id=42&x=5km now."},
		{HiIN, true, "Call +1 (415) 555-0100 or 415-555-0100 before 5 PM.", "Call एक चार एक पाँच पाँच पाँच पाँच शून्य एक शून्य शून्य or चार एक पाँच पाँच पाँच पाँच शून्य एक शून्य शून्य before शाम पाँच बजे."},
		{HiIN, true, "Call 4155550100 and pay 500.", "Call चार एक पाँच पाँच पाँच पाँच शून्य एक शून्य शून्य and pay पाँच सौ."},
		{HiIN, true, "You have 3 items, the 1st and 22nd, about 3/4 full.", "You have तीन items, the 1st and 22nd, about 3/4 full."},
		{HiIN, true, "Order number 1234567890123456 shipped.", "Order number एक दो तीन चार पाँच छह सात आठ नौ शून्य एक दो तीन चार पाँच छह shipped."},
		{HiIN, true, "Your OTP is 004216. It expires in 10 minutes.", "Your O T P is शून्य शून्य चार दो एक छह. It expires in दस minutes."},
		{HiIN, true, "Use promo code SAVE20 for 20% off.", "Use promo code S A V E दो शून्य for बीस प्रतिशत off."},
		{HiIN, true, "The code is valid for 30 days.", "The code is valid for तीस days."},
		{HiIN, true, "Your PIN code 560001 is serviceable.", "Your P I N code पाँच छह शून्य शून्य शून्य एक is serviceable."},
		{HiIN, true, "Email john.doe+test@example.co.in. Thanks.", "Email john dot doe plus test at example dot co dot in. Thanks."},
		{HiIN, true, "Flight AI 0420 leaves at 0900 hours, agent 007.", "Flight A I शून्य चार दो शून्य leaves at शून्य नौ शून्य शून्य hours, agent शून्य शून्य सात."},
		{HiIN, true, "Plans at 250 500 1000 rupees.", "Plans at दो सौ पचास पाँच सौ एक हज़ार rupees."},
		{HiIN, true, "Card 4111 1111 1111 1111 expires 12/26.", "Card चार एक एक एक एक एक एक एक एक एक एक एक एक एक एक एक expires 12/26."},
		{HiIN, true, "Tracking ID: AB-1234/XY 5678 arrives soon.", "Tracking I D: A B एक दो तीन चार X Y पाँच छह सात आठ arrives soon."},
		{HiIN, true, "Account number ending in 4321 has $2500000000.", "Account number ending in चार तीन दो एक has दो सौ पचास करोड़ डॉलर."},
		{HiIN, true, "Your reference number is 12345 dated 12/25/2024.", "Your reference number is एक दो तीन चार पाँच dated 12/25/2024."},
		{HiIN, true, "Booking code 2024-12-25 is confirmed.", "Booking code दो शून्य दो चार एक दो दो पाँच is confirmed."},
		{HiIN, true, "A 3.5% fee, 2x speed, top 10, room 404, 24/7 support.", "A तीन दशमलव पाँच प्रतिशत fee, 2x speed, top दस, room चार सौ चार, 24/7 support."},
		{HiIN, true, "Rs. 500 and Rs 1,50,000 and ₹2.5 lakh and 500 Rs and Rs.499/- only.", "पाँच सौ रुपये and एक लाख पचास हज़ार रुपये and दो दशमलव पाँच लाख रुपये and पाँच सौ रुपये and चार सौ निन्यानवे रुपये only."},
		{HiIN, true, "INR 2,00,000 or ₹10 cr or ₹5L or Re. 1 or ₹0.50.", "दो लाख रुपये or दस करोड़ रुपये or पाँच लाख रुपये or एक रुपया or पचास पैसे."},
		{HiIN, true, "Others 5 and hours 10 and offers 3.", "Others पाँच and hours दस and offers तीन."},
		{HiIN, true, "Date 25/12/2024 and 25.12.2024 and 01-02-2024.", "Date पच्चीस दिसंबर दो हज़ार चौबीस and पच्चीस दिसंबर दो हज़ार चौबीस and एक फ़रवरी दो हज़ार चौबीस."},
		{HiIN, true, "आपका बकाया ₹1,50,000 है और OTP 123456 है।", "आपका बकाया एक लाख पचास हज़ार रुपये है और O T P एक दो तीन चार पाँच छह है।"},
		{HiIN, true, "कुल राशि रु. 2.5 लाख है, 15% छूट, 25/12/2024 तक।", "कुल राशि दो दशमलव पाँच लाख रुपये है, पंद्रह प्रतिशत छूट, पच्चीस दिसंबर दो हज़ार चौबीस तक।"},
		{HiIN, true, "आपका order 3 दिन में आएगा, total Rs. 499 है, call करें 9876543210 पर।", "आपका order तीन दिन में आएगा, total चार सौ निन्यानवे रुपये है, call करें नौ आठ सात छह पाँच चार तीन दो एक शून्य पर।"},
		{HiIN, true, "कोड 4821 दर्ज करें। 21 और 99 और 100 और 1001 और 79 और 89।", "कोड चार आठ दो एक दर्ज करें। इक्कीस और निन्यानवे और एक सौ और एक हज़ार एक और उनासी और नवासी।"},
		{HiIN, true, "Salary between 80000 90000 rupees, range ₹10-12 lakh.", "Salary between अस्सी हज़ार नब्बे हज़ार rupees, range दस से बारह लाख रुपये."},
		{HiIN, true, "Aadhaar 9123 4567 8901 and PAN number ABCDE1234F.", "Aadhaar नौ एक दो तीन चार पाँच छह सात आठ नौ शून्य एक and P A N number A B C D E एक दो तीन चार F."},
	}
	for _, tt := range tests {
		o := DefaultVoiceFormatterOptions(tt.loc)
		o.ExpandNumbers = tt.numbers
		f, err := NewVoiceFormatter(o)
		if err != nil {
			t.Fatal(err)
		}
		got := f.Transform(tt.in)
		if got != tt.want {
			t.Errorf("%s numbers=%v %q:\n got %q\nwant %q", tt.loc.Tag, tt.numbers, tt.in, got, tt.want)
		}
		if again := f.Transform(got); again != got {
			t.Errorf("%s numbers=%v: second pass changed %q to %q", tt.loc.Tag, tt.numbers, got, again)
		}
	}
}

func TestNormalizeTimes(t *testing.T) {
	run(t, "en-US", NormalizeTimes(EnUS), []tcase{
		{"at 10:30 AM", "at ten thirty A M"},
		{"at 10:05 p.m. Then", "at ten oh five P M. Then"},
		{"by 5 PM", "by five P M"},
		{"at 9:00 am", "at nine A M"},
		{"at 14:30", "at two thirty P M"},
		{"at 09:00", "at nine o'clock"},
		{"at 10:30:00", "at 10:30:00"},
		{"ratio 25:61", "ratio 25:61"},
		{"I am 5 amazing", "I am 5 amazing"},
		{"at 13 PM", "at 13 PM"},
	})
	run(t, "hi-IN", NormalizeTimes(HiIN), []tcase{
		{"10:30 बजे पर", "साढ़े दस बजे पर"}, {"सुबह 9:00 पर", "सुबह नौ बजे पर"}, {"16:10 का अनुपात", "16:10 का अनुपात"},
		{"शाम 6:00 तक", "शाम छह बजे तक"},
		{"9:10 बजे", "नौ बजकर दस मिनट"}, {"9:05 बजे", "नौ बजकर पाँच मिनट"},
		{"at 9:15 AM", "at सुबह सवा नौ बजे"}, {"at 9:45 PM", "at रात पौने दस बजे"}, {"at 12:45 PM", "at दोपहर पौने एक बजे"},
		{"at 1:30 PM", "at दोपहर डेढ़ बजे"}, {"at 14:30", "at दोपहर ढाई बजे"}, {"at 00:00", "at रात बारह बजे"},
		{"at 3:00 AM", "at रात तीन बजे"}, {"at 6:00 AM", "at सुबह छह बजे"}, {"at 12 PM", "at दोपहर बारह बजे"},
		{"at 19:20", "at शाम सात बजकर बीस मिनट"}, {"at 20:00", "at रात आठ बजे"}, {"रात 11:30 PM", "रात साढ़े ग्यारह बजे"},
		{"at 09:00", "at नौ बजे"}, {"at 9:30", "at साढ़े नौ बजे"},
	})
}

func TestChainFlushesHeldPrefixWithoutAmount(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultVoiceFormatterOptions(EnIN))
	if err != nil {
		t.Fatal(err)
	}
	steps := []tcase{
		{"Prices in Rs.", "Prices in "},
		{"Thank you.", "rupees. Thank you."},
		{"Total Rs.", "Total "},
		{" 2.5 lakh.", "two point five lakh rupees."},
	}
	for _, s := range steps {
		if got := f.Filter(s.in); got != s.want {
			t.Fatalf("Filter(%q) = %q, want %q", s.in, got, s.want)
		}
	}
	a, b := f.Fork(), f.Fork()
	if got := a.Filter("Your total is Rs."); got != "Your total is " {
		t.Fatalf("fork a = %q", got)
	}
	if got := b.Filter("Hello there."); got != "Hello there." {
		t.Fatalf("fork b got held text of fork a: %q", got)
	}
	if got := a.Filter("500 only."); got != "five hundred rupees only." {
		t.Fatalf("fork a = %q", got)
	}
}

func TestFormatterLeavesURLsAlone(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultVoiceFormatterOptions(EnUS))
	if err != nil {
		t.Fatal(err)
	}
	in := "See https://example.com/a?id=42&x=5km. Then pay $5."
	want := "See https://example.com/a?id=42&x=5km. Then pay five dollars."
	if got := f.Transform(in); got != want {
		t.Fatalf("got %q", got)
	}
}

func TestRoundTwoRegressions(t *testing.T) {
	tests := []struct {
		loc     *Locale
		numbers bool
		in      string
		want    string
	}{
		{EnIN, false, "Please pay ₹500 to 9876543210 on UPI.", "Please pay five hundred rupees to 9 8 7 6 5 4 3 2 1 0 on U P I."},
		{EnIN, false, "Send Rs. 2,000 to 3 vendors today.", "Send two thousand rupees to 3 vendors today."},
		{EnIN, false, "Refund ₹500 to 12 customers.", "Refund five hundred rupees to 12 customers."},
		{EnIN, false, "Price is ₹999 - 10% off today.", "Price is nine hundred and ninety-nine rupees - ten percent off today."},
		{EnIN, false, "Price is ₹999-10% off today.", "Price is nine hundred and ninety-nine rupees-ten percent off today."},
		{EnIN, false, "EMI ₹5,000 to 6,000 per month.", "E M I five thousand rupees to 6,000 per month."},
		{EnUS, false, "Send $20 to 5 friends", "Send twenty dollars to 5 friends"},
		{EnUS, false, "Transfer $50 to 2 accounts; price $100 - 20% discount.", "Transfer fifty dollars to 2 accounts; price one hundred dollars - twenty percent discount."},
		{HiIN, false, "Please pay ₹500 to 9876543210 on UPI.", "Please pay पाँच सौ रुपये to नौ आठ सात छह पाँच चार तीन दो एक शून्य on U P I."},
		{HiIN, false, "Send Rs. 2,000 to 3 vendors today.", "Send दो हज़ार रुपये to 3 vendors today."},
		{EnIN, false, "Budget ₹10-12 lakh.", "Budget ten to twelve lakh rupees."},
		{EnIN, false, "Budget ₹10–12 lakh.", "Budget ten to twelve lakh rupees."},
		{EnIN, false, "Budget ₹10 to ₹12 lakh.", "Budget ten to twelve lakh rupees."},
		{EnIN, false, "Budget ₹10 to 12 lakh.", "Budget ten to twelve lakh rupees."},
		{EnIN, false, "Budget ₹2-3k.", "Budget two to three thousand rupees."},
		{EnUS, false, "Pay $10-12 or $10 to $12.", "Pay ten to twelve dollars or ten to twelve dollars."},
		{EnUS, false, "Budget $5-10 million.", "Budget five to ten million dollars."},
		{HiIN, false, "₹10 से 12 लाख, ₹10-12 लाख", "दस रुपये से 12 लाख, दस से बारह लाख रुपये"},
		{HiIN, false, "कल 9:00 बजे या 10:30 बजे आइए।", "कल नौ बजे या साढ़े दस बजे आइए।"},
		{EnIN, false, "कल 9:00 बजे या 10:30 बजे आइए।", "कल nine बजे या ten thirty आइए।"},
		{EnUS, false, "Come at 9:00 o'clock.", "Come at nine o'clock."},
		{EnUS, false, "Open 09:00 to 18:00, midnight 00:00, from 12:00.", "Open nine o'clock to six P M, midnight twelve A M, from twelve o'clock."},
		{EnUS, false, "Lunch at 13:30.", "Lunch at one thirty P M."},
		{EnUS, false, "Score 21:19, aspect 16:10, John 3:16, ratio 4:30.", "Score 21:19, aspect 16:10, John 3:16, ratio 4:30."},
		{HiIN, false, "शाम 18:00 बजे", "शाम छह बजे"},
		{EnUS, false, "Refund of -$5 or $-5 or -₹500 or (₹500).", "Refund of minus five dollars or minus five dollars or minus five hundred rupees or (five hundred rupees)."},
		{HiIN, false, "Refund -$5", "Refund माइनस पाँच डॉलर"},
		{EnUS, true, "Seat 07 on 05 March, March 09, order 00123.", "Seat 0 7 on five March, March nine, order 0 0 1 2 3."},
		{HiIN, true, "05 March को", "पाँच March को"},
		{EnUS, false, "Ref. 12345 and Order No. 12345 and A/c. 000123456.", "Ref. 1 2 3 4 5 and Order No. 1 2 3 4 5 and A/c. 0 0 0 1 2 3 4 5 6."},
		{EnUS, false, "Please enter the code. 2024 was a good year.", "Please enter the code. 2024 was a good year."},
	}
	for _, tt := range tests {
		o := DefaultVoiceFormatterOptions(tt.loc)
		o.ExpandNumbers = tt.numbers
		f, err := NewVoiceFormatter(o)
		if err != nil {
			t.Fatal(err)
		}
		got := f.Transform(tt.in)
		if got != tt.want {
			t.Errorf("%s numbers=%v %q:\n got %q\nwant %q", tt.loc.Tag, tt.numbers, tt.in, got, tt.want)
		}
		if again := f.Transform(got); again != got {
			t.Errorf("%s numbers=%v: second pass changed %q to %q", tt.loc.Tag, tt.numbers, got, again)
		}
	}
}

func TestOuterChainKeepsFormatterHoldBack(t *testing.T) {
	formatter, err := NewVoiceFormatter(DefaultVoiceFormatterOptions(EnIN))
	if err != nil {
		t.Fatal(err)
	}
	pronounce, _ := Pronunciations(map[string]string{"Siobhan": "ʃəˈvɔːn"}, SSMLPhoneme)
	outer := NewChain(formatter, pronounce)
	if got := outer.Filter("Siobhan, that costs Rs."); got != `<phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>, that costs ` {
		t.Fatalf("first = %q", got)
	}
	if got := outer.Filter("2.5 lakh."); got != "two point five lakh rupees." {
		t.Fatalf("second = %q", got)
	}
	outer.Filter("Total Rs.")
	fork := outer.Fork()
	if got := fork.Filter("500 only."); got != "500 only." {
		t.Fatalf("fork inherited held text: %q", got)
	}
	outer.Reset()
	if got := outer.Filter("500 only."); got != "500 only." {
		t.Fatalf("held text survived Reset: %q", got)
	}
}
