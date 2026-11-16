package transforms

import (
	"regexp"
	"strings"
)

type DateOrder int

const (
	MonthDayYear DateOrder = iota
	DayMonthYear
)

type Currency struct {
	Symbols       []string
	SuffixSymbols []string
	Singular      string
	Plural        string
	MinorSingular string
	MinorPlural   string
}

type PhonePattern struct {
	Pattern   *regexp.Regexp
	MinDigits int
	MaxDigits int
}

type SplitPrefix struct {
	Text   string
	Spoken string
}

type Unit struct {
	Singular string
	Plural   string
}

type DayPeriod struct {
	From int
	Word string
}

type Locale struct {
	Tag string

	Below100       [100]string
	Hundred        string
	HundredJoin    string
	IndianGrouping bool
	Scales         []string
	GroupJoin      string
	FinalJoin      string
	Point          string
	DigitWords     bool
	Minus          string
	RangeJoin      string

	Percent            string
	Currencies         []Currency
	MinorJoin          string
	OmitZeroMajor      bool
	ScaleWords         map[string]string
	ScaleAbbreviations map[string]string
	AmountSuffixes     []string
	SplitPrefixes      []SplitPrefix

	DateOrder      DateOrder
	DateSeparators string
	DateLayout     string
	Months         [12]string
	OrdinalDays    bool
	EnglishYears   bool
	HundredYears   bool

	TimeLayout   string
	TimeOnHour   string
	TimeMinuteOh string
	TimeWords    []string
	TimeCues     []string
	AM           string
	PM           string
	DayPeriods   []DayPeriod
	TimeQuarters map[int]string
	TimeHalves   map[int]string

	Units map[string]Unit

	PhonePatterns []PhonePattern
	CodeKeywords  []string
	CodeFillers   []string
}

func orDefault(l *Locale) *Locale {
	if l == nil {
		return EnUS
	}
	return l
}

func LocaleFor(tag string) (*Locale, bool) {
	l, ok := locales[strings.ToLower(strings.ReplaceAll(tag, "_", "-"))]
	return l, ok
}

func englishBelow100() [100]string {
	ones := [20]string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
		"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	tens := [10]string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
	var w [100]string
	for i := range w {
		switch {
		case i < 20:
			w[i] = ones[i]
		case i%10 == 0:
			w[i] = tens[i/10]
		default:
			w[i] = tens[i/10] + "-" + ones[i%10]
		}
	}
	return w
}

var hindiBelow100 = [100]string{
	"शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
	"दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
	"बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
	"तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
	"चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
	"पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
	"साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
	"सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
	"अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सतासी", "अठासी", "नवासी",
	"नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
}

var englishMonths = [12]string{"January", "February", "March", "April", "May", "June",
	"July", "August", "September", "October", "November", "December"}

var englishUnits = map[string]Unit{
	"km":  {"kilometer", "kilometers"},
	"m":   {"meter", "meters"},
	"cm":  {"centimeter", "centimeters"},
	"mm":  {"millimeter", "millimeters"},
	"mi":  {"mile", "miles"},
	"ft":  {"foot", "feet"},
	"in":  {"inch", "inches"},
	"yd":  {"yard", "yards"},
	"kg":  {"kilogram", "kilograms"},
	"g":   {"gram", "grams"},
	"mg":  {"milligram", "milligrams"},
	"lb":  {"pound", "pounds"},
	"oz":  {"ounce", "ounces"},
	"l":   {"liter", "liters"},
	"ml":  {"milliliter", "milliliters"},
	"mph": {"mile per hour", "miles per hour"},
	"kph": {"kilometer per hour", "kilometers per hour"},
	"kmh": {"kilometer per hour", "kilometers per hour"},
	"gb":  {"gigabyte", "gigabytes"},
	"mb":  {"megabyte", "megabytes"},
	"kb":  {"kilobyte", "kilobytes"},
	"tb":  {"terabyte", "terabytes"},
	"hz":  {"hertz", "hertz"},
	"khz": {"kilohertz", "kilohertz"},
	"mhz": {"megahertz", "megahertz"},
	"ghz": {"gigahertz", "gigahertz"},
}

var englishCurrencies = []Currency{
	{Symbols: []string{"$"}, Singular: "dollar", Plural: "dollars", MinorSingular: "cent", MinorPlural: "cents"},
	{Symbols: []string{"€"}, Singular: "euro", Plural: "euros", MinorSingular: "cent", MinorPlural: "cents"},
	{Symbols: []string{"£"}, Singular: "pound", Plural: "pounds", MinorSingular: "penny", MinorPlural: "pence"},
	{Symbols: []string{"¥"}, Singular: "yen", Plural: "yen"},
	{Symbols: rupeeSymbols, SuffixSymbols: rupeeSuffixes, Singular: "rupee", Plural: "rupees", MinorSingular: "paisa", MinorPlural: "paise"},
}

var rupeeSymbols = []string{"₹", "Rs.", "Rs", "Re.", "INR", "रु.", "रु"}
var rupeeSuffixes = []string{"Rs", "INR"}

var englishScaleWords = map[string]string{
	"thousand": "thousand", "million": "million", "billion": "billion", "trillion": "trillion",
	"lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh", "crore": "crore", "crores": "crore",
	"हज\u093cार": "thousand", "हजार": "thousand", "लाख": "lakh", "करोड\u093c": "crore", "करोड": "crore",
}

var englishScaleAbbreviations = map[string]string{
	"k": "thousand", "l": "lakh", "cr": "crore", "m": "million", "mn": "million", "b": "billion", "bn": "billion", "t": "trillion", "tn": "trillion",
}

var englishTimeCues = []string{"at", "by", "from", "to", "until", "till", "before", "after", "around", "since", "between", "opens", "closes"}

var englishSplitPrefixes = []SplitPrefix{{"Rs.", "rupees."}, {"Re.", "rupee."}, {"रु.", "rupees."}}

var groupedDigits = PhonePattern{
	Pattern:   regexp.MustCompile(`\d{4}(?:[ \-]\d{4}){2,3}`),
	MinDigits: 12,
	MaxDigits: 16,
}

var usPhone = PhonePattern{
	Pattern:   regexp.MustCompile(`(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?)(\d{4})`),
	MinDigits: 10,
	MaxDigits: 11,
}

var indianPhones = []PhonePattern{
	{
		Pattern:   regexp.MustCompile(`(?:\+?91[\s\-]?|0)?(?:[6-9]\d{4}[\s\-]?\d{5}|[6-9]\d{2}[\s\-]?\d{3}[\s\-]?\d{4})`),
		MinDigits: 10,
		MaxDigits: 12,
	},
	{
		Pattern:   regexp.MustCompile(`(?:\+?91[\s\-]?|0)[1-9]\d{1,3}[\s\-]?\d{3,4}[\s\-]?\d{3,4}`),
		MinDigits: 11,
		MaxDigits: 12,
	},
	{
		Pattern:   regexp.MustCompile(`1[89]00[\s\-]?\d{3}[\s\-]?\d{3,4}`),
		MinDigits: 10,
		MaxDigits: 11,
	},
}

var englishCodeKeywords = []string{
	`otp`, `one[- ]time (?:password|passcode|pin)`, `passcode`, `pass code`, `pin`, `pin ?code`, `zip ?code`, `postal code`,
	`(?:verification|security|confirmation|access|activation|authori[sz]ation|auth|promo|coupon|voucher|referral|discount|booking|reference|tracking|customer) code`,
	`code`, `cvv`, `mpin`, `upi pin`, `ifsc`, `pnr`, `utr`, `awb`, `gstin`,
	`(?:account|acct|a/c|order|booking|reservation|ticket|tracking|shipment|transaction|txn|reference|ref|invoice|policy|claim|case|customer|consumer|member|registration|application|complaint|request|loan|card|folio|pan|aadhaar|aadhar|employee|vehicle|meter|service|incident) (?:id|number|no\.?|num\.?|#)`,
	`a/c\.?`, `acct\.?`, `ref\.?`,
}

var englishCodeFillers = []string{`is`, `are`, `was`, `will be`, `number`, `no`, `num`, `id`, `ending in`, `ending with`}

var hindiCodeKeywords = append([]string{
	`ओटीपी`, `कोड`, `पिन`, `पिन कोड`, `खाता संख्या`, `खाता नंबर`, `ऑर्डर आईडी`, `ऑर्डर नंबर`, `बुकिंग आईडी`, `टिकट नंबर`, `संदर्भ संख्या`,
}, englishCodeKeywords...)

var EnUS = &Locale{
	Tag:                "en-US",
	Below100:           englishBelow100(),
	Hundred:            "hundred",
	HundredJoin:        " and ",
	Scales:             []string{"thousand", "million", "billion", "trillion", "quadrillion", "quintillion"},
	GroupJoin:          ", ",
	FinalJoin:          " and ",
	Point:              "point",
	Minus:              "minus",
	RangeJoin:          " to ",
	Percent:            "percent",
	Currencies:         englishCurrencies,
	MinorJoin:          " and ",
	ScaleWords:         englishScaleWords,
	ScaleAbbreviations: englishScaleAbbreviations,
	AmountSuffixes:     []string{"/-"},
	SplitPrefixes:      englishSplitPrefixes,
	DateOrder:          MonthDayYear,
	DateSeparators:     "/-",
	DateLayout:         "{month} {day}, {year}",
	Months:             englishMonths,
	OrdinalDays:        true,
	EnglishYears:       true,
	TimeLayout:         "{h} {m}",
	TimeOnHour:         "{h} o'clock",
	TimeWords:          []string{"o'clock", "बजे"},
	TimeCues:           englishTimeCues,
	TimeMinuteOh:       "oh ",
	AM:                 "A M",
	PM:                 "P M",
	Units:              englishUnits,
	PhonePatterns:      []PhonePattern{usPhone, groupedDigits},
	CodeKeywords:       englishCodeKeywords,
	CodeFillers:        englishCodeFillers,
}

var EnIN = &Locale{
	Tag:                "en-IN",
	Below100:           englishBelow100(),
	Hundred:            "hundred",
	HundredJoin:        " and ",
	IndianGrouping:     true,
	Scales:             []string{"thousand", "lakh", "crore"},
	GroupJoin:          " ",
	FinalJoin:          " and ",
	Point:              "point",
	Minus:              "minus",
	RangeJoin:          " to ",
	Percent:            "percent",
	Currencies:         englishCurrencies,
	MinorJoin:          " and ",
	ScaleWords:         englishScaleWords,
	ScaleAbbreviations: englishScaleAbbreviations,
	AmountSuffixes:     []string{"/-"},
	SplitPrefixes:      englishSplitPrefixes,
	DateOrder:          DayMonthYear,
	DateSeparators:     "/-.",
	DateLayout:         "{day} {month} {year}",
	Months:             englishMonths,
	OrdinalDays:        true,
	EnglishYears:       true,
	TimeLayout:         "{h} {m}",
	TimeOnHour:         "{h} o'clock",
	TimeWords:          []string{"o'clock", "बजे"},
	TimeCues:           englishTimeCues,
	TimeMinuteOh:       "oh ",
	AM:                 "A M",
	PM:                 "P M",
	Units:              englishUnits,
	PhonePatterns:      append(append([]PhonePattern{}, indianPhones...), usPhone, groupedDigits),
	CodeKeywords:       englishCodeKeywords,
	CodeFillers:        englishCodeFillers,
}

var HiIN = &Locale{
	Tag:            "hi-IN",
	Below100:       hindiBelow100,
	Hundred:        "सौ",
	HundredJoin:    " ",
	IndianGrouping: true,
	Scales:         []string{"हज़ार", "लाख", "करोड़"},
	GroupJoin:      " ",
	FinalJoin:      " ",
	Point:          "दशमलव",
	DigitWords:     true,
	Minus:          "माइनस",
	RangeJoin:      " से ",
	Percent:        "प्रतिशत",
	Currencies: []Currency{
		{Symbols: rupeeSymbols, SuffixSymbols: rupeeSuffixes, Singular: "रुपया", Plural: "रुपये", MinorSingular: "पैसा", MinorPlural: "पैसे"},
		{Symbols: []string{"$"}, Singular: "डॉलर", Plural: "डॉलर", MinorSingular: "सेंट", MinorPlural: "सेंट"},
		{Symbols: []string{"€"}, Singular: "यूरो", Plural: "यूरो", MinorSingular: "सेंट", MinorPlural: "सेंट"},
		{Symbols: []string{"£"}, Singular: "पाउंड", Plural: "पाउंड", MinorSingular: "पेंस", MinorPlural: "पेंस"},
		{Symbols: []string{"¥"}, Singular: "येन", Plural: "येन"},
	},
	MinorJoin:     " और ",
	OmitZeroMajor: true,
	ScaleWords: map[string]string{
		"thousand": "हज़ार", "lakh": "लाख", "lakhs": "लाख", "lac": "लाख", "lacs": "लाख", "crore": "करोड़", "crores": "करोड़",
		"million": "मिलियन", "billion": "बिलियन", "trillion": "ट्रिलियन",
		"हज़ार": "हज़ार", "हजार": "हज़ार", "लाख": "लाख", "करोड़": "करोड़", "करोड": "करोड़",
	},
	ScaleAbbreviations: map[string]string{
		"k": "हज़ार", "l": "लाख", "cr": "करोड़", "m": "मिलियन", "mn": "मिलियन", "b": "बिलियन", "bn": "बिलियन", "t": "ट्रिलियन", "tn": "ट्रिलियन",
	},
	AmountSuffixes: []string{"/-"},
	SplitPrefixes:  []SplitPrefix{{"Rs.", "रुपये।"}, {"Re.", "रुपया।"}, {"रु.", "रुपये।"}},
	DateOrder:      DayMonthYear,
	DateSeparators: "/-.",
	DateLayout:     "{day} {month} {year}",
	Months:         [12]string{"जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"},
	HundredYears:   true,
	TimeLayout:     "{h} बजकर {m} मिनट",
	TimeOnHour:     "{h} बजे",
	TimeWords:      []string{"बजे"},
	TimeCues:       []string{"सुबह", "शाम", "रात", "दोपहर", "at", "by", "from", "to", "until", "till", "before", "after", "around"},
	DayPeriods:     []DayPeriod{{0, "रात"}, {4, "सुबह"}, {12, "दोपहर"}, {16, "शाम"}, {20, "रात"}},
	TimeQuarters:   map[int]string{15: "सवा {h} बजे", 30: "साढ़े {h} बजे", 45: "पौने {next} बजे"},
	TimeHalves:     map[int]string{1: "डेढ़ बजे", 2: "ढाई बजे"},
	PhonePatterns:  append(append([]PhonePattern{}, indianPhones...), usPhone, groupedDigits),
	CodeKeywords:   hindiCodeKeywords,
	CodeFillers:    englishCodeFillers,
}

var locales = map[string]*Locale{
	"en":    EnUS,
	"en-us": EnUS,
	"en-in": EnIN,
	"hi":    HiIN,
	"hi-in": HiIN,
}
