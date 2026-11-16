package user_start

func EnglishBackchannelWords() []string {
	return []string{
		"mm", "hmm", "hm", "mhm", "mm hmm", "uh huh", "uh hum", "aha", "ah", "oh",
		"okay", "ok", "okey", "okie", "kay",
		"right", "alright", "all right",
		"yeah", "yea", "yah", "ya", "yep", "yup", "yes",
		"sure", "got it", "gotcha", "i see", "i got it", "cool",
	}
}

func HindiBackchannelWords() []string {
	return []string{
		"haan", "han", "haa", "ha", "haanji", "hanji", "haan ji", "han ji", "ha ji",
		"ji", "jee", "ji haan", "ji han", "ji ha",
		"achha", "accha", "acha", "achcha", "achchha", "achha ji", "acha ji",
		"theek hai", "thik hai", "theek he", "thik he", "theek h", "thik h", "theek", "thik", "theek hai ji", "thik hai ji",
		"sahi hai", "bilkul", "hmm", "hm", "ok ji", "okay ji",
		"हाँ", "हां", "हा", "हाँजी", "हांजी", "हाँ जी", "हां जी", "जी", "जी हाँ", "जी हां",
		"अच्छा", "अच्छा जी", "ठीक", "ठीक है", "ठीक है जी",
		"सही है", "बिल्कुल", "हम्म", "ओके", "ओके जी",
	}
}

func DefaultBackchannelWords() []string {
	return append(EnglishBackchannelWords(), HindiBackchannelWords()...)
}
