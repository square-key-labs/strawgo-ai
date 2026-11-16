package transforms

import "regexp"

type Replacement struct {
	Pattern string
	Replace string
}

func ReplaceText(rules ...Replacement) (Transform, error) {
	type compiled struct {
		re   *regexp.Regexp
		with string
	}
	list := make([]compiled, 0, len(rules))
	for _, r := range rules {
		re, err := regexp.Compile(r.Pattern)
		if err != nil {
			return nil, err
		}
		list = append(list, compiled{re, r.Replace})
	}
	return Func(func(text string) string {
		for _, c := range list {
			text = c.re.ReplaceAllString(text, c.with)
		}
		return text
	}), nil
}
