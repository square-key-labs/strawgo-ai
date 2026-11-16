package transforms

import (
	"regexp"
	"strings"
)

var (
	emailRe     = regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`)
	emailLocal  = strings.NewReplacer(".", " dot ", "_", " underscore ", "-", " dash ", "+", " plus ")
	emailDomain = strings.NewReplacer(".", " dot ", "-", " dash ")
)

var EmailToSpeech Transform = Func(func(text string) string {
	return emailRe.ReplaceAllStringFunc(text, func(email string) string {
		local, domain, _ := strings.Cut(email, "@")
		return emailLocal.Replace(local) + " at " + emailDomain.Replace(domain)
	})
})
