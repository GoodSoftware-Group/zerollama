package kokoro

import "strings"

// nonEnglishPrefix matches Kokoro v1.1 ids (Spanish, French, Hindi, Italian,
// Japanese, Portuguese, Chinese). v1.0 English ids are af_, am_, bf_, and bm_.
var nonEnglishPrefix = map[string]struct{}{
	"ef": {}, "em": {},
	"ff": {}, "fm": {},
	"hf": {}, "hm": {},
	"if": {}, "im": {},
	"jf": {}, "jm": {},
	"pf": {}, "pm": {},
	"zf": {}, "zm": {},
}

// EnglishVoice is false for Kokoro's non-English id prefixes.
// Any other id, including af_/am_/bf_/bm_, is spoken as English.
func EnglishVoice(id string) bool {
	v := strings.ToLower(strings.TrimSpace(id))
	if len(v) < 3 || v[2] != '_' {
		return v != ""
	}
	_, non := nonEnglishPrefix[v[:2]]
	return !non
}

// ResolveLang picks the espeak tag for one utterance.
// An empty lang follows the voice: bf_/bm_ are en-gb, everything else en-us.
// en and en-us are American. en-gb and en-uk are British. An explicit English
// tag overrides the voice prefix. Any other language, or a non-English voice
// id, is a request error.
func ResolveLang(voice, lang string) (string, error) {
	voice = strings.TrimSpace(voice)
	if voice != "" && !EnglishVoice(voice) {
		return "", &BadRequestError{Msg: "kokoro speaks English only, voice " + voice}
	}
	raw := strings.ToLower(strings.TrimSpace(lang))
	raw = strings.ReplaceAll(raw, "_", "-")
	switch raw {
	case "":
		return langForVoice(voice), nil
	case "en", "en-us":
		return "en-us", nil
	case "en-gb", "en-uk":
		return "en-gb", nil
	default:
		return "", &BadRequestError{Msg: "kokoro speaks English only (en-us or en-gb)"}
	}
}

func langForVoice(voice string) string {
	v := strings.ToLower(strings.TrimSpace(voice))
	if strings.HasPrefix(v, "bf_") || strings.HasPrefix(v, "bm_") {
		return "en-gb"
	}
	return "en-us"
}
