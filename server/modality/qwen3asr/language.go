package qwen3asr

import "strings"

// LanguageName maps a whisper-style code or English name to the canonical
// name qwen-asr expects (English, Chinese, …). Empty or auto means detect.
// A regional tag uses the part before the hyphen. An unmapped code is empty
// so the model detects the language.
func LanguageName(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.IndexAny(s, "-_"); i > 0 {
		s = s[:i]
	}
	s = strings.ToLower(s)
	if s == "auto" {
		return ""
	}
	if name, ok := languageNames[s]; ok {
		return name
	}
	return ""
}

// languageNames covers the codes and English names Whisper already accepts
// that also appear in qwen-asr's SUPPORTED_LANGUAGES.
var languageNames = map[string]string{
	"en": "English", "english": "English",
	"zh": "Chinese", "chinese": "Chinese",
	"yue": "Cantonese", "cantonese": "Cantonese",
	"ar": "Arabic", "arabic": "Arabic",
	"de": "German", "german": "German",
	"fr": "French", "french": "French",
	"es": "Spanish", "spanish": "Spanish",
	"pt": "Portuguese", "portuguese": "Portuguese",
	"id": "Indonesian", "indonesian": "Indonesian",
	"it": "Italian", "italian": "Italian",
	"ko": "Korean", "korean": "Korean",
	"ru": "Russian", "russian": "Russian",
	"th": "Thai", "thai": "Thai",
	"vi": "Vietnamese", "vietnamese": "Vietnamese",
	"ja": "Japanese", "japanese": "Japanese",
	"tr": "Turkish", "turkish": "Turkish",
	"hi": "Hindi", "hindi": "Hindi",
	"ms": "Malay", "malay": "Malay",
	"nl": "Dutch", "dutch": "Dutch",
	"sv": "Swedish", "swedish": "Swedish",
	"da": "Danish", "danish": "Danish",
	"fi": "Finnish", "finnish": "Finnish",
	"pl": "Polish", "polish": "Polish",
	"cs": "Czech", "czech": "Czech",
	"tl": "Filipino", "tagalog": "Filipino", "filipino": "Filipino", "fil": "Filipino",
	"fa": "Persian", "persian": "Persian",
	"el": "Greek", "greek": "Greek",
	"ro": "Romanian", "romanian": "Romanian",
	"hu": "Hungarian", "hungarian": "Hungarian",
	"mk": "Macedonian", "macedonian": "Macedonian",
}
