package qwen3asr

import "testing"

func TestLanguageName(t *testing.T) {
	cases := map[string]string{
		"":          "",
		"auto":      "",
		"en":        "English",
		"EN":        "English",
		"en-US":     "English",
		"english":   "English",
		"zh":        "Chinese",
		"yue":       "Cantonese",
		"tl":        "Filipino",
		"zz":        "",
		"ht":        "",
		"fr-CA":     "French",
	}
	for in, want := range cases {
		if got := LanguageName(in); got != want {
			t.Fatalf("%q: got %q want %q", in, got, want)
		}
	}
}
