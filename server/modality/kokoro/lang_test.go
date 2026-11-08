package kokoro

import (
	"errors"
	"testing"
)

func TestResolveLang(t *testing.T) {
	cases := []struct {
		voice, lang, want string
	}{
		{"af_heart", "", "en-us"},
		{"am_adam", "", "en-us"},
		{"bf_emma", "", "en-gb"},
		{"BM_Lewis", "", "en-gb"},
		{"af_heart", "en-GB", "en-gb"},
		{"bf_emma", "en", "en-us"},
		{"af_heart", "en_uk", "en-gb"},
		{"", "", "en-us"},
		{"nope", "", "en-us"},
	}
	for _, tc := range cases {
		got, err := ResolveLang(tc.voice, tc.lang)
		if err != nil || got != tc.want {
			t.Fatalf("ResolveLang(%q, %q)=%q %v, want %q", tc.voice, tc.lang, got, err, tc.want)
		}
	}
	for _, voice := range []string{"zf_xiaobei", "jf_alpha", "ef_dora", "ff_siwis", "hf_alpha", "if_sara", "pf_dora", "zm_yunxi"} {
		if EnglishVoice(voice) {
			t.Fatalf("%s should not be English", voice)
		}
		if _, err := ResolveLang(voice, ""); !errors.Is(err, ErrBadRequest) {
			t.Fatalf("%s: %v", voice, err)
		}
	}
	if _, err := ResolveLang("af_heart", "zh"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("zh: %v", err)
	}
	if _, err := ResolveLang("af_heart", "ja"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("ja: %v", err)
	}
}
