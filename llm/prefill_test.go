package llm

import "testing"

func TestPrefillBeforeLastWord(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"Summ", ""},
		{"Summarize this.\n\nWhat is", "Summarize this.\n\nWhat"},
		{"two words  ", "two"},
		{"zażółć gęślą", "zażółć"},
	}
	for _, tc := range cases {
		if got := PrefillBeforeLastWord(tc.in); got != tc.want {
			t.Fatalf("PrefillBeforeLastWord(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestLlamaServerCachePromptHonorsReset(t *testing.T) {
	if !llamaServerCachePrompt(false) {
		t.Fatal("a normal request keeps the prompt cache")
	}
	if llamaServerCachePrompt(true) {
		t.Fatal("cache_reset must miss the prompt cache")
	}
}

func TestPrefillSentinelIsPrivateUse(t *testing.T) {
	if PrefillSentinel != "\uF8FF" {
		t.Fatalf("sentinel %q", PrefillSentinel)
	}
}
