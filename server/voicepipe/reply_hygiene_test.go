package voicepipe

import "testing"

func TestSpokenReplyHygieneOK(t *testing.T) {
	if err := SpokenReplyHygiene("Yes, there is a cup nearby."); err != nil {
		t.Fatal(err)
	}
}

func TestCommitSupersedesGuess(t *testing.T) {
	if !commitSupersedesGuess(507, 330) || !commitSupersedesGuess(901, 330) {
		t.Fatal("expected a longer commit to win")
	}
	if commitSupersedesGuess(256, 256) || commitSupersedesGuess(400, 330) || commitSupersedesGuess(0, 100) {
		t.Fatal("quiet tail or empty commit should keep the guess")
	}
}

func TestHeardWords(t *testing.T) {
	if !heardWords("Hello.") || !heardWords("  42 ") {
		t.Fatal("expected words")
	}
	for _, c := range []string{"", "   ", ".", "...", "!", "？"} {
		if heardWords(c) {
			t.Fatalf("expected silence for %q", c)
		}
	}
}

func TestSpokenReplyHygieneRejectsRoleLeaks(t *testing.T) {
	cases := []string{
		"",
		"   ",
		"Assistant: Yes, I can see a keyboard.",
		"Can you see anything else? Assistant: Yes.",
		"User: [BLANK_AUDIO Is there anything else",
		"System:\nA bookshelf is nearby.",
		"<|im_start|>assistant\nHello",
		"…",
		"12345",
	}
	for _, c := range cases {
		if err := SpokenReplyHygiene(c); err == nil {
			t.Fatalf("expected fail for %q", c)
		}
	}
}
