package voicepipe

import "testing"

func TestSpokenReplyHygieneOK(t *testing.T) {
	if err := SpokenReplyHygiene("Yes, there is a cup nearby."); err != nil {
		t.Fatal(err)
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
