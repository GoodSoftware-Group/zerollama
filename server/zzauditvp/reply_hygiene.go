package voicepipe

import (
	"fmt"
	"strings"
	"unicode"
)

// spokenLeakMarkers are substrings that mean the chat template failed and the
// model is continuing role lines (or raw ChatML) into the spoken reply.
var spokenLeakMarkers = []string{
	"Assistant:",
	"User:",
	"System:",
	"<|im_start|>",
	"<|im_end|>",
	"[BLANK_AUDIO",
}

// SpokenReplyHygiene reports why a duplex or TTS reply is unsafe to play.
// Empty replies and leaked role / template markers fail. Used by smokes and
// realtime_probe -check-hygiene so the ear never ships "Assistant: …" again.
func SpokenReplyHygiene(text string) error {
	t := strings.TrimSpace(text)
	if t == "" {
		return fmt.Errorf("empty reply")
	}
	for _, bad := range spokenLeakMarkers {
		if strings.Contains(t, bad) {
			return fmt.Errorf("reply contains %q", bad)
		}
	}
	// Case-folded role lines at a sentence start ("assistant: yes").
	lower := strings.ToLower(t)
	for _, role := range []string{"assistant:", "user:", "system:"} {
		if strings.HasPrefix(lower, role) || strings.Contains(lower, "\n"+role) {
			return fmt.Errorf("reply contains role label %q", strings.TrimSuffix(role, ":"))
		}
	}
	if !hasSpokenLetter(t) {
		return fmt.Errorf("reply has no spoken letters")
	}
	return nil
}

func hasSpokenLetter(text string) bool {
	for _, r := range text {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// heardWords reports whether a transcript has a letter or a digit.
// A period, an ellipsis, or whitespace is silence. Duplex must not start
// a spoken reply from that.
func heardWords(text string) bool {
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
