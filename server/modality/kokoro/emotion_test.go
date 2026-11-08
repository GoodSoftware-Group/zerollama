package kokoro

import (
	"errors"
	"testing"
)

func TestApplyEmotion(t *testing.T) {
	got, canon, err := ApplyEmotion(1, " Excited ")
	if err != nil || canon != "excited" || got != 1.2 {
		t.Fatalf("got %v %q %v", got, canon, err)
	}
	got, canon, err = ApplyEmotion(1, "")
	if err != nil || canon != "" || got != 1 {
		t.Fatalf("empty %v %q %v", got, canon, err)
	}
	if _, _, err := ApplyEmotion(1, "joyful"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("unknown %v", err)
	}
	if _, _, err := ApplyEmotion(1.8, "excited"); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("range %v", err)
	}
}
