package openai

import (
	"encoding/base64"
	"testing"
)

func TestCleanSpeechSample(t *testing.T) {
	k := 0
	if err := CleanSpeechTopK(&k); err != nil {
		t.Fatal(err)
	}
	badK := -1
	if err := CleanSpeechTopK(&badK); err == nil {
		t.Fatal("expected reject")
	}
	p := 1.0
	if err := CleanSpeechTopP(&p); err != nil {
		t.Fatal(err)
	}
	badP := 1.1
	if err := CleanSpeechTopP(&badP); err == nil {
		t.Fatal("expected reject")
	}
	r := 1.05
	if err := CleanSpeechRepeatPenalty(&r); err != nil {
		t.Fatal(err)
	}
	badR := 0.0
	if err := CleanSpeechRepeatPenalty(&badR); err == nil {
		t.Fatal("expected reject")
	}
	if err := CleanSpeechTopK(nil); err != nil || CleanSpeechTopP(nil) != nil || CleanSpeechRepeatPenalty(nil) != nil {
		t.Fatal("nil")
	}
}

func TestCleanSpeechLang(t *testing.T) {
	got, err := CleanSpeechLang(" en-US ")
	if err != nil || got != "en-US" {
		t.Fatalf("%q %v", got, err)
	}
	if _, err := CleanSpeechLang("en US"); err == nil {
		t.Fatal("expected a space to fail")
	}
	if s, err := CleanSpeechLang("  "); err != nil || s != "" {
		t.Fatalf("%q %v", s, err)
	}
}

func TestDecodeSpeakerRef(t *testing.T) {
	raw := []byte("RIFF")
	enc := base64.StdEncoding.EncodeToString(raw)
	got, err := DecodeSpeakerRef("data:audio/wav;base64," + enc)
	if err != nil || string(got) != string(raw) {
		t.Fatalf("%q %v", got, err)
	}
	if b, err := DecodeSpeakerRef(""); err != nil || b != nil {
		t.Fatal(err)
	}
	if _, err := DecodeSpeakerRef("!!!!"); err == nil {
		t.Fatal("expected bad base64")
	}
}
