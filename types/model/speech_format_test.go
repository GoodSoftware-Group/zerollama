package model

import "testing"

func TestSpeechContainer(t *testing.T) {
	got, err := SpeechContainer(" PCM ", true)
	if err != nil || got != "pcm" {
		t.Fatal(got, err)
	}
	got, err = SpeechContainer("", true)
	if err != nil || got != "wav" {
		t.Fatal(got, err)
	}
	if _, err := SpeechContainer("mp3", true); err == nil {
		t.Fatal("expected reject")
	}
	if _, err := SpeechContainer("pcm", false); err == nil {
		t.Fatal("expected piper reject")
	}
	got, err = SpeechContainer("WAV", false)
	if err != nil || got != "wav" {
		t.Fatal(got, err)
	}
	if got := PCMContentType("", 24000); got != "audio/pcm;rate=24000;encoding=float;bits=32" {
		t.Fatal(got)
	}
	kept := "audio/pcm;rate=16000;encoding=float;bits=32"
	if PCMContentType(kept, 24000) != kept {
		t.Fatal("kept")
	}
	if PCMContentType("audio/wav", 24000) != "audio/wav" {
		t.Fatal("wav")
	}
	if PCMContentType("audio/pcm;encoding=float;bits=32", 32000) != "audio/pcm;encoding=float;bits=32;rate=32000" {
		t.Fatal("rate only")
	}
	if PCMContentType("audio/pcm", 0) != "audio/pcm" {
		t.Fatal("unknown rate")
	}
}
