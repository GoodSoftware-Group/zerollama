package ollamarunner

import "testing"

func TestMllamaImageSlotToken(t *testing.T) {
	if got := mllamaImageSlotToken(); got != 128256 {
		t.Fatalf("got %d want 128256", got)
	}
}

func TestGemma3ImageSlotToken(t *testing.T) {
	if got := gemma3ImageSlotToken(); got != 255999 {
		t.Fatalf("got %d want 255999", got)
	}
}
