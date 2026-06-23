package ollamarunner

import "testing"

func TestMistral3ImageInjectStart(t *testing.T) {
	slots := Mistral3VisionTokens{Img: 10, Break: 12, End: 13}
	block := []int{10, 10, 12, 10, 13}
	if !isMistral3ImageInjectStart(block, 0, slots) {
		t.Fatal("expected image block start at first IMG")
	}
	if isMistral3ImageInjectStart(block, 1, slots) {
		t.Fatal("mid-row IMG should not start inject")
	}
	if isMistral3ImageInjectStart(block, 3, slots) {
		t.Fatal("IMG after BREAK is same image")
	}
}

func TestMistral3ImageBlockEndIndex(t *testing.T) {
	slots := Mistral3VisionTokens{Img: 10, Break: 12, End: 13}
	block := []int{10, 10, 12, 10, 13, 42}
	if got := mistral3ImageBlockEndIndex(block, 0, slots); got != 4 {
		t.Fatalf("end=%d want 4", got)
	}
}
