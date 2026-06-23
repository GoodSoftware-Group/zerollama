package ollamarunner

import (
	"testing"

	"github.com/ollama/ollama/model/input"
)

func TestVisionTokensFromGridTHW_qwenStyle(t *testing.T) {
	got := visionTokensFromGridTHW([]int{1, 24, 32}, 2)
	if got != 192 {
		t.Fatalf("got %d want 192", got)
	}
}

func TestOllamaVisionEmbedTokenCount_sameBatch(t *testing.T) {
	got := ollamaVisionEmbedTokenCount(&input.Input{SameBatch: 192, Multimodal: []input.Multimodal{{}}})
	if got != 192 {
		t.Fatalf("got %d want 192", got)
	}
}

func TestVisionEmbedCountsFromInputs(t *testing.T) {
	inputs := []*input.Input{
		{Token: 1},
		{SameBatch: 64, Multimodal: []input.Multimodal{{}}},
		{Token: 2},
		{SameBatch: 32, Multimodal: []input.Multimodal{{}}},
	}
	got := visionEmbedCountsFromInputs(inputs)
	if len(got) != 2 || got[0] != 64 || got[1] != 32 {
		t.Fatalf("got %v", got)
	}
}

func TestLogVisionGridHint_noPanic(t *testing.T) {
	logVisionGridHint(0, []int{1, 24, 32}, 192)
	logVisionGridHint(1, nil, 0)
}
