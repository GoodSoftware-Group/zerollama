package ollamarunner

import (
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/input"
)

func TestEstimateVisionTokenSpan(t *testing.T) {
	n := estimateVisionTokenSpan(llm.ImageData{GridTHW: []int{1, 24, 32}})
	if n <= 0 {
		t.Fatalf("expected positive token span, got %d", n)
	}
	if estimateVisionTokenSpan(llm.ImageData{}) != 0 {
		t.Fatal("missing grid should not defer")
	}
}

func TestIsDeferVisionStub(t *testing.T) {
	if !isDeferVisionStub([]input.Multimodal{{Data: deferVisionStubMarker}}) {
		t.Fatal("expected stub marker")
	}
	if isDeferVisionStub(nil) || isDeferVisionStub([]input.Multimodal{{Data: "other"}}) {
		t.Fatal("unexpected stub match")
	}
}
