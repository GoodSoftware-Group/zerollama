package llamarunner

import (
	"testing"

	"github.com/ollama/ollama/llm"
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

func TestIsVisionStub(t *testing.T) {
	if !isVisionStub(input{embedHash: 1}) {
		t.Fatal("expected stub")
	}
	if isVisionStub(input{embedHash: 1, embed: []float32{1}}) {
		t.Fatal("hydrated embed is not a stub")
	}
	if isVisionStub(input{token: 5}) {
		t.Fatal("token is not a stub")
	}
}

func TestInputsMatch_embedHashIgnoresFloats(t *testing.T) {
	stub := input{embedHash: 42}
	real := input{embedHash: 42, embed: []float32{0.1, 0.2}}
	if !inputsMatch(stub, real) {
		t.Fatal("stub should match hydrated embed with same hash")
	}
	if inputsMatch(input{embedHash: 1}, input{embedHash: 2}) {
		t.Fatal("different hashes must not match")
	}
}

func TestCountCommonPrefix_stubHitsHydrated(t *testing.T) {
	cached := []input{
		{token: 1},
		{embedHash: 9, embed: []float32{1, 2, 3}},
		{embedHash: 9, embed: []float32{4, 5, 6}},
		{token: 2},
	}
	prompt := []input{
		{token: 1},
		{embedHash: 9},
		{embedHash: 9},
		{token: 2},
		{token: 3},
	}
	if got := countCommonPrefix(cached, prompt); got != 4 {
		t.Fatalf("count=%d want 4", got)
	}
}
