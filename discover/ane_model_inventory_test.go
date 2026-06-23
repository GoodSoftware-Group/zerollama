package discover

import "testing"

func TestSelectANEModelPreferred(t *testing.T) {
	entries := []ANEModelEntry{
		{Tag: "qwen3.6:latest", Name: "registry.ollama.ai/library/qwen3.6:latest", EmbeddingLength: 4096},
		{Tag: "eliza-1-2b:latest", Name: "registry.ollama.ai/library/eliza-1-2b:latest", EmbeddingLength: 2048},
	}
	got, ok := SelectANEModel(entries, "qwen")
	if !ok || got.Tag != "qwen3.6:latest" {
		t.Fatalf("SelectANEModel = %+v ok=%v", got, ok)
	}
}

func TestSelectANEModelMissingPreferred(t *testing.T) {
	entries := []ANEModelEntry{{Tag: "a:latest"}, {Tag: "b:latest"}}
	_, ok := SelectANEModel(entries, "missing")
	if ok {
		t.Fatal("expected no match")
	}
}
