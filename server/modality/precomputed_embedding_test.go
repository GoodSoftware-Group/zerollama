package modality

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMessageUnmarshal_precomputedInImages(t *testing.T) {
	raw := `{"role":"user","padded_input_ids":[1,2,3],"images":[{"format":"precomputed_embedding","feature":[[0.1,0.2],[0.3,0.4]]}]}`
	var msg api.Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.PrecomputedEmbeddings) != 1 || len(msg.Images) != 0 {
		t.Fatalf("precomputed=%d images=%d", len(msg.PrecomputedEmbeddings), len(msg.Images))
	}
	if len(msg.PrecomputedEmbeddings[0].Feature) != 2 {
		t.Fatalf("feature rows=%d", len(msg.PrecomputedEmbeddings[0].Feature))
	}
}

func TestPreflightPrecomputedEmbeddings_requiresPaddedIDs(t *testing.T) {
	req := &api.ChatRequest{Messages: []api.Message{{
		PrecomputedEmbeddings: []api.PrecomputedEmbedding{{Feature: [][]float32{{1}}}},
	}}}
	if err := PreflightPrecomputedEmbeddings(req); err == nil {
		t.Fatal("expected error without padded_input_ids")
	}
}

func TestPreflightPrecomputedEmbeddings_ok(t *testing.T) {
	req := &api.ChatRequest{Messages: []api.Message{{
		PaddedInputIDs:        []int{1, 2},
		PrecomputedEmbeddings: []api.PrecomputedEmbedding{{Format: "precomputed_embedding", Feature: [][]float32{{1, 2}, {3, 4}}}},
	}}}
	if err := PreflightPrecomputedEmbeddings(req); err != nil {
		t.Fatal(err)
	}
}

func TestAppendPrecomputedImagesToLLM(t *testing.T) {
	msg := api.Message{
		PrecomputedEmbeddings: []api.PrecomputedEmbedding{{Feature: [][]float32{{1, 2}}}},
	}
	out := AppendPrecomputedImagesToLLM(msg, nil)
	if len(out) != 1 || !out[0].HasPrecomputedEmbedding() {
		t.Fatalf("got %+v", out)
	}
}
