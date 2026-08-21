package server

import (
	"context"
	"testing"
)

func TestCosineSim(t *testing.T) {
	if cosineSim([]float32{1, 0}, []float32{1, 0}) < 0.999 {
		t.Fatal("identical")
	}
	if cosineSim([]float32{1, 0}, []float32{0, 1}) > 0.01 {
		t.Fatal("orthogonal")
	}
}

func TestKNNVoteUndecidable(t *testing.T) {
	active, _, best := knnVote([]RouterNeighbor{
		{ID: "a", Labels: []string{"code"}, Similarity: 0.2},
	}, 0.80, 0.5)
	if len(active) != 0 || best != 0.2 {
		t.Fatalf("active=%v best=%v", active, best)
	}
}

func TestKNNVoteMajority(t *testing.T) {
	active, shares, _ := knnVote([]RouterNeighbor{
		{ID: "a", Labels: []string{"code"}, Similarity: 0.95},
		{ID: "b", Labels: []string{"code"}, Similarity: 0.90},
		{ID: "c", Labels: []string{"general"}, Similarity: 0.85},
	}, 0.80, 0.5)
	if len(active) != 1 || active[0] != "code" {
		t.Fatalf("active=%v shares=%v", active, shares)
	}
}

func TestDecideRouterKNN(t *testing.T) {
	spec := RouterSpec{
		Classifier: "knn",
		Embedder:   "embed",
		Fallback:   "llama3.2:3b",
		Candidates: []RouterCandidate{
			{Model: "coder", Labels: []string{"code"}},
			{Model: "chat", Labels: []string{"general"}},
		},
		KNN: &RouterKNN{
			K:                   2,
			SimilarityThreshold: 0.5,
			VoteThreshold:       0.5,
			Corpus: []RouterCorpusEntry{
				{Text: "rust compile error", Labels: []string{"code"}},
				{Text: "hello how are you", Labels: []string{"general"}},
			},
		},
	}
	embed := func(ctx context.Context, model, text string) ([]float32, error) {
		if text == "rust compile error" || text == "fix this rust borrow checker error" {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	}
	dec, err := decideRouter(t.Context(), "agent", spec, "fix this rust borrow checker error", nil, embed)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Fallback || dec.Candidate != "coder" {
		t.Fatalf("%+v", dec)
	}
}

func TestDecideRouterKNNFallbackFar(t *testing.T) {
	spec := RouterSpec{
		Classifier: "knn",
		Embedder:   "embed",
		Fallback:   "llama3.2:3b",
		Candidates: []RouterCandidate{{Model: "coder", Labels: []string{"code"}}},
		KNN: &RouterKNN{
			SimilarityThreshold: 0.99,
			Corpus:              []RouterCorpusEntry{{Text: "rust", Labels: []string{"code"}}},
		},
	}
	embed := func(ctx context.Context, model, text string) ([]float32, error) {
		if text == "rust" {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	}
	dec, err := decideRouter(t.Context(), "agent", spec, "unrelated weather chat", nil, embed)
	if err != nil {
		t.Fatal(err)
	}
	if !dec.Fallback || dec.Candidate != "llama3.2:3b" {
		t.Fatalf("%+v", dec)
	}
}
