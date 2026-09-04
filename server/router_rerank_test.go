package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestDecideRouterRerank(t *testing.T) {
	spec := RouterSpec{
		Classifier:          "rerank",
		Reranker:            "bge-reranker",
		ActivationThreshold: 0.5,
		Fallback:            "llama3.2:3b",
		Policies: []RouterPolicy{
			{Label: "code", Description: "programming in rust"},
			{Label: "general", Description: "chitchat"},
		},
		Candidates: []RouterCandidate{
			{Model: "coder", Labels: []string{"code"}},
			{Model: "chat", Labels: []string{"general"}},
		},
	}
	rerank := func(ctx context.Context, model, query string, docs []string) ([]llm.RerankHit, error) {
		if model != "bge-reranker" || query == "" || len(docs) != 2 {
			t.Fatalf("unexpected %q %q %v", model, query, docs)
		}
		return []llm.RerankHit{{Index: 0, RelevanceScore: 0.9}, {Index: 1, RelevanceScore: 0.1}}, nil
	}
	dec, err := decideRouter(t.Context(), "agent", spec, "fix rust compile error", nil, nil, rerank)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Fallback || dec.Candidate != "coder" {
		t.Fatalf("%+v", dec)
	}
	if len(dec.Labels) != 1 || dec.Labels[0] != "code" {
		t.Fatalf("labels=%v", dec.Labels)
	}
}

func TestDecideRouterRerankFallback(t *testing.T) {
	spec := RouterSpec{
		Classifier: "colbert",
		Embedder:   "rerank-gguf",
		Fallback:   "llama3.2:3b",
		Policies:   []RouterPolicy{{Label: "code", Description: "code"}},
		Candidates: []RouterCandidate{{Model: "coder", Labels: []string{"code"}}},
	}
	rerank := func(ctx context.Context, model, query string, docs []string) ([]llm.RerankHit, error) {
		return []llm.RerankHit{{Index: 0, RelevanceScore: 0.01}}, nil
	}
	dec, err := decideRouter(t.Context(), "agent", spec, "hello", nil, nil, rerank)
	if err != nil {
		t.Fatal(err)
	}
	if !dec.Fallback || dec.Candidate != "llama3.2:3b" {
		t.Fatalf("%+v", dec)
	}
}
