package server

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/ollama/ollama/llm"
)

func TestLoadRouterFile(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/router.yaml"
	raw := []byte(`
routers:
  agent:
    classifier: tiny
    fallback: llama3.2:3b
    policies:
      - label: code
        description: programming
    candidates:
      - model: qwen2.5-coder:7b
        labels: [code]
`)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_ROUTER_CONFIG", path)
	routerFileMu.Lock()
	routerFileCache = nil
	routerFileMu.Unlock()
	spec, ok := lookupRouter("agent")
	if !ok || spec.Classifier != "tiny" || spec.Fallback != "llama3.2:3b" {
		t.Fatalf("%v %+v", ok, spec)
	}
}

func TestMatchRouterCandidate(t *testing.T) {
	cands := []RouterCandidate{
		{Model: "coder", Labels: []string{"code"}},
		{Model: "general", Labels: []string{"code", "general"}},
	}
	if got := matchRouterCandidate(cands, []string{"code"}); got != "coder" {
		t.Fatalf("got %q", got)
	}
	if got := matchRouterCandidate(cands, []string{"code", "general"}); got != "general" {
		t.Fatalf("got %q", got)
	}
	if got := matchRouterCandidate(cands, nil); got != "" {
		t.Fatalf("empty active: %q", got)
	}
}

func TestSoftmax(t *testing.T) {
	sm := softmax([]float64{0, math.Inf(-1)})
	if sm[0] < 0.99 || sm[1] > 0.01 {
		t.Fatalf("%v", sm)
	}
}

func TestDecideRouterFallback(t *testing.T) {
	spec := RouterSpec{
		Classifier: "cls",
		Fallback:   "llama3.2:3b",
		Policies:   []RouterPolicy{{Label: "code"}, {Label: "general"}},
		Candidates: []RouterCandidate{{Model: "coder", Labels: []string{"code"}}},
	}
	score := func(ctx context.Context, classifier, prompt string, cands []string, _ bool) ([]llm.CandidateScore, error) {
		return []llm.CandidateScore{
			{Candidate: "code", LogProb: -10},
			{Candidate: "general", LogProb: -10},
		}, nil
	}
	dec, err := decideRouter(t.Context(), "agent", spec, "hello", score, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !dec.Fallback || dec.Candidate != "llama3.2:3b" {
		t.Fatalf("%+v", dec)
	}
}

func TestDecideRouterPicksCoveringCandidate(t *testing.T) {
	spec := RouterSpec{
		Classifier:          "cls",
		ActivationThreshold: 0.4,
		Policies:            []RouterPolicy{{Label: "code"}, {Label: "general"}},
		Candidates: []RouterCandidate{
			{Model: "coder", Labels: []string{"code"}},
			{Model: "chat", Labels: []string{"general"}},
		},
	}
	score := func(ctx context.Context, classifier, prompt string, cands []string, _ bool) ([]llm.CandidateScore, error) {
		return []llm.CandidateScore{
			{Candidate: "code", LogProb: -0.1},
			{Candidate: "general", LogProb: -5},
		}, nil
	}
	dec, err := decideRouter(t.Context(), "agent", spec, "fix this rust borrow checker error", score, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Fallback || dec.Candidate != "coder" {
		t.Fatalf("%+v", dec)
	}
	if len(dec.Labels) == 0 || dec.Labels[0] != "code" {
		t.Fatalf("labels=%v", dec.Labels)
	}
}
