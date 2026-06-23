package cmd

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestSelectBenchModels(t *testing.T) {
	models := []api.ListModelResponse{
		{Name: "llama3.2:latest", Digest: "sha256-a", Capabilities: []model.Capability{model.CapabilityCompletion}},
		{Name: "nomic-embed-text", Digest: "sha256-b", Capabilities: []model.Capability{model.CapabilityEmbedding}},
		{Name: "cloud:stub", Digest: "sha256-c", RemoteModel: "gpt-4", RemoteHost: "eliza"},
		{Name: "lmstudio:local", Digest: "sha256-d", RemoteModel: "x", RemoteHost: "lmstudio", Capabilities: []model.Capability{model.CapabilityCompletion}},
	}

	got := selectBenchModels(models, nil)
	if len(got) != 2 {
		t.Fatalf("selectBenchModels() len = %d, want 2", len(got))
	}

	filtered := selectBenchModels(models, []string{"llama"})
	if len(filtered) != 1 || filtered[0].Name != "llama3.2:latest" {
		t.Fatalf("filter mismatch: %+v", filtered)
	}
}

func TestBenchTokPerSec(t *testing.T) {
	metrics := &api.Metrics{
		EvalCount:    128,
		EvalDuration: time.Second,
	}
	if got := benchTokPerSec(metrics); got != 128 {
		t.Fatalf("benchTokPerSec() = %v, want 128", got)
	}
	if benchTokPerSec(nil) != 0 {
		t.Fatal("expected 0 for nil metrics")
	}
}

func TestBenchPromptForEpoch(t *testing.T) {
	a := benchPromptForEpoch(0)
	b := benchPromptForEpoch(1)
	if a == b {
		t.Fatal("expected different prompts per epoch")
	}
	if a == "" {
		t.Fatal("expected non-empty prompt")
	}
}
