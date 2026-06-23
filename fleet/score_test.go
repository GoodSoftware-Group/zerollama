package fleet

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestScoreCandidatesWarmLowestQueue(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", URL: "http://a:11434", Available: true, LoadedModels: []string{"llama3:latest"}, QueueDepth: 2},
		{ID: "b", URL: "http://b:11434", Available: true, LoadedModels: []string{"llama3:latest"}, QueueDepth: 0},
		{ID: "c", URL: "http://c:11434", Available: true, QueueDepth: 0},
	}
	result := ScoreCandidates(nodes, ScoreRequest{Model: "llama3:latest"}, nil)
	if result.Best == nil || result.Best.ID != "b" {
		t.Fatalf("best=%+v", result.Best)
	}
	if !result.Best.Warm {
		t.Fatalf("expected warm best")
	}
}

func TestScoreCandidatesAffinityBeatsQueue(t *testing.T) {
	cache := NewPrefixCache(0)
	cache.Remember("llama3", "thread-1", "b")

	nodes := []NodeSnapshot{
		{ID: "a", Available: true, URL: "http://a:11434", LoadedModels: []string{"llama3"}, QueueDepth: 0},
		{ID: "b", Available: true, URL: "http://b:11434", LoadedModels: []string{"llama3"}, QueueDepth: 5},
	}
	result := ScoreCandidates(nodes, ScoreRequest{Model: "llama3", SessionKey: "thread-1"}, cache)
	if result.Best == nil || result.Best.ID != "b" {
		t.Fatalf("best=%+v", result.Best)
	}
	if !result.Best.Affinity {
		t.Fatalf("expected affinity on best")
	}
}

func TestScoreCandidatesWarmOnlyFiltersCold(t *testing.T) {
	nodes := []NodeSnapshot{
		{ID: "a", Available: true, URL: "http://a:11434", QueueDepth: 0},
	}
	result := ScoreCandidates(nodes, ScoreRequest{Model: "llama3", WarmOnly: true}, nil)
	if result.Best != nil {
		t.Fatalf("expected no best, got %+v", result.Best)
	}
}

func TestScoreCandidatesCapacityPressure(t *testing.T) {
	nodes := []NodeSnapshot{
		{
			ID: "a", Available: true, URL: "http://a:11434",
			LoadedModels: []string{"llama3:latest", "vision:latest"},
			QueueDepth:   0,
			Inference: api.InferenceStatus{
				Ggml: api.GgmlStatus{
					LoadedModelDetails: []api.GgmlLoadedModelStatus{
						{Name: "llama3:latest", LoadedModelMetadata: api.LoadedModelMetadata{NumCtx: 8192}},
						{Name: "vision:latest", LoadedModelMetadata: api.LoadedModelMetadata{NumCtx: 8192}},
					},
				},
			},
		},
		{
			ID: "b", Available: true, URL: "http://b:11434",
			LoadedModels: []string{"llama3:latest"},
			QueueDepth:   0,
			Inference: api.InferenceStatus{
				Ggml: api.GgmlStatus{
					LoadedModelDetails: []api.GgmlLoadedModelStatus{
						{Name: "llama3:latest", LoadedModelMetadata: api.LoadedModelMetadata{NumCtx: 4096}},
					},
				},
			},
		},
	}
	result := ScoreCandidates(nodes, ScoreRequest{Model: "llama3:latest"}, nil)
	if result.Best == nil || result.Best.ID != "b" {
		t.Fatalf("best=%+v", result.Best)
	}
}

func TestScoreCandidatesPreferWarmFalseStillWarm(t *testing.T) {
	preferWarm := false
	nodes := []NodeSnapshot{
		{ID: "a", Available: true, URL: "http://a:11434", LoadedModels: []string{"llama3"}, QueueDepth: 0},
	}
	result := ScoreCandidates(nodes, ScoreRequest{Model: "llama3", PreferWarm: &preferWarm}, nil)
	if result.Best == nil || !result.Best.Warm {
		t.Fatalf("best=%+v", result.Best)
	}
}
