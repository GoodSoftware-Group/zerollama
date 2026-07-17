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

func TestScoreCandidatesRadixResidency(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_RADIX_SCORE", "1")
	nodes := []NodeSnapshot{
		{
			ID: "a", Available: true, URL: "http://a:11434",
			LoadedModels: []string{"llama3"}, QueueDepth: 0,
		},
		{
			ID: "b", Available: true, URL: "http://b:11434",
			LoadedModels: []string{"llama3"}, QueueDepth: 1,
			Inference: api.InferenceStatus{
				Runtime: api.RuntimeStatus{
					Radix: &api.RadixMirrorStatus{
						Enabled:    true,
						RadixShare: true,
						EntryCount: 40,
					},
				},
			},
		},
	}
	// Session hint required for radix soft score.
	result := ScoreCandidates(nodes, ScoreRequest{Model: "llama3", SessionKey: "agent-1"}, nil)
	if result.Best == nil || result.Best.ID != "b" {
		t.Fatalf("best=%+v (want radix-warm b over empty a)", result.Best)
	}
	found := false
	for _, r := range result.Best.Reasons {
		if r == "radix" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected radix reason, got %v", result.Best.Reasons)
	}
}

func TestScoreCandidatesRadixHashLongestPrefix(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_RADIX_HASH_SCORE", "1")
	nodes := []NodeSnapshot{
		{
			ID: "a", Available: true, URL: "http://a:11434",
			LoadedModels: []string{"llama3"}, QueueDepth: 0,
			Inference: api.InferenceStatus{
				Runtime: api.RuntimeStatus{
					Radix: &api.RadixMirrorStatus{
						Enabled:     true,
						RadixShare:  true,
						EntryCount:  2,
						BlockHashes: []string{"h0"},
					},
				},
			},
		},
		{
			ID: "b", Available: true, URL: "http://b:11434",
			LoadedModels: []string{"llama3"}, QueueDepth: 0,
			Inference: api.InferenceStatus{
				Runtime: api.RuntimeStatus{
					Radix: &api.RadixMirrorStatus{
						Enabled:     true,
						RadixShare:  true,
						EntryCount:  3,
						BlockHashes: []string{"h0", "h1", "h2"},
					},
				},
			},
		},
	}
	result := ScoreCandidates(nodes, ScoreRequest{
		Model:             "llama3",
		PrefixBlockHashes: []string{"h0", "h1", "h2"},
	}, nil)
	if result.Best == nil || result.Best.ID != "b" {
		t.Fatalf("best=%+v (want longer hash match b over a)", result.Best)
	}
	found := false
	for _, r := range result.Best.Reasons {
		if r == "radix_hash" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected radix_hash reason, got %v", result.Best.Reasons)
	}
}

func TestScoreCandidatesRadixHashPrefersBlobDigests(t *testing.T) {
	t.Setenv("ZEROLLAMA_FLEET_RADIX_HASH_SCORE", "1")
	nodes := []NodeSnapshot{
		{
			ID: "a", Available: true, URL: "http://a:11434",
			LoadedModels: []string{"llama3"}, QueueDepth: 0,
			Inference: api.InferenceStatus{
				Runtime: api.RuntimeStatus{
					Radix: &api.RadixMirrorStatus{
						Enabled:     true,
						RadixShare:  true,
						EntryCount:  3,
						BlockHashes: []string{"h0", "h1", "h2"},
					},
				},
			},
		},
		{
			ID: "b", Available: true, URL: "http://b:11434",
			LoadedModels: []string{"llama3"}, QueueDepth: 0,
			Inference: api.InferenceStatus{
				Runtime: api.RuntimeStatus{
					Radix: &api.RadixMirrorStatus{
						Enabled:          true,
						RadixShare:       true,
						EntryCount:       3,
						BlockHashes:      []string{"h0", "h1", "h2"},
						BlobDigestBlocks: 2,
						BlobDigests:      []string{"ddd"},
					},
				},
			},
		},
	}
	result := ScoreCandidates(nodes, ScoreRequest{
		Model:             "llama3",
		PrefixBlockHashes: []string{"h0", "h1"},
	}, nil)
	if result.Best == nil || result.Best.ID != "b" {
		t.Fatalf("best=%+v (want blob-capable b)", result.Best)
	}
	found := false
	for _, r := range result.Best.Reasons {
		if r == "radix_blob" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected radix_blob reason, got %v", result.Best.Reasons)
	}
}

func TestLongestPrefixHashMatch(t *testing.T) {
	if got := longestPrefixHashMatch([]string{"a", "b", "c"}, []string{"a", "b"}); got != 2 {
		t.Fatalf("got %d want 2", got)
	}
	if got := longestPrefixHashMatch([]string{"a", "x"}, []string{"a", "b"}); got != 1 {
		t.Fatalf("got %d want 1", got)
	}
	if got := longestPrefixHashMatch([]string{"x"}, []string{"a"}); got != 0 {
		t.Fatalf("got %d want 0", got)
	}
}
