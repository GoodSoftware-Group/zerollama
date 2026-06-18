package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
)

func TestEnrichModelDetailsFromGGML_MoEExperts(t *testing.T) {
	details := api.ModelDetails{}
	kv := ggml.KV{
		"general.architecture":         "test",
		"general.parameter_count":      uint64(1_000),
		"test.expert_count":            uint32(4),
		"test.expert_used_count":       uint32(2),
		"test.context_length":          uint32(8192),
		"test.feed_forward_length":     uint32(128),
		"test.embedding_length":        uint32(16),
		"test.attention.head_count":    uint32(1),
		"test.attention.head_count_kv": uint32(1),
	}

	expertParams := routedExpertParameterItems([]*ggml.Tensor{
		{Name: "blk.0.ffn_gate_exps.weight", Shape: []uint64{10, 10, 4}},
		{Name: "blk.0.ffn_down_shexp.weight", Shape: []uint64{20, 5}},
		{Name: "blk.0.ffn_up.weight", Shape: []uint64{10, 10}},
	})
	if expertParams != 400 {
		t.Fatalf("expert params = %d, want 400", expertParams)
	}

	enrichModelDetailsFromGGML(&details, kv, ggml.Tensors{})

	if details.ArchitectureType != "moe" {
		t.Fatalf("architecture_type = %q, want moe", details.ArchitectureType)
	}
	if details.ExpertCount != 4 || details.ExpertUsedCount != 2 {
		t.Fatalf("experts = %dx%d, want 4x2", details.ExpertCount, details.ExpertUsedCount)
	}
}

func TestActiveParameterCount(t *testing.T) {
	got := activeParameterCount(1_000, 400, 4, 2)
	if got != 800 {
		t.Fatalf("active params = %d, want 800", got)
	}
}
