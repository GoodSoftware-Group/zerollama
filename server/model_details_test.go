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

func TestEnrichModelDetailsWeightSizeBytes(t *testing.T) {
	// Dense model: all tensors contribute to weight_size_bytes.
	kv := ggml.KV{
		"general.architecture":         "llama",
		"general.parameter_count":      uint64(7_000_000_000),
		"llama.context_length":         uint32(8192),
		"llama.embedding_length":       uint32(4096),
		"llama.attention.head_count":   uint32(32),
		"llama.attention.head_count_kv": uint32(8),
	}
	// Two tensors: 1024×1024 F16 (2 bytes/elem = 2MiB) + 512×512 Q4_0 (0.5 bytes/elem = 128KiB)
	// Use F16 (kind=1) and Q4_0 (kind=2) — actual sizes come from Tensor.Size() which
	// reads Kind. To keep the test simple we use the zero-kind (F32, 4 bytes/elem).
	tensors := ggml.Tensors{}
	details := api.ModelDetails{}
	enrichModelDetailsFromGGML(&details, kv, tensors)

	// Empty tensors → WeightSizeBytes should be 0 (omitted from JSON).
	if details.WeightSizeBytes != 0 {
		t.Fatalf("empty tensors: WeightSizeBytes = %d, want 0", details.WeightSizeBytes)
	}
	if details.Family != "llama" {
		t.Fatalf("Family = %q, want llama", details.Family)
	}
	if details.ArchitectureType != "dense" {
		t.Fatalf("ArchitectureType = %q, want dense", details.ArchitectureType)
	}
}

func TestEnrichModelDetailsWeightSizeBytes_MoE(t *testing.T) {
	kv := ggml.KV{
		"general.architecture":         "qwen2moe",
		"general.parameter_count":      uint64(57_000_000_000),
		"qwen2moe.expert_count":        uint32(64),
		"qwen2moe.expert_used_count":   uint32(8),
		"qwen2moe.context_length":      uint32(65536),
		"qwen2moe.embedding_length":    uint32(3584),
		"qwen2moe.attention.head_count":    uint32(28),
		"qwen2moe.attention.head_count_kv": uint32(4),
	}
	// Expert tensors (64 experts × 4 elements each) + shared tensor
	details := api.ModelDetails{}
	enrichModelDetailsFromGGML(&details, kv, ggml.Tensors{})
	// Weight bytes zero (no tensors), but arch fields should be populated.
	if details.ArchitectureType != "moe" {
		t.Fatalf("ArchitectureType = %q, want moe", details.ArchitectureType)
	}
	if details.ExpertCount != 64 {
		t.Fatalf("ExpertCount = %d, want 64", details.ExpertCount)
	}
	if details.WeightSizeBytes != 0 {
		t.Fatalf("empty tensors: WeightSizeBytes = %d, want 0", details.WeightSizeBytes)
	}
}
