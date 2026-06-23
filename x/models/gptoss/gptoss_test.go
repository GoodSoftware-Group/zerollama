package gptoss

import (
	"testing"
)

func TestParseConfigGPTOSS120B(t *testing.T) {
	data := []byte(`{
		"architectures": ["GptOssForCausalLM"],
		"model_type": "gpt_oss",
		"hidden_size": 2880,
		"intermediate_size": 2880,
		"num_hidden_layers": 36,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"head_dim": 64,
		"num_local_experts": 128,
		"num_experts_per_tok": 4,
		"sliding_window": 128,
		"max_position_embeddings": 131072,
		"initial_context_length": 4096,
		"rope_theta": 150000,
		"rope_scaling": {"factor": 32.0}
	}`)

	cfg, err := parseConfig(data)
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if cfg.NumHiddenLayers != 36 {
		t.Errorf("NumHiddenLayers = %d, want 36", cfg.NumHiddenLayers)
	}
	if cfg.NumLocalExperts != 128 {
		t.Errorf("NumLocalExperts = %d, want 128", cfg.NumLocalExperts)
	}
	if cfg.NumExpertsPerTok != 4 {
		t.Errorf("NumExpertsPerTok = %d, want 4", cfg.NumExpertsPerTok)
	}
	if cfg.RopeScalingFactor != 32 {
		t.Errorf("RopeScalingFactor = %v, want 32", cfg.RopeScalingFactor)
	}
	if cfg.RopeInvScale != 1.0/32.0 {
		t.Errorf("RopeInvScale = %v, want %v", cfg.RopeInvScale, 1.0/32.0)
	}
	if len(cfg.LayerTypes) != 36 {
		t.Fatalf("layer_types len = %d, want 36 (auto-filled)", len(cfg.LayerTypes))
	}
	if !cfg.layerIsSliding(0) || cfg.layerIsSliding(1) {
		t.Errorf("unexpected sliding pattern at layers 0/1")
	}
}
