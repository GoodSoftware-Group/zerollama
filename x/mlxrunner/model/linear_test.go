package model

import "testing"

func TestIsAttentionProjection(t *testing.T) {
	yes := []string{
		"model.layers.0.self_attn.q_proj",
		"layers.3.attn.k_proj",
		"blk.0.o_proj",
		"q_proj",
		"language_model.model.layers.1.self_attn.qkv_proj",
	}
	no := []string{
		"model.layers.0.mlp.up_proj",
		"model.layers.0.mlp.down_proj",
		"model.layers.0.mlp.gate_proj",
		"lm_head",
		"embed_tokens",
	}
	for _, p := range yes {
		if !isAttentionProjection(p) {
			t.Fatalf("%q should wrap decode-quant", p)
		}
	}
	for _, p := range no {
		if isAttentionProjection(p) {
			t.Fatalf("%q must stay dense/FFN", p)
		}
	}
}
