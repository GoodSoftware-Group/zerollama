package server

import "testing"

func TestTensorNameHasInCheckpointMTP(t *testing.T) {
	if !TensorNameHasInCheckpointMTP("mtp.fc.weight") {
		t.Fatal("mtp.fc")
	}
	if !TensorNameHasInCheckpointMTP("language_model.mtp.layers.0.self_attn.q_proj.weight") {
		t.Fatal("language_model.mtp")
	}
	if TensorNameHasInCheckpointMTP("model.layers.0.self_attn.q_proj.weight") {
		t.Fatal("plain layer")
	}
	if TensorNameHasInCheckpointMTP("drafter.embed_tokens.weight") {
		t.Fatal("gemma drafter is not MTP advertise")
	}
}

func TestConfigMapHasInCheckpointMTP(t *testing.T) {
	if !ConfigMapHasInCheckpointMTP(map[string]any{"num_nextn_predict_layers": 1.0}) {
		t.Fatal("nextn")
	}
	if !ConfigMapHasInCheckpointMTP(map[string]any{
		"text_config": map[string]any{"num_nextn_predict_layers": 1.0},
	}) {
		t.Fatal("nested nextn")
	}
	if ConfigMapHasInCheckpointMTP(map[string]any{"num_hidden_layers": 28.0}) {
		t.Fatal("plain config")
	}
}
