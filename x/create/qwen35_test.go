package create

import "testing"

func TestQwen35ShouldShiftNormKeySkipsMTP(t *testing.T) {
	if qwen35ShouldShiftNormKey("mtp.pre_fc_norm_embedding.weight") {
		t.Fatal("import must not +1 MTP norms; mlxrunner detects the convention")
	}
	if qwen35ShouldShiftNormKey("language_model.mtp.layers.0.input_layernorm.weight") {
		t.Fatal("import must not +1 namespaced MTP norms")
	}
	if !qwen35ShouldShiftNormKey("model.layers.0.input_layernorm.weight") {
		t.Fatal("trunk layernorm still shifts when conv1d says so")
	}
	if !qwen35ShouldShiftNormKey("model.norm.weight") {
		t.Fatal("trunk final norm still shifts")
	}
}
