package gemma4

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/model"
)

func TestExpandGemma4OptiqExpertQuant(t *testing.T) {
	tq := map[string]*model.TensorQuantInfo{
		"language_model.model.layers.0.experts.switch_glu.gate_proj.weight": {
			QuantType: "int8",
			GroupSize: 64,
		},
		"language_model.model.embed_tokens.weight": {
			QuantType: "int8",
			GroupSize: 64,
		},
	}
	expandGemma4OptiqExpertQuant(tq, 3)
	for e := 0; e < 3; e++ {
		key := fmt.Sprintf("language_model.model.layers.0.moe.experts.%d.gate_proj.weight", e)
		info := tq[key]
		if info == nil || info.QuantType != "int8" || info.GroupSize != 64 {
			t.Fatalf("expert %d missing/wrong: %#v", e, info)
		}
	}
	if tq["language_model.model.embed_tokens.weight"].QuantType != "int8" {
		t.Fatal("embed entry should be unchanged")
	}
}
