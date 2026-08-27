package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestMLXRunnerLinksQwen2(t *testing.T) {
	for _, arch := range []string{
		"Qwen2ForCausalLM",
		"Qwen3ForCausalLM",
		"GraniteForCausalLM",
		"Lfm2ForCausalLM",
	} {
		if !base.Registered(arch) {
			t.Errorf("architecture %s is implemented but not blank-imported in imports.go", arch)
		}
	}
}

func TestMLXRunnerLinksQwen35MTPDraft(t *testing.T) {
	if !base.DraftRegistered("qwen3_5_mtp") {
		t.Fatal("qwen3_5_mtp draft must be registered via x/models/qwen3_5 init")
	}
}
