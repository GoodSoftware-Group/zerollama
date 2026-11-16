package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestMLXArchitecturesRegistered(t *testing.T) {
	for _, arch := range []string{
		"Lfm2VlForConditionalGeneration",
		"NemotronHForCausalLM",
		"NemotronH_Nano_VL_V2",
		"NemotronH_Nano_Omni_Reasoning_V3",
		"NemotronH_Omni_Reasoning_V3",
		"MuseGlimmerForConditionalGeneration",
		"KevForDecision",
	} {
		if !base.Registered(arch) {
			t.Errorf("%s is not registered", arch)
		}
	}
}
