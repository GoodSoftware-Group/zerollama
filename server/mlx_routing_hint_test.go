package server

import (
	"runtime"
	"strings"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestEnrichShowMLXRoutingDarwinRegistered(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only messaging")
	}
	m := &Model{Config: model.ConfigV2{ModelFormat: "safetensors", ModelFamily: "Qwen3ForCausalLM"}}
	info := map[string]any{"general.architecture": "qwen3"}
	enrichShowMLXRouting(info, m)
	if info["zerollama.inference_path"] != string(InferencePathMLX) {
		t.Fatalf("inference_path=%v", info["zerollama.inference_path"])
	}
	if info["zerollama.mlx_arch_registered"] != true {
		t.Fatalf("expected registered arch")
	}
	note, _ := info["zerollama.routing_note"].(string)
	if note == "" || !strings.Contains(note, "MLX-native") {
		t.Fatalf("note=%q", note)
	}
}
