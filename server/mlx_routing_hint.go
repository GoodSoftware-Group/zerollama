package server

import (
	"runtime"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// enrichShowMLXRouting adds advisory routing keys for safetensors models on Darwin.
// GGUF remains the default for pulled library models; this does not change sched.go.
func enrichShowMLXRouting(modelInfo map[string]any, m *Model) {
	if modelInfo == nil || m == nil || !m.IsMLX() {
		return
	}
	modelInfo["zerollama.inference_path"] = string(InferencePathMLX)
	if runtime.GOOS != "darwin" {
		modelInfo["zerollama.routing_note"] = "safetensors → mlxrunner (build MLX dylibs on this host to load)"
		return
	}
	arch := strings.TrimSpace(m.PrimaryFamily())
	if arch == "" {
		if a, ok := modelInfo["general.architecture"].(string); ok {
			arch = strings.TrimSpace(a)
		}
	}
	registered := arch != "" && base.Registered(arch)
	modelInfo["zerollama.mlx_arch_registered"] = registered
	if registered {
		modelInfo["zerollama.routing_note"] = "MLX-native safetensors: prefer this tag on Apple Silicon (mlxrunner). GGUF pulls still default to ggml Metal unless you opt into runtime/llama-server."
	} else {
		modelInfo["zerollama.routing_note"] = "safetensors tag; MLX load may fail until architecture is registered in x/models. GGUF remains the default Metal path for library models."
	}
}
