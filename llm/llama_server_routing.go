package llm

import (
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
)

var (
	llamaServerProbeOnce sync.Once
	llamaServerProbeOK   bool
)

// LlamaServerDiscoverable reports whether llama-server is on disk (cached per process).
func LlamaServerDiscoverable() bool {
	llamaServerProbeOnce.Do(func() {
		_, err := FindLlamaServer()
		llamaServerProbeOK = err == nil
	})
	return llamaServerProbeOK
}

// ModelNeedsLlamaServerSpec reports whether a model manifest requires llama-server for
// speculative decoding (Eagle3/DFlash, MTP, n-gram). Plain GGUF stays on ggml Metal on Mac.
func ModelNeedsLlamaServerSpec(config LlamaServerConfig) bool {
	if config.EnableMTP {
		return true
	}
	if strings.TrimSpace(config.DraftModelPath) != "" {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(config.SpecType)) {
	case "ngram", "ngram-simple",
		"draft-eagle3", "eagle3", "dflash",
		"draft-mtp", "mtp":
		return true
	default:
		return false
	}
}

// useLlamaServerBackend picks upstream-shaped Go → llama-server for eligible GGUF loads
// when no per-model config is available (legacy callers).
func useLlamaServerBackend(projectors []string) bool {
	return useLlamaServerBackendForModel(projectors, LlamaServerConfig{})
}

// llamaServerBlockedByOllamaRawMXFP4 reports GGUFs that store MXFP4 as tensor
// type id 4 (deprecated Q4_2 slot). Ollama's engine remaps those to ggml
// GGML_TYPE_MXFP4 (39) and reshuffles the block bytes on load
// (ml/backend/ggml/ggml.go). Stock llama.cpp rejects type 4
// ("DEPRECATED" / blck_size 0) — e.g. registry gpt-oss:20b MXFP4.
// Re-exports that already use type 39 (e.g. some community gpt-oss builds)
// are fine on llama-server.
func llamaServerBlockedByOllamaRawMXFP4(f *ggml.GGML) bool {
	if f == nil {
		return false
	}
	for _, t := range f.Tensors().Items() {
		if t.Kind == 4 {
			return true
		}
	}
	return false
}

// ggmlUsesNativeMXFP4 reports llama.cpp GGML_TYPE_MXFP4 (type id 39) tensors.
// Serve defaults GGML_CUDA_FORCE_CUBLAS=1 for IQ* stability on 5080/CUDA 13,
// but that disables MMQ and sends MoE MUL_MAT_ID through a broken cuBLAS path
// (upstream ggml-org/llama.cpp#19659) — gpt-oss MXFP4 then loops on token "?".
func ggmlUsesNativeMXFP4(f *ggml.GGML) bool {
	if f == nil {
		return false
	}
	for _, t := range f.Tensors().Items() {
		if ggml.TensorType(t.Kind) == ggml.TensorTypeMXFP4 {
			return true
		}
	}
	return false
}

// applyLlamaServerMXFP4CUDAEnv clears FORCE_CUBLAS for native MXFP4 so MMQ/MMVQ
// run. IQ* models without type-39 keep the serve default (FORCE_CUBLAS=1).
// Operator escape: ZEROLLAMA_MXFP4_ALLOW_FORCE_CUBLAS=1 keeps inherited value.
func applyLlamaServerMXFP4CUDAEnv(envs map[string]string, f *ggml.GGML) {
	if envs == nil || !ggmlUsesNativeMXFP4(f) {
		return
	}
	if strings.EqualFold(strings.TrimSpace(os.Getenv("ZEROLLAMA_MXFP4_ALLOW_FORCE_CUBLAS")), "1") ||
		strings.EqualFold(strings.TrimSpace(os.Getenv("ZEROLLAMA_MXFP4_ALLOW_FORCE_CUBLAS")), "true") {
		return
	}
	envs["GGML_CUDA_FORCE_CUBLAS"] = "0"
	slog.Info("MXFP4 GGUF: disabling GGML_CUDA_FORCE_CUBLAS for llama-server (need MMQ)",
		"architecture", f.KV().Architecture())
}

// useLlamaServerBackendForModel decides engine routing for one model load.
//
// Explicit: ZEROLLAMA_LLAMA_SERVER=1 or --llama-server-backend — all GGUF.
// Linux auto: all GGUF when llama-server is discoverable.
// Darwin spec auto: speculative tags only (plain GGUF keeps ggml Metal default).
// Vision split mmproj on Darwin still requires explicit opt-in (Linux auto includes vision).
func useLlamaServerBackendForModel(projectors []string, config LlamaServerConfig) bool {
	return useLlamaServerBackendForModelGOOS(runtime.GOOS, projectors, LlamaServerDiscoverable(), config)
}

func useLlamaServerBackendForModelGOOS(goos string, projectors []string, discoverable bool, config LlamaServerConfig) bool {
	if envconfig.LlamaServerBackendDisabled() {
		return false
	}
	if envconfig.LlamaServerBackendExplicit() {
		return true
	}
	if len(projectors) > 0 && goos != "linux" {
		return false
	}
	if ModelNeedsLlamaServerSpec(config) && discoverable {
		if goos == "darwin" {
			slog.Debug("Phase 17: routing speculative model through llama-server (Darwin spec auto)",
				"spec_type", config.SpecType,
				"draft_model", config.DraftModelPath != "",
			)
		}
		return true
	}
	if !envconfig.LlamaServerBackendAuto() {
		return false
	}
	if goos != "linux" {
		return false
	}
	if !discoverable {
		return false
	}
	_ = projectors
	slog.Debug("Phase 17: routing GGUF through llama-server (Linux auto-default)")
	return true
}

// SpecModelRequiresLlamaServerError is returned when a spec-tagged model cannot load on ggml.
func SpecModelRequiresLlamaServerError(config LlamaServerConfig) error {
	if !ModelNeedsLlamaServerSpec(config) {
		return nil
	}
	spec := strings.TrimSpace(config.SpecType)
	if spec == "" {
		spec = "speculative"
	}
	if envconfig.LlamaServerBackendDisabled() {
		return fmt.Errorf("model requires llama-server for %s but ZEROLLAMA_LLAMA_SERVER=0", spec)
	}
	if !LlamaServerDiscoverable() {
		return fmt.Errorf("model requires llama-server for %s; build llama-server (./scripts/build_ollama_llama_server_darwin.sh) or set LLAMA_SERVER_BIN", spec)
	}
	return nil
}
