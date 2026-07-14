package llm

import (
	"fmt"
	"log/slog"
	"runtime"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
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
		"draft-eagle3", "eagle3", "dflash", "draft-dflash",
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
		return fmt.Errorf("model requires llama-server for %s; build llama-server (./scripts/build/build_ollama_llama_server_darwin.sh) or set LLAMA_SERVER_BIN", spec)
	}
	return nil
}
