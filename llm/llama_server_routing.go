package llm

import (
	"log/slog"
	"runtime"
	"sync"

	"github.com/ollama/ollama/envconfig"
)

var (
	llamaServerProbeOnce sync.Once
	llamaServerProbeOK   bool
)

// plainTextGGUFEligibleForLlamaServer reports whether a load may use Go → llama-server
// instead of the legacy CGO llamarunner or Go ollamarunner subprocess.
//
// WHY projectors only: vision/thinking models may still need ggml runner paths until
// llama-server parity is signed off on ship hardware (Phase 17 criterion 6).
func plainTextGGUFEligibleForLlamaServer(projectors []string) bool {
	return len(projectors) == 0
}

// LlamaServerDiscoverable reports whether llama-server is on disk (cached per process).
func LlamaServerDiscoverable() bool {
	llamaServerProbeOnce.Do(func() {
		_, err := FindLlamaServer()
		llamaServerProbeOK = err == nil
	})
	return llamaServerProbeOK
}

// useLlamaServerBackend picks upstream-shaped Go → llama-server for eligible GGUF loads.
//
// Explicit: ZEROLLAMA_LLAMA_SERVER=1 or --llama-server-backend.
// Linux auto (Phase 17): when unset and llama-server is discoverable, plain text GGUF
// uses llama-server so llamarunner/ollamarunner are not on the hot path.
// Darwin: opt-in only — M7 bench keeps ggml Metal default (~164 vs ~158 tok/s).
func useLlamaServerBackend(projectors []string) bool {
	if envconfig.LlamaServerBackendDisabled() {
		return false
	}
	if !plainTextGGUFEligibleForLlamaServer(projectors) {
		return false
	}
	if envconfig.LlamaServerBackend() {
		return true
	}
	if runtime.GOOS != "linux" {
		return false
	}
	if !LlamaServerDiscoverable() {
		return false
	}
	slog.Debug("Phase 17: routing plain text GGUF through llama-server (Linux auto-default)")
	return true
}
