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
// Explicit: ZEROLLAMA_LLAMA_SERVER=1 or --llama-server-backend — all GGUF including
// vision (split mmproj or inline v.* tensors) and thinking templates (enable_thinking).
// Linux auto: ZEROLLAMA_LLAMA_SERVER=auto (set by serve on Linux when binary found) —
// all GGUF on Linux, matching upstream default shape.
// Darwin: opt-in only — M7 bench keeps ggml Metal default (~164 vs ~158 tok/s).
func useLlamaServerBackend(projectors []string) bool {
	return useLlamaServerBackendGOOS(runtime.GOOS, projectors, LlamaServerDiscoverable())
}

func useLlamaServerBackendGOOS(goos string, projectors []string, discoverable bool) bool {
	if envconfig.LlamaServerBackendDisabled() {
		return false
	}
	if envconfig.LlamaServerBackendExplicit() {
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
	_ = projectors // Linux auto routes all GGUF (text, vision, thinking) like upstream.
	slog.Debug("Phase 17: routing GGUF through llama-server (Linux auto-default)")
	return true
}
