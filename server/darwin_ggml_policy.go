package server

import (
	"context"
	"errors"
	"runtime"

	"github.com/ollama/ollama/envconfig"
)

// ErrDarwinMetalContention is returned when the runtime sidecar holds Metal and a
// ggml runner load would share the same device (common Mac smoke failure mode).
var ErrDarwinMetalContention = errors.New(
	"darwin: runtime sidecar holds Metal; use runtime-routed /api/generate or /api/chat, " +
		"unload the runtime model, or set ZEROLLAMA_LEGACY_RUNNER=1 to force ggml",
)

// darwinRuntimeMetalBlocksGgml reports whether a new ggml load should defer to the
// Python runtime because the sidecar already loaded a model on Metal.
func darwinRuntimeMetalBlocksGgml(ctx context.Context, m *Model) bool {
	if runtime.GOOS != "darwin" {
		return false
	}
	if effectiveRuntimeURL() == "" || envconfig.LegacyRunnerForced() {
		return false
	}
	h := runtimeInferenceHealth(ctx)
	if !h.ok || !h.llamaLoaded {
		return false
	}
	return modelEligibleForRuntimeDefault(m)
}

// darwinGgmlContentionWithRuntime is true when runtime holds Metal but the model
// cannot use runtime-default routing (vision, MLX, etc.).
func darwinGgmlContentionWithRuntime(ctx context.Context, m *Model) bool {
	if runtime.GOOS != "darwin" {
		return false
	}
	if effectiveRuntimeURL() == "" || envconfig.LegacyRunnerForced() {
		return false
	}
	h := runtimeInferenceHealth(ctx)
	if !h.ok || !h.llamaLoaded {
		return false
	}
	// Scheduler should always pass a model; nil is treated as "cannot route to runtime"
	// so block ggml rather than risk dual Metal residency.
	if m == nil {
		return true
	}
	return !modelEligibleForRuntimeDefault(m)
}
