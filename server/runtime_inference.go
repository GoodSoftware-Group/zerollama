package server

import (
	"errors"

	"github.com/ollama/ollama/envconfig"
)

// ErrRuntimeInferenceModel is returned when the scheduler must not load ggml
// because the Modelfile routes inference to the Python runtime sidecar.
var ErrRuntimeInferenceModel = errors.New(
	"model uses zerollama-runtime inference: use /api/generate or /api/chat (runtime proxy; set ZEROLLAMA_RUNTIME_URL); " +
		"embed/vision/thinking still use the legacy runner when required",
)

// deferInferenceToRuntime reports whether load should skip spawning the ggml runner.
// Models with thinking in the manifest keep ggml available for legacy chat paths.
func deferInferenceToRuntime(m *Model) bool {
	if effectiveRuntimeURL() == "" || envconfig.LegacyRunnerForced() {
		return false
	}
	if !modelUsesRuntimeInference(m) {
		return false
	}
	return !modelRequiresLegacyRunnerCapabilities(m)
}
