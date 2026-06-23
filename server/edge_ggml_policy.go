package server

import (
	"errors"
	"fmt"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

// ErrEdgeGgmlRunnerDisabled is returned when Phase 16 runtime edge policy forbids ggml loads.
// WHY a distinct error from ErrGgmlRunnerUnlinked: edge *mode* (--edge) is an operator choice;
// unlinked *build* (-tags edge) is compile-time. Messages differ so smokes and doctor can tell them apart.
var ErrEdgeGgmlRunnerDisabled = errors.New(
	"ggml runner disabled in edge mode: route GGUF through llama-server (set ZEROLLAMA_LLAMA_SERVER=1, ZEROLLAMA_LLAMA_SERVER=auto, or place llama-server on disk for Linux auto routing)",
)

// schedSkipGgmlRunnerLoad reports whether the scheduler must not spawn ggml/ollama-engine
// for a GGUF model. MLX and imagegen paths are unchanged.
//
// WHY in the scheduler (not only NewLlamaServer): some code paths enqueue loads before llm
// construction; blocking here surfaces HTTP 400 to the client instead of a hung subprocess.
func schedSkipGgmlRunnerLoad(m *Model) (bool, error) {
	if m == nil || m.IsMLX() || m.ModelPath == "" {
		return false, nil
	}
	if envconfig.GgmlRunnerLinked() && !envconfig.EdgeMode() {
		return false, nil
	}
	if envconfig.LlamaServerBackend() && !envconfig.LlamaServerBackendDisabled() {
		return false, nil
	}
	if envconfig.EdgeMode() {
		return true, ErrEdgeGgmlRunnerDisabled
	}
	if !envconfig.GgmlRunnerLinked() {
		return true, fmt.Errorf("%w; set ZEROLLAMA_LLAMA_SERVER=1/auto or use --edge", llm.ErrGgmlRunnerUnlinked)
	}
	return false, nil
}
