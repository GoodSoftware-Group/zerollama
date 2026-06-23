//go:build edge

// server_edge.go is the Phase 16 v2 edge entry for GGUF inference.
//
// WHY edge-only: default builds compile server.go (llama.cpp CGO subprocess). Edge artifacts
// must link only Go → llama-server so operators get upstream-shaped binaries without shipping
// two inference engines. StartRunner is stubbed because GPU discovery uses llama-server bootstrap
// when GgmlRunnerLinked() is false (see discover/gpu_discovery_upstream.go).
package llm

import (
	"fmt"
	"io"
	"log/slog"
	"os/exec"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

// NewLlamaServer routes eligible GGUF loads through llama-server only.
// WHY no ggml fallback: edge builds exclude server.go; attempting ggml would require linking
// llama.cpp CGO we deliberately dropped in v2.
func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int, config LlamaServerConfig) (LlamaServer, error) {
	if err := ggmlRunnerRequired(projectors); err != nil {
		return nil, err
	}
	if !useLlamaServerBackend(projectors) {
		return nil, fmt.Errorf("%w; set ZEROLLAMA_LLAMA_SERVER=1/auto or use --edge", ErrGgmlRunnerUnlinked)
	}

	trainCtx := f.KV().ContextLength()
	if opts.NumCtx > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx)
	}
	kvct := opts.KvCacheTypeEffective()
	slog.Info("using llama-server subprocess for model", "model", modelPath)
	return NewLlamaServerRunner(gpus, modelPath, f, adapters, projectors, opts, numParallel, kvct, config)
}

// StartRunner is unavailable in edge builds.
// WHY stub instead of panic: discover/bootstrap used to spawn ollama-engine for GPU enumeration;
// edge discovery must use llama-server probe paths only.
func StartRunner(_ bool, _ string, _ []string, _ io.Writer, _ map[string]string) (*exec.Cmd, int, error) {
	return nil, 0, fmt.Errorf("ggml runner subprocess is not included in edge builds; use llama-server discovery (ZEROLLAMA_LLAMA_SERVER=1/auto)")
}
