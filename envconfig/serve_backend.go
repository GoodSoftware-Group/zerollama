package envconfig

import (
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync"

	"github.com/ollama/ollama/version"
)

var warnDeprecatedNewEngineOnce sync.Once

// ServeBackendOpts configures Phase 16/17 env before scheduler init.
type ServeBackendOpts struct {
	LlamaCppBackend         bool
	LlamaServerBackend      bool
	Edge                    bool
	GOOS                    string
	LlamaServerDiscoverable bool
}

// LinuxLlamaServerAutoEnv returns "auto" when Linux serve should enable upstream routing
// without an explicit operator flag. Empty when edge mode already set llama-server=1.
func LinuxLlamaServerAutoEnv(goos, llamaServerEnv string, edgeMode, discoverable bool) string {
	if goos != "linux" {
		return ""
	}
	if strings.TrimSpace(llamaServerEnv) != "" {
		return ""
	}
	if edgeMode {
		return ""
	}
	if !discoverable {
		return ""
	}
	return "auto"
}

// ApplyServeBackendEnv sets process env for llama.cpp harness, llama-server, edge, and Linux auto.
func ApplyServeBackendEnv(opts ServeBackendOpts) {
	if opts.GOOS == "" {
		opts.GOOS = runtime.GOOS
	}
	warnDeprecatedNewEngine()
	applyEdgeBuildServeDefault(&opts)
	if opts.LlamaCppBackend {
		_ = os.Setenv("ZEROLLAMA_LLAMA_CPP_BACKEND", "1")
	}
	if opts.LlamaServerBackend {
		_ = os.Setenv("ZEROLLAMA_LLAMA_SERVER", "1")
	}
	if opts.Edge {
		_ = os.Setenv("ZEROLLAMA_EDGE", "1")
	}
	ApplyEdgeModeDefaults()
	if auto := LinuxLlamaServerAutoEnv(opts.GOOS, Var("ZEROLLAMA_LLAMA_SERVER"), EdgeMode(), opts.LlamaServerDiscoverable); auto != "" {
		slog.Info("Phase 17: Linux auto Go → llama-server (ZEROLLAMA_LLAMA_SERVER=auto; set 0 to disable)")
		_ = os.Setenv("ZEROLLAMA_LLAMA_SERVER", auto)
	}
	ApplyLlamaCppBackendDefaults()
	ApplyLlamaServerBackendDefaults()
}

// applyEdgeBuildServeDefault enables Phase 16 edge env when built with -tags edge /
// version.EdgeBuild=true unless the operator sets ZEROLLAMA_EDGE=0.
func applyEdgeBuildServeDefault(opts *ServeBackendOpts) {
	if opts.Edge || EdgeMode() {
		return
	}
	if !version.IsEdgeBuild() {
		return
	}
	v := strings.TrimSpace(Var("ZEROLLAMA_EDGE"))
	if v == "0" || strings.EqualFold(v, "false") {
		return
	}
	opts.Edge = true
	slog.Info("Phase 16: edge-marked binary — applying upstream-shaped serve defaults (ZEROLLAMA_EDGE=0 to disable)")
}

func warnDeprecatedNewEngine() {
	if !NewEngine() {
		return
	}
	warnDeprecatedNewEngineOnce.Do(func() {
		slog.Warn("OLLAMA_NEW_ENGINE is deprecated and will be removed; use --llama-server-backend, --edge, or Linux auto (Phase 17); for Go ggml engine use OllamaEngineRequired() model families only")
	})
}
