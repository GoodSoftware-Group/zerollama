package server

import (
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/runtimeworker"
)

// effectiveRuntimeURL is the Python runtime base URL (embedded loopback or external sidecar).
// Why two sources: Phase 7 embed registers runtimeworker.BaseURL() after CGO uvicorn start;
// external deployments set ZEROLLAMA_RUNTIME_URL only. Proxy, tokenize, and handoff must
// use the same resolution order so embed and sidecar behave identically to callers.
func effectiveRuntimeURL() string {
	if u := strings.TrimSpace(runtimeworker.BaseURL()); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return envconfig.RuntimeURL()
}

func runtimeProxyConfigured() bool {
	return effectiveRuntimeURL() != ""
}
