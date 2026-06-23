package server

import (
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

// inferenceBackendPolicy reports how local GGUF inference is routed for fleet ops and smokes.
// WHY expose in /api/status: env vars alone are ambiguous (Linux auto, edge mode, harness flags);
// one JSON snapshot lets phase16/17 smokes and fleet schedulers assert policy without parsing logs.
func inferenceBackendPolicy() api.BackendPolicy {
	p := api.BackendPolicy{
		Edge:       envconfig.EdgeMode(),
		EdgeBuild:  version.IsEdgeBuild(),
		GgmlLinked: envconfig.GgmlRunnerLinked(),
		GgufPath:   "ggml",
	}
	switch {
	case envconfig.LlamaServerBackendDisabled():
		p.LlamaServer = "off"
	case envconfig.LlamaServerBackendAuto():
		p.LlamaServer = "auto"
	case envconfig.LlamaServerBackendExplicit():
		p.LlamaServer = "explicit"
	default:
		p.LlamaServer = "off"
	}
	if envconfig.LlamaCppBackend() {
		p.LlamaCppHarness = true
	}
	switch {
	case envconfig.LlamaCppBackend():
		p.GgufPath = "runtime"
	case envconfig.LlamaServerBackend() && !envconfig.LlamaServerBackendDisabled():
		p.GgufPath = "llama-server"
	default:
		p.GgufPath = "ggml"
	}
	if p.Edge {
		p.RuntimeChat = "off"
	} else if envconfig.RuntimeDefaultOn() {
		p.RuntimeChat = "on"
	} else {
		p.RuntimeChat = "off"
	}
	return p
}
