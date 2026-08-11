package server

// Inference-path detection and cross-backend safety gates.
//
// WHY this file exists: zerollama serves MLX safetensors, ggml Metal/CUDA, llama-server,
// and Python runtime from one daemon. Optimizations (trie branch rewrite, eliza metadata,
// session gate participation) help agents on MLX but can hurt or confuse vanilla Ollama,
// vLLM, or unkeyed CUDA batch jobs if applied unconditionally.
//
// modelInferencePath     — advertises runner_paths on GET /api/version
// modelSupportsSessionQoS — skips embedding-only models; avoids GGUF file read at schedule time
// gateSessionKey         — MLX rewrites aux/bg branches; GGUF preserves client keys
// agentSessionMetadataEnabled — eliza/prefixHash only for harness/zerollama hints
//
// See docs/agent-qos-and-project-tracking.md.

import (
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// InferenceRunnerPath is the backend that serves a model after load routing.
type InferenceRunnerPath string

const (
	InferencePathUnknown       InferenceRunnerPath = "unknown"
	InferencePathMLX           InferenceRunnerPath = "mlx"
	InferencePathGGUFGgml      InferenceRunnerPath = "gguf_ggml"
	InferencePathGGUFLlama     InferenceRunnerPath = "gguf_llama_server"
	InferencePathRuntime       InferenceRunnerPath = "runtime"
)

// modelInferencePath reports the expected runner for scheduling and client hints.
// Context-free: used before load when only manifest/path is known.
func modelInferencePath(m *Model) InferenceRunnerPath {
	if m == nil {
		return InferencePathUnknown
	}
	if m.IsMLX() {
		return InferencePathMLX
	}
	if m.ModelPath == "" {
		return InferencePathUnknown
	}
	if deferInferenceToRuntime(m) || modelUsesRuntimeInference(m) {
		return InferencePathRuntime
	}
	if skip, _ := schedSkipGgmlRunnerLoad(m); skip {
		return InferencePathGGUFLlama
	}
	if envconfig.LlamaServerBackend() && !envconfig.LlamaServerBackendDisabled() {
		return InferencePathGGUFLlama
	}
	return InferencePathGGUFGgml
}

// modelSupportsSessionQoS is true for text runners that participate in the session gate.
// We deliberately avoid CheckCapabilities here — that requires reading the GGUF file,
// which is unavailable at schedule time and returns false for embedding-only models.
// Instead we use a conservative allow-list: MLX always participates; GGUF models with
// a ModelPath are assumed to be text/completion unless their capabilities explicitly say
// embedding-only (populated from manifest when available).
func modelSupportsSessionQoS(m *Model) bool {
	if m == nil {
		return false
	}
	if m.IsMLX() {
		return true
	}
	if m.ModelPath == "" {
		return false
	}
	// Embedding-only models should not enter the interactive gate.
	// Capabilities slice is populated from manifest; if empty we assume completion.
	caps := m.Config.Capabilities
	if len(caps) == 0 {
		return true
	}
	for _, c := range caps {
		if c == string(model.CapabilityCompletion) {
			return true
		}
	}
	// Has capabilities but none is completion — skip gate.
	return false
}

// modelUsesMLXScheduleOptimizations is true for safetensors MLX-only schedule helpers.
func modelUsesMLXScheduleOptimizations(m *Model) bool {
	return m != nil && m.IsMLX()
}

// gateSessionKey maps client session metadata to the internal defer gate key.
// MLX may rewrite auxiliary/background keys onto shared trie branches; GGUF and
// llama-server keep explicit client keys so ps/L3 labels stay aligned.
// Unkeyed GGUF must stay empty — inventing aux:/bg: branches stalls vanilla
// chat behind MLX cooldowns (90s) and desyncs llama-server L3 labels.
func gateSessionKey(m *Model, modelKey, rawKey string, class mlxSessionClass, qos mlxQoS) string {
	key := strings.TrimSpace(rawKey)
	if m != nil && !m.IsMLX() {
		return key
	}
	return injectMLXSessionKey(modelKey, rawKey, class, qos)
}

// agentSessionMetadataEnabled reports whether to enrich eliza/prefixHash metadata.
// Plain prompt_cache_key without harness hints is enough for llama-server cache_n.
func agentSessionMetadataEnabled(opts map[string]any) bool {
	key := promptCacheKeyFromOptions(opts)
	if key == "" {
		return false
	}
	if z, ok := opts["zerollama"].(map[string]any); ok && len(z) > 0 {
		return true
	}
	if agentPromptCacheKeyFromEliza(opts) != "" {
		return true
	}
	for _, prefix := range []string{"hermes:", "ruby-trivia:", "simpleagent:", "conv:"} {
		if strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}

// zerollamaVersionRunnerPaths advertises backends this node can use (build + policy).
func zerollamaVersionRunnerPaths() []string {
	paths := []string{string(InferencePathMLX), string(InferencePathGGUFGgml)}
	if envconfig.LlamaServerBackend() && !envconfig.LlamaServerBackendDisabled() {
		paths = append(paths, string(InferencePathGGUFLlama))
	} else if !envconfig.GgmlRunnerLinked() || envconfig.EdgeMode() {
		paths = append(paths, string(InferencePathGGUFLlama))
	}
	if effectiveRuntimeURL() != "" {
		paths = append(paths, string(InferencePathRuntime))
	}
	return paths
}
