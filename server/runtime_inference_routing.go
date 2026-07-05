package server

import (
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
)

// modelInferenceBackend returns ModalityBackends["inference"] when set.
func modelInferenceBackend(m *Model) string {
	if m == nil || m.Config.ModalityBackends == nil {
		return ""
	}
	return strings.TrimSpace(m.Config.ModalityBackends[model.ModalityInference])
}

// capabilitiesRequiringLegacyRunner are manifest capabilities that still need the
// ggml runner for some requests (tools in chat, thinking templates, etc.).
func capabilitiesRequiringLegacyRunner() []model.Capability {
	return []model.Capability{
		model.CapabilityThinking,
	}
}

// modelRequiresLegacyRunnerCapabilities reports manifest caps that force ggml for some paths.
func modelRequiresLegacyRunnerCapabilities(m *Model) bool {
	if m == nil {
		return false
	}
	for _, cap := range capabilitiesRequiringLegacyRunner() {
		if m.CheckCapabilities(cap) == nil {
			return true
		}
	}
	return false
}

// modelExcludedFromRuntimeDefault reports models that must not use runtime-default routing.
func modelExcludedFromRuntimeDefault(m *Model) bool {
	if m == nil {
		return true
	}
	if m.CheckCapabilities(model.CapabilityCompletion) != nil {
		return true
	}
	for _, cap := range append([]model.Capability{
		model.CapabilityEmbedding,
		model.CapabilityVision,
		model.CapabilityVideo,
		model.CapabilityImage,
		model.CapabilityVideoGen,
		model.CapabilityAudio,
		model.CapabilitySpeech,
	}, capabilitiesRequiringLegacyRunner()...) {
		if m.CheckCapabilities(cap) == nil {
			return true
		}
	}
	return false
}

// modelEligibleForLlamaCppRuntime reports whether a local GGUF text model may use
// the Python runtime + llama.cpp (shared preconditions for default-on and explicit flag).
func modelEligibleForLlamaCppRuntime(m *Model) bool {
	if m == nil || effectiveRuntimeURL() == "" || envconfig.LegacyRunnerForced() {
		return false
	}
	if m.ModelPath == "" || m.IsMLX() {
		return false
	}
	backend := modelInferenceBackend(m)
	if backend != "" && backend != model.BackendZerollamaRuntime {
		return false
	}
	if modelExcludedFromRuntimeDefault(m) {
		return false
	}
	return true
}

// modelEligibleForAgentCacheRuntime reports whether a keyed agent chat may use the Python
// runtime for L3 prefix cache. Unlike modelEligibleForLlamaCppRuntime, thinking-capable
// text models (e.g. qwen3.6) remain eligible — only multimodal/embedding caps exclude.
func modelEligibleForAgentCacheRuntime(m *Model) bool {
	if m == nil || effectiveRuntimeURL() == "" || envconfig.LegacyRunnerForced() {
		return false
	}
	if m.ModelPath == "" || m.IsMLX() {
		return false
	}
	backend := modelInferenceBackend(m)
	if backend != "" && backend != model.BackendZerollamaRuntime {
		return false
	}
	if m.CheckCapabilities(model.CapabilityCompletion) != nil {
		return false
	}
	for _, cap := range []model.Capability{
		model.CapabilityEmbedding,
		model.CapabilityVision,
		model.CapabilityVideo,
		model.CapabilityImage,
		model.CapabilityVideoGen,
		model.CapabilityAudio,
		model.CapabilitySpeech,
	} {
		if m.CheckCapabilities(cap) == nil {
			return false
		}
	}
	return true
}

// modelEligibleForRuntimeDefault reports whether a local GGUF text model may use
// the Python runtime when ZEROLLAMA_RUNTIME default-on is active (Phase 12).
func modelEligibleForRuntimeDefault(m *Model) bool {
	return modelEligibleForLlamaCppRuntime(m)
}

// modelUsesRuntimeInference is true when the Modelfile or runtime-default policy
// routes text inference to the Python runtime sidecar.
func modelUsesRuntimeInference(m *Model) bool {
	if m == nil {
		return false
	}
	// Phase 16 edge: runtime chat middleman is intentionally off — all GGUF goes Go → llama-server.
	if envconfig.EdgeMode() {
		return false
	}
	if m.IsMLX() {
		return false
	}
	if modelInferenceBackend(m) == model.BackendZerollamaRuntime {
		return true
	}
	if envconfig.LlamaCppBackend() && modelEligibleForLlamaCppRuntime(m) {
		return true
	}
	if envconfig.RuntimeDefaultOn() && modelEligibleForRuntimeDefault(m) {
		return true
	}
	return false
}
