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

// modelEligibleForRuntimeDefault reports whether a local GGUF text model may use
// the Python runtime when ZEROLLAMA_RUNTIME default-on is active (Phase 12).
func modelEligibleForRuntimeDefault(m *Model) bool {
	if m == nil || effectiveRuntimeURL() == "" || envconfig.LegacyRunnerForced() {
		return false
	}
	if m.ModelPath == "" || m.IsMLX() {
		return false
	}
	backend := modelInferenceBackend(m)
	if backend != "" && backend != model.BackendZerollamaRuntime {
		// Explicit non-runtime inference backend (e.g. future "ggml" opt-out).
		return false
	}
	if modelExcludedFromRuntimeDefault(m) {
		return false
	}
	return true
}

// modelUsesRuntimeInference is true when the Modelfile or runtime-default policy
// routes text inference to the Python runtime sidecar.
func modelUsesRuntimeInference(m *Model) bool {
	if m == nil {
		return false
	}
	if modelInferenceBackend(m) == model.BackendZerollamaRuntime {
		return true
	}
	if envconfig.RuntimeDefaultOn() && modelEligibleForRuntimeDefault(m) {
		return true
	}
	return false
}
