package server

import (
	"slices"

	"github.com/ollama/ollama/types/model"
)

// projectorArchitectures are vision/projector-only GGUF arch strings. They must not
// become the primary ModelFamily when a manifest also includes an LLM layer.
var projectorArchitectures = map[string]struct{}{
	"clip": {},
}

func isProjectorArchitecture(arch string) bool {
	_, ok := projectorArchitectures[arch]
	return ok
}

// llmFamilyPreference orders architectures when picking a primary family from
// ModelFamilies (e.g. clip + qwen35 on VL manifests).
var llmFamilyPreference = []string{
	"qwen35moe", "qwen35", "qwen3next", "qwen3vlmoe", "qwen3vl", "qwen3",
	"gemma4", "gemma3", "llama4", "llama3", "gptoss", "gpt-oss", "mistral3",
}

// PrimaryFamily returns the LLM architecture for routing renderers, parsers, and
// thinking defaults. Existing VL manifests may store ModelFamily=clip when a
// projector layer was processed first at create time.
func (m *Model) PrimaryFamily() string {
	if m == nil {
		return ""
	}
	return primaryModelFamily(m.Config)
}

func primaryModelFamily(cfg model.ConfigV2) string {
	for _, pref := range llmFamilyPreference {
		if slices.Contains(cfg.ModelFamilies, pref) {
			return pref
		}
	}
	if cfg.ModelFamily != "" && !isProjectorArchitecture(cfg.ModelFamily) {
		return cfg.ModelFamily
	}
	for _, f := range cfg.ModelFamilies {
		if !isProjectorArchitecture(f) {
			return f
		}
	}
	// Projector-only manifest (e.g. standalone clip mmproj) — no LLM family to route on.
	return ""
}

func defaultParserForFamily(m *Model) string {
	if m == nil {
		return ""
	}
	switch m.PrimaryFamily() {
	case "qwen35", "qwen35moe":
		return "qwen3.5"
	default:
		return ""
	}
}

func resolveParserName(m *Model) string {
	if m == nil {
		return ""
	}
	if m.Config.Parser != "" {
		return m.Config.Parser
	}
	return defaultParserForFamily(m)
}
