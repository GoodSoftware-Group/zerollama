package create

import "github.com/ollama/ollama/types/model"

// SourceCapabilities infers manifest capabilities from HF / Laya source layout.
func SourceCapabilities(modelDir string) ([]string, error) {
	cfg, _, err := readSourceModelConfig(modelDir)
	if err != nil {
		return nil, err
	}
	switch cfg.Architecture() {
	case "ClefForDecision":
		caps := []string{string(model.CapabilityDecision)}
		if cfg.VisionConfig != nil || cfg.HasVision {
			caps = append(caps, "vision")
		}
		return caps, nil
	case "LayaForDecision", "StrandsDeciderForDecision":
		return []string{string(model.CapabilityDecision)}, nil
	case "EmbeddingGemma2Model":
		caps, _, _, err := EmbeddingManifestCaps(modelDir)
		return caps, err
	}
	// Nimble / Tev1 HF packs often look like Qwen3.5 with a decision head directory.
	if hasDecisionHeadLayout(modelDir) {
		return []string{string(model.CapabilityCompletion), string(model.CapabilityDecision)}, nil
	}
	return nil, nil
}
