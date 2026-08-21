package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type nemotronHImportTransform struct {
	numLayers int
}

func newNemotronHImportTransform(modelDir string, _ sourceModelConfig) (tensorImportTransform, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nemotronHImportTransform{}, nil //nolint:nilerr
	}
	var cfg struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
		LLMConfig       struct {
			NumHiddenLayers int `json:"num_hidden_layers"`
		} `json:"llm_config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("nemotron_h: parse config.json: %w", err)
	}
	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = cfg.LLMConfig.NumHiddenLayers
	}
	return nemotronHImportTransform{numLayers: numLayers}, nil
}

func (nemotronHImportTransform) skipTensor(string) bool { return false }

func (nemotronHImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	return []*safetensors.TensorData{td}, nil
}

func nemotronHIsUnsupportedModalityTensor(name string) bool {
	return strings.HasPrefix(name, "vision_model.") ||
		strings.HasPrefix(name, "mlp1.") ||
		strings.HasPrefix(name, "sound_encoder.") ||
		strings.HasPrefix(name, "sound_projection.")
}

func nemotronHShouldKeepBF16ForDirectNonAffine(name string) bool {
	switch {
	case strings.HasSuffix(name, ".mixer.gate.weight"):
		return true
	case strings.HasSuffix(name, ".mixer.conv1d.weight"):
		return true
	default:
		return false
	}
}

func nemotronHIsAttentionProjection(name string) bool {
	return strings.HasSuffix(name, ".mixer.q_proj.weight") ||
		strings.HasSuffix(name, ".mixer.k_proj.weight") ||
		strings.HasSuffix(name, ".mixer.v_proj.weight") ||
		strings.HasSuffix(name, ".mixer.o_proj.weight")
}

func (t nemotronHImportTransform) promoteSensitive(name string) bool {
	if nemotronHIsAttentionProjection(name) {
		return true
	}
	layerIdx := layerIndex(name)
	return layerIdx < 0 || useMoreBits(layerIdx, t.numLayers)
}

func (t nemotronHImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if nemotronHIsUnsupportedModalityTensor(name) || nemotronHShouldKeepBF16ForDirectNonAffine(name) {
		return ""
	}

	quantNorm := normalizeQuantType(quantize)

	if strings.HasSuffix(name, "embeddings.weight") || strings.HasSuffix(name, "lm_head.weight") {
		return promoteEmbedding(shape, quantNorm)
	}

	if quantNorm == "nvfp4" || quantNorm == "mxfp4" {
		isSensitive := nemotronHIsAttentionProjection(name) ||
			strings.HasSuffix(name, ".mixer.out_proj.weight") ||
			strings.HasSuffix(name, ".mixer.down_proj.weight") ||
			strings.Contains(name, ".mixer.experts.") && strings.HasSuffix(name, ".down_proj.weight") ||
			strings.HasSuffix(name, ".mixer.shared_experts.down_proj.weight")
		if isSensitive {
			if isAligned(shape, "mxfp8") && t.promoteSensitive(name) {
				return "mxfp8"
			}
			if isAligned(shape, quantNorm) {
				return quantNorm
			}
			return ""
		}
	}

	return GetTensorQuantization(name, shape, quantize)
}
