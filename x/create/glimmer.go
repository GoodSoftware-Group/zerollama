package create

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type glimmerImportTransform struct {
	numLayers int
}

type glimmerConfig struct {
	NumHiddenLayers int `json:"num_hidden_layers"`
	TextConfig      struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
	} `json:"text_config"`
}

func newGlimmerImportTransform(modelDir string, _ sourceModelConfig) (tensorImportTransform, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return glimmerImportTransform{}, nil //nolint:nilerr
	}
	var cfg glimmerConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("glimmer: parse config.json: %w", err)
	}
	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = cfg.TextConfig.NumHiddenLayers
	}
	return glimmerImportTransform{numLayers: numLayers}, nil
}

func (glimmerImportTransform) skipTensor(string) bool { return false }

func (glimmerImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	return []*safetensors.TensorData{td}, nil
}

func (t glimmerImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if isGlimmerVisionTensor(name) {
		return ""
	}

	base := normalizeQuantType(quantize)
	if isEmbedTokensWeight(name) {
		if e := promoteEmbedding(shape, base); e != "" {
			return e
		}
		if isAligned(shape, base) {
			return base
		}
		return ""
	}

	if isGlimmerSensitiveProjection(name) && eightBit(base) != base {
		return sensitiveType(t.promoteSensitive(name), shape, base)
	}

	return GetTensorQuantization(name, shape, quantize)
}

func isGlimmerVisionTensor(name string) bool {
	return isVision(name)
}

func isGlimmerSensitiveProjection(name string) bool {
	return strings.HasSuffix(name, ".self_attn.v_proj.weight") ||
		strings.HasSuffix(name, ".self_attn.k_proj.weight") ||
		strings.HasSuffix(name, ".self_attn.o_proj.weight") ||
		strings.HasSuffix(name, ".mlp.down_proj.weight") ||
		(strings.Contains(name, ".mlp.experts.") && strings.HasSuffix(name, ".down_proj.weight"))
}

func (t glimmerImportTransform) promoteSensitive(name string) bool {
	layerIdx := layerIndex(name)
	return layerIdx < 0 || useMoreBits(layerIdx, t.numLayers)
}
