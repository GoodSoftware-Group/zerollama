package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

// qwen4ExpImportTransform quantizes Qwen4-Exp (QFN) PLE n-gram shards and
// keeps the low-rank linear-attention projections dense. Tensor names stay in
// the publisher layout the MLX loader reads (model.language_model.*).
type qwen4ExpImportTransform struct{}

func newQwen4ExpImportTransform(string, sourceModelConfig) (tensorImportTransform, error) {
	return qwen4ExpImportTransform{}, nil
}

func (qwen4ExpImportTransform) skipTensor(string) bool { return false }

func (qwen4ExpImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	return []*safetensors.TensorData{td}, nil
}

func (qwen4ExpImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// PLE is a hashed n-gram table, not the token embedding. It dominates the
	// checkpoint, and the lookup path dequantizes selected rows, so it takes
	// the requested quant type when the trailing axis is aligned. The generic
	// policy skips every name containing "embed".
	if qwen4ExpIsPLEEmbeddingShard(name) {
		quantNorm := normalizeQuantType(quantize)
		if quantNorm != "" && len(shape) == 2 && isAligned(shape, quantNorm) {
			return quantNorm
		}
		return ""
	}
	if strings.HasPrefix(name, "vision_tower.") || strings.HasPrefix(name, "model.visual.") {
		return ""
	}
	if qwen4ExpIsLowRankProjection(name) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func qwen4ExpIsPLEEmbeddingShard(name string) bool {
	return strings.Contains(name, ".ple.ple_embedding.ngram_embedding.shard_") &&
		strings.HasSuffix(name, ".weight")
}

func qwen4ExpIsLowRankProjection(name string) bool {
	switch {
	case strings.HasSuffix(name, ".linear_attn.in_proj_a.weight"):
		return true
	case strings.HasSuffix(name, ".linear_attn.in_proj_b.weight"):
		return true
	case strings.HasSuffix(name, ".linear_attn.in_proj_ba.weight"):
		return true
	default:
		return false
	}
}
