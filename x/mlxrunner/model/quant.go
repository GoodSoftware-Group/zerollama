package model

import (
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// QuantizationParams returns default groupSize, bits, and mode for a quantization type.
func QuantizationParams(quantization string) (groupSize, bits int, mode string) {
	switch strings.ToUpper(quantization) {
	case "NVFP4":
		return 16, 4, "nvfp4"
	case "MXFP4":
		return 32, 4, "mxfp4"
	case "FP4", "Q4", "INT4":
		return 64, 4, "affine"
	case "MXFP8":
		return 32, 8, "mxfp8"
	case "FP8", "Q8", "INT8":
		return 64, 8, "affine"
	case "FP6", "Q6", "INT6":
		return 64, 6, "affine"
	case "FP3", "Q3", "INT3":
		return 64, 3, "affine"
	case "FP2", "Q2", "INT2":
		// mlx-lm 2-bit exports commonly use group_size 128 (e.g. bonsai).
		return 128, 2, "affine"
	case "":
		return 0, 0, ""
	default:
		return 32, 8, "affine"
	}
}

// TensorQuantParams resolves quant params for a tensor using per-tensor metadata
// when available, otherwise falling back to the provided model defaults.
func TensorQuantParams(
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
	tensorName string,
) (groupSize, bits int, mode string, fromTensor bool) {
	if tensorQuant != nil {
		if tq := tensorQuant[tensorName]; tq != nil {
			groupSize, bits, mode = QuantizationParams(tq.QuantType)
			if tq.GroupSize > 0 {
				groupSize = tq.GroupSize
			}
			return groupSize, bits, mode, true
		}
	}
	return defaultGroupSize, defaultBits, defaultMode, false
}

// ResolveLinearQuantParams resolves quantization params for a quantized linear
// tensor, preferring per-tensor metadata and falling back to shape-based
// inference for affine packed tensors.
func ResolveLinearQuantParams(
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
	tensorName string,
	weight, scales *mlx.Array,
) (groupSize, bits int, mode string) {
	// config.json quantization that matches packed shapes disambiguates
	// 2-bit/gs128 vs 4-bit/gs64 (identical packed column counts). Prefer
	// these model defaults over ambiguous/wrong blob TensorQuant guesses.
	if defaultBits > 0 && defaultGroupSize > 0 {
		dm := defaultMode
		if dm == "" {
			dm = "affine"
		}
		if dm == "affine" && affinePackedShapeMatches(weight, scales, defaultGroupSize, defaultBits) {
			return defaultGroupSize, defaultBits, dm
		}
	}

	groupSize, bits, mode, fromTensor := TensorQuantParams(
		defaultGroupSize,
		defaultBits,
		defaultMode,
		tensorQuant,
		tensorName,
	)

	// Per-tensor metadata (OptIQ mixed 4/6/8) when defaults do not pack.
	if fromTensor && mode == "affine" && bits > 0 && groupSize > 0 &&
		affinePackedShapeMatches(weight, scales, groupSize, bits) {
		return groupSize, bits, mode
	}

	// mlx-lm / ollama create often ships packed weight+scale without blob
	// quant metadata. Treat companion scales as affine and infer (gs, bits)
	// from shapes — otherwise QuantizedMatmul runs with bits=0 and returns
	// an invalid array (BOOL/empty), which then panics inside compiled SwiGLU.
	if mode == "" && weight != nil && scales != nil {
		mode = "affine"
	}

	if mode == "affine" {
		if inferredGroupSize, inferredBits, ok := InferAffineQuantParamsFromShapes(weight, scales, bits); ok {
			if !fromTensor || groupSize == 0 || bits == 0 || !affinePackedShapeMatches(weight, scales, groupSize, bits) {
				groupSize = inferredGroupSize
				bits = inferredBits
			}
		}
	}

	return groupSize, bits, mode
}

// affinePackedShapeMatches reports whether weight/scales shapes match the
// packed column count implied by (groupSize, bits).
func affinePackedShapeMatches(weight, scales *mlx.Array, groupSize, bits int) bool {
	if weight == nil || scales == nil || groupSize <= 0 || bits <= 0 {
		return false
	}
	weightShape := weight.Dims()
	scaleShape := scales.Dims()
	if len(weightShape) == 0 || len(scaleShape) == 0 {
		return false
	}
	weightCols := int(weightShape[len(weightShape)-1])
	scalesCols := int(scaleShape[len(scaleShape)-1])
	if weightCols <= 0 || scalesCols <= 0 {
		return false
	}
	inFeatures := scalesCols * groupSize
	expectedCols := inFeatures * bits / 32
	return weightCols == expectedCols
}

// InferAffineQuantParamsFromShapes infers (groupSize,bits) for affine quantized
// tensors from packed weight and scale shapes.
func InferAffineQuantParamsFromShapes(weight, scales *mlx.Array, hintBits int) (groupSize, bits int, ok bool) {
	if weight == nil || scales == nil {
		return 0, 0, false
	}

	weightShape := weight.Dims()
	scaleShape := scales.Dims()
	if len(weightShape) == 0 || len(scaleShape) == 0 {
		return 0, 0, false
	}

	weightCols := weightShape[len(weightShape)-1]
	scalesCols := scaleShape[len(scaleShape)-1]
	if weightCols <= 0 || scalesCols <= 0 {
		return 0, 0, false
	}

	type pair struct{ gs, bits int }
	var matches []pair
	for _, gs := range []int{128, 64, 32, 16} {
		inFeatures := scalesCols * gs
		if inFeatures <= 0 {
			continue
		}
		for _, b := range []int{2, 3, 4, 6, 8} {
			if hintBits > 0 && b != hintBits {
				continue
			}
			if weightCols == inFeatures*b/32 {
				matches = append(matches, pair{gs, b})
			}
		}
	}
	if len(matches) == 1 {
		return matches[0].gs, matches[0].bits, true
	}
	if hintBits > 0 {
		for _, m := range matches {
			if m.bits == hintBits {
				return m.gs, m.bits, true
			}
		}
	}

	groupSize4 := weightCols * 8 / scalesCols
	groupSize8 := weightCols * 4 / scalesCols

	switch {
	case groupSize4 == 32:
		return 32, 4, true
	case groupSize8 == 64:
		return 64, 8, true
	}

	if isCommonGroupSize(groupSize4) && !isCommonGroupSize(groupSize8) {
		return groupSize4, 4, true
	}
	if isCommonGroupSize(groupSize8) && !isCommonGroupSize(groupSize4) {
		return groupSize8, 8, true
	}

	// Ambiguous (e.g. 2/128 vs 4/64) — caller must supply config defaults.
	return 0, 0, false
}

func isCommonGroupSize(v int) bool {
	switch v {
	case 16, 32, 64, 128:
		return true
	default:
		return false
	}
}
