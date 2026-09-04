package model

import (
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// LinearFactory builds linear layers using shared tensor maps and quant defaults.
type LinearFactory struct {
	tensors          map[string]*mlx.Array
	defaultGroupSize int
	defaultBits      int
	defaultMode      string
	tensorQuant      map[string]*TensorQuantInfo
}

// NewLinearFactory creates a reusable constructor for model linear layers.
func NewLinearFactory(
	tensors map[string]*mlx.Array,
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
) LinearFactory {
	return LinearFactory{
		tensors:          tensors,
		defaultGroupSize: defaultGroupSize,
		defaultBits:      defaultBits,
		defaultMode:      defaultMode,
		tensorQuant:      tensorQuant,
	}
}

// Make constructs a linear layer at path.
func (f LinearFactory) Make(path string) nn.LinearLayer {
	return MakeLinearLayer(
		f.tensors,
		path,
		f.defaultGroupSize,
		f.defaultBits,
		f.defaultMode,
		f.tensorQuant,
	)
}

// MakeLinearLayer constructs a linear layer from a tensor map.
//
// For quantized tensors (path.weight + path.weight_scale), it resolves per-tensor
// quant params via TensorQuant metadata (with shape-based affine fallback).
// For non-quantized tensors, it returns a standard nn.Linear.
func MakeLinearLayer(
	tensors map[string]*mlx.Array,
	path string,
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
) nn.LinearLayer {
	w := tensors[path+".weight"]
	if w == nil {
		return nil
	}

	scales := tensors[path+".weight_scale"]
	if scales == nil {
		scales = tensors[path+".scales"]
	}
	if scales != nil {
		qbiases := tensors[path+".weight_qbias"]
		if qbiases == nil {
			// Alternate mlx-lm export name for affine zero-points.
			qbiases = tensors[path+".biases"]
		}
		bias := tensors[path+".bias"]

		groupSize, bits, mode := ResolveLinearQuantParams(
			defaultGroupSize,
			defaultBits,
			defaultMode,
			tensorQuant,
			path+".weight",
			w,
			scales,
		)

		// Check for per-tensor global scale (NVIDIA double-scale nvfp4).
		// NVIDIA ModelOpt stores this as "weight_scale_2"; our import
		// pipeline maps it to "weight.global_scale".
		globalScale := tensors[path+".weight.global_scale"]
		if globalScale == nil {
			globalScale = tensors[path+".weight_scale_2"]
		}

		return &nn.QuantizedLinear{
			Weight:      w,
			Scales:      scales,
			QBiases:     qbiases,
			Bias:        bias,
			GlobalScale: globalScale,
			GroupSize:   groupSize,
			Bits:        bits,
			Mode:        mode,
		}
	}

	bias := tensors[path+".bias"]
	dense := nn.NewLinear(w, bias)
	if isAttentionProjection(path) {
		return nn.WrapDecodeQuant(dense)
	}
	return dense
}

func isAttentionProjection(path string) bool {
	p := strings.ToLower(strings.ReplaceAll(path, "/", "."))
	base := p
	if i := strings.LastIndex(p, "."); i >= 0 {
		base = p[i:]
	} else {
		base = "." + p
	}
	switch base {
	case ".q_proj", ".k_proj", ".v_proj", ".o_proj", ".out_proj",
		".qkv_proj", ".wqkv", ".wq", ".wk", ".wv", ".wo",
		".query_proj", ".key_proj", ".value_proj",
		".q_a_proj", ".q_b_proj", ".kv_a_proj", ".kv_b_proj",
		".wq_a", ".wq_b", ".wkv", ".wo_b":
		// why not .wo_a: DSv4 grouped 3-D out-proj; decode-quant wrap is 2-D only.
		return true
	default:
		return false
	}
}
