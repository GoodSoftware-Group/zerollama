package nn

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Decode-only quantized copy of dense (bf16/fp16/fp32) attention projections.
// Prefill and speculative fused forwards (L>1) keep the dense weights;
// single-token decode uses affine 4-bit (mlx-serve --decode-attn-quant).
const (
	decodeQuantGroup = 64
	decodeQuantBits  = 4
	decodeQuantMode  = "affine"
)

// DecodeQuantLinear is a dense linear plus a 4-bit copy used when the token
// axis is 1. Quantized checkpoints never wrap — they already are QuantizedLinear.
type DecodeQuantLinear struct {
	Dense *Linear
	Quant *QuantizedLinear
}

func canDecodeQuantWeight(w *mlx.Array) bool {
	if w == nil || !w.Valid() || w.NumDims() != 2 {
		return false
	}
	switch w.DType() {
	case mlx.DTypeFloat32, mlx.DTypeFloat16, mlx.DTypeBFloat16:
	default:
		return false
	}
	in := w.Dim(1)
	return in >= decodeQuantGroup && in%decodeQuantGroup == 0
}

// QuantizeLinearLayer replaces a dense (or decode-quant dual) projection with
// affine 4-bit. Used for draft heads: the target still verifies, so quality is
// decided by the main model (mlx-serve load-time draft quant). Already-quantized
// layers are left alone. Ineligible shapes stay dense.
func QuantizeLinearLayer(layer LinearLayer) LinearLayer {
	if layer == nil {
		return nil
	}
	switch l := layer.(type) {
	case *QuantizedLinear:
		return l
	case *DecodeQuantLinear:
		if l.Quant != nil {
			return l.Quant
		}
		return QuantizeLinearLayer(l.Dense)
	case *Linear:
		if !canDecodeQuantWeight(l.Weight) {
			return l
		}
		return NewQuantizedLinear(l.Weight, l.Bias, decodeQuantGroup, decodeQuantBits, decodeQuantMode)
	default:
		return layer
	}
}

// WrapDecodeQuant returns a decode-quant wrapper, or dense unchanged when the
// weight shape/dtype cannot be 4-bit grouped.
func WrapDecodeQuant(dense *Linear) LinearLayer {
	if dense == nil || !canDecodeQuantWeight(dense.Weight) {
		return dense
	}
	return &DecodeQuantLinear{
		Dense: dense,
		Quant: NewQuantizedLinear(dense.Weight, dense.Bias, decodeQuantGroup, decodeQuantBits, decodeQuantMode),
	}
}

func isDecodeActivation(x *mlx.Array) bool {
	if x == nil || !x.Valid() {
		return false
	}
	switch x.NumDims() {
	case 3:
		return x.Dim(1) == 1
	case 2:
		return x.Dim(0) == 1
	default:
		return false
	}
}

func (d *DecodeQuantLinear) Forward(x *mlx.Array) *mlx.Array {
	if d.Quant != nil && isDecodeActivation(x) {
		return d.Quant.Forward(x)
	}
	return d.Dense.Forward(x)
}

func (d *DecodeQuantLinear) OutputDim() int32 {
	return d.Dense.OutputDim()
}
