package discover

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/ollama/ollama/fs/ggml"
)

// PackANEMILWeightBlob wraps fp16 conv weights in the maderix BLOBFILE layout used by
// tools/ane-draft/draft_bench.m and ane-draft-daemon.
func PackANEMILWeightBlob(fp16Weights []byte) ([]byte, error) {
	if len(fp16Weights) == 0 || len(fp16Weights)%2 != 0 {
		return nil, fmt.Errorf("fp16 weights must be non-empty and 2-byte aligned")
	}
	wsize := len(fp16Weights)
	total := 64 + 64 + wsize
	buf := make([]byte, total)
	buf[0] = 0x01
	buf[4] = 0x02
	chunk := buf[64:]
	chunk[0] = 0xEF
	chunk[1] = 0xBE
	chunk[2] = 0xAD
	chunk[3] = 0xDE
	chunk[4] = 0x01
	chunk[10] = 0x08
	copy(chunk[64:], fp16Weights)
	return buf, nil
}

// ExtractTopLeftSquareFP16 reads the top-left channels×channels block from a GGUF
// weight matrix stored row-major. Supports F16 and F32 source tensors.
func ExtractTopLeftSquareFP16(raw []byte, tensor *ggml.Tensor, channels int) ([]byte, error) {
	if tensor == nil {
		return nil, fmt.Errorf("nil tensor")
	}
	if channels <= 0 {
		return nil, fmt.Errorf("channels must be positive")
	}
	if len(tensor.Shape) < 2 {
		return nil, fmt.Errorf("tensor %q needs rank-2 shape, got %v", tensor.Name, tensor.Shape)
	}

	rows := tensor.Shape[0]
	cols := tensor.Shape[1]
	if rows < uint64(channels) || cols < uint64(channels) {
		return nil, fmt.Errorf("tensor %q shape [%d,%d] smaller than %d×%d proxy", tensor.Name, rows, cols, channels, channels)
	}

	kind := ggml.TensorType(tensor.Kind)
	switch kind {
	case ggml.TensorTypeF16:
		return extractSquareF16(raw, int(cols), channels)
	case ggml.TensorTypeF32:
		return extractSquareF32(raw, int(cols), channels)
	case ggml.TensorTypeBF16:
		return extractSquareBF16(raw, int(cols), channels)
	default:
		return nil, fmt.Errorf("tensor %q kind %v unsupported for MIL extract (need f16/f32/bf16)", tensor.Name, kind)
	}
}

func extractSquareF16(raw []byte, cols, channels int) ([]byte, error) {
	need := channels * channels * 2
	out := make([]byte, need)
	rowBytes := cols * 2
	for r := 0; r < channels; r++ {
		srcOff := r * rowBytes
		if srcOff+channels*2 > len(raw) {
			return nil, fmt.Errorf("tensor data truncated at row %d", r)
		}
		dstOff := r * channels * 2
		copy(out[dstOff:dstOff+channels*2], raw[srcOff:srcOff+channels*2])
	}
	return out, nil
}

func extractSquareBF16(raw []byte, cols, channels int) ([]byte, error) {
	need := channels * channels * 2
	out := make([]byte, need)
	rowBytes := cols * 2
	for r := 0; r < channels; r++ {
		srcOff := r * rowBytes
		if srcOff+channels*2 > len(raw) {
			return nil, fmt.Errorf("tensor data truncated at row %d", r)
		}
		for c := 0; c < channels; c++ {
			h := binary.LittleEndian.Uint16(raw[srcOff+c*2:])
			f := bfloat16BitsToFloat32(h)
			binary.LittleEndian.PutUint16(out[(r*channels+c)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

func bfloat16BitsToFloat32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

func extractSquareF32(raw []byte, cols, channels int) ([]byte, error) {
	need := channels * channels * 2
	out := make([]byte, need)
	rowFloats := cols
	for r := 0; r < channels; r++ {
		for c := 0; c < channels; c++ {
			idx := r*rowFloats + c
			off := idx * 4
			if off+4 > len(raw) {
				return nil, fmt.Errorf("tensor data truncated at (%d,%d)", r, c)
			}
			f := math.Float32frombits(binary.LittleEndian.Uint32(raw[off : off+4]))
			binary.LittleEndian.PutUint16(out[(r*channels+c)*2:], float32ToFloat16Bits(f))
		}
	}
	return out, nil
}

func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint32((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	switch {
	case exp <= 0:
		if exp < -10 {
			return uint16(sign)
		}
		mant |= 0x800000
		shift := uint32(1 - exp)
		m := mant >> (shift + 13)
		if (mant>>(shift+12))&1 != 0 {
			m++
		}
		return uint16(sign | m)
	case exp >= 31:
		if mant != 0 {
			return uint16(sign | 0x7e00) // NaN
		}
		return uint16(sign | 0x7c00) // Inf
	default:
		return uint16(sign | (uint32(exp) << 10) | (mant >> 13))
	}
}

// DefaultProxyConvTensor picks the sidecar tensor used for lab conv proxy extract.
func DefaultProxyConvTensor() string {
	return "blk.0.ffn_gate.weight"
}
