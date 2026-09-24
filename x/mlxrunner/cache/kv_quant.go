package cache

import (
	"os"
	"strings"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Live KV affine quant (mlx-serve --kv-quant). Storage is group-wise affine
// via mlx_quantize; SDPA always reads dense via denseView. No fused packed
// attention yet — long-ctx decode may regress vs dense until matmul2d lands.
// Idle trie snapshots still use packOwnedKV (FP8) after denseView.

const kvQuantGroupSize = 64

// Scheme is the live KV storage backend.
type Scheme int

const (
	SchemeOff Scheme = iota
	SchemeAffine
)

// KVQuantConfig mirrors mlx-serve kv_quant.KVQuantConfig.
type KVQuantConfig struct {
	Scheme    Scheme
	Bits      int
	GroupSize int
}

// DenseKVQuantConfig is the default: live bf16/f16/f32 buffers.
func DenseKVQuantConfig() KVQuantConfig {
	return KVQuantConfig{Scheme: SchemeOff}
}

// AffineKVQuant returns affine config for bits 4 or 8 (group size 64).
func AffineKVQuant(bits int) KVQuantConfig {
	if bits != 4 && bits != 8 {
		return DenseKVQuantConfig()
	}
	return KVQuantConfig{Scheme: SchemeAffine, Bits: bits, GroupSize: kvQuantGroupSize}
}

func (c KVQuantConfig) IsAffine() bool { return c.Scheme == SchemeAffine }

func (c KVQuantConfig) WireName() string {
	switch {
	case !c.IsAffine():
		return "off"
	case c.Bits == 4:
		return "4"
	default:
		return "8"
	}
}

// KVQuantFromEnv reads ZEROLLAMA_MLX_KV_QUANT: off|0|4|8 (default off).
func KVQuantFromEnv() KVQuantConfig {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_KV_QUANT"))) {
	case "4":
		return AffineKVQuant(4)
	case "8":
		return AffineKVQuant(8)
	default:
		return DenseKVQuantConfig()
	}
}

// QuantizedKV is one affine (q, scales, biases) triple.
type QuantizedKV struct {
	Q, Scales, Biases *mlx.Array
}

func quantizeAffine(dense *mlx.Array, cfg KVQuantConfig) QuantizedKV {
	q, scales, biases := mlx.Quantize(dense, cfg.GroupSize, cfg.Bits, "affine")
	return QuantizedKV{Q: q, Scales: scales, Biases: biases}
}

func dequantizeAffine(q, scales, biases *mlx.Array, cfg KVQuantConfig, elem mlx.DType) *mlx.Array {
	out := mlx.Dequantize(q, scales, biases, cfg.GroupSize, cfg.Bits, "affine", nil)
	if out.DType() != elem {
		out = out.AsType(elem)
	}
	return out
}

// affineBuffers holds the six live arrays for one KVCache / RotatingKVCache.
type affineBuffers struct {
	kq, kScales, kBiases *mlx.Array
	vq, vScales, vBiases *mlx.Array
}

func (a *affineBuffers) all() []*mlx.Array {
	return []*mlx.Array{a.kq, a.kScales, a.kBiases, a.vq, a.vScales, a.vBiases}
}

func (a *affineBuffers) pin() {
	mlx.Pin(a.kq, a.kScales, a.kBiases, a.vq, a.vScales, a.vBiases)
}

func (a *affineBuffers) unpin() {
	mlx.Unpin(a.kq, a.kScales, a.kBiases, a.vq, a.vScales, a.vBiases)
}

func (a *affineBuffers) clear() {
	a.kq, a.kScales, a.kBiases = nil, nil, nil
	a.vq, a.vScales, a.vBiases = nil, nil, nil
}

func (a *affineBuffers) initialized() bool { return a.kq != nil }

func (a *affineBuffers) seqCap() int {
	if a.kq == nil {
		return 0
	}
	return a.kq.Dim(2)
}

func (a *affineBuffers) denseSlice(cfg KVQuantConfig, elem mlx.DType, start, end int) (k, v *mlx.Array) {
	kq := a.kq.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	ks := a.kScales.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	kb := a.kBiases.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	vq := a.vq.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	vs := a.vScales.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	vb := a.vBiases.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	return dequantizeAffine(kq, ks, kb, cfg, elem), dequantizeAffine(vq, vs, vb, cfg, elem)
}

func (a *affineBuffers) sliceSeq(start, end int) affineBuffers {
	return affineBuffers{
		kq:      a.kq.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()),
		kScales: a.kScales.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()),
		kBiases: a.kBiases.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()),
		vq:      a.vq.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()),
		vScales: a.vScales.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()),
		vBiases: a.vBiases.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()),
	}
}

func (a *affineBuffers) setSliceSeq(other affineBuffers) {
	a.kq.Set(other.kq)
	a.kScales.Set(other.kScales)
	a.kBiases.Set(other.kBiases)
	a.vq.Set(other.vq)
	a.vScales.Set(other.vScales)
	a.vBiases.Set(other.vBiases)
}

func (a *affineBuffers) concatenate(axis int, other affineBuffers) {
	a.kq.Set(a.kq.Concatenate(axis, other.kq))
	a.kScales.Set(a.kScales.Concatenate(axis, other.kScales))
	a.kBiases.Set(a.kBiases.Concatenate(axis, other.kBiases))
	a.vq.Set(a.vq.Concatenate(axis, other.vq))
	a.vScales.Set(a.vScales.Concatenate(axis, other.vScales))
	a.vBiases.Set(a.vBiases.Concatenate(axis, other.vBiases))
}

func growAffineZeros(B, H, steps, qLast, vqLast, scLast, vscLast int) affineBuffers {
	return affineBuffers{
		kq:      mlx.Zeros(mlx.DTypeUint32, B, H, steps, qLast),
		vq:      mlx.Zeros(mlx.DTypeUint32, B, H, steps, vqLast),
		kScales: mlx.Zeros(mlx.DTypeBFloat16, B, H, steps, scLast),
		kBiases: mlx.Zeros(mlx.DTypeBFloat16, B, H, steps, scLast),
		vScales: mlx.Zeros(mlx.DTypeBFloat16, B, H, steps, vscLast),
		vBiases: mlx.Zeros(mlx.DTypeBFloat16, B, H, steps, vscLast),
	}
}

func (a *affineBuffers) writeAt(prev, end int, kq, vq QuantizedKV) {
	a.kq.Set(a.kq.SliceUpdate(kq.Q, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, end), mlx.Slice()))
	a.kScales.Set(a.kScales.SliceUpdate(kq.Scales, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, end), mlx.Slice()))
	a.kBiases.Set(a.kBiases.SliceUpdate(kq.Biases, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, end), mlx.Slice()))
	a.vq.Set(a.vq.SliceUpdate(vq.Q, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, end), mlx.Slice()))
	a.vScales.Set(a.vScales.SliceUpdate(vq.Scales, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, end), mlx.Slice()))
	a.vBiases.Set(a.vBiases.SliceUpdate(vq.Biases, mlx.Slice(), mlx.Slice(), mlx.Slice(prev, end), mlx.Slice()))
}

func headDimOK(d, groupSize int) bool {
	return groupSize > 0 && d%groupSize == 0
}
