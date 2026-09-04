package deepseekv4

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

type ropePack struct {
	freqs  *mlx.Array
	mscale float32
	base   float32
}

func buildRope(cfg *Config, compress bool) ropePack {
	// Why yarn only when compress: llama.cpp ext_factor=0 on ratio-0 layers.
	base := cfg.RopeTheta
	if compress && cfg.CompressRopeTheta > 0 {
		base = cfg.CompressRopeTheta
	}
	p := ropePack{base: base, mscale: 1}
	if compress && cfg.RopeScaling != nil && cfg.RopeScaling.TypeName() == "yarn" {
		p.freqs, _ = nn.BuildYarnRopeFreqs(int(cfg.QKRopeHeadDim), base, cfg.RopeScaling)
		// Why mscale stays 1: this checkpoint is mlx-lm. HF DeepSeek-V4 sets
		// attention_factor=1 (does not multiply cos/sin by YaRN mscale or by
		// llama.cpp dsv4_rope_attn_factor). Scaling rotary amplitude was the
		// fill-in-the-blank / UnUnUn collapse on the linked pack.
		p.mscale = 1
	}
	return p
}

// compressedWritePos is llama.cpp state_write_pos: source_start = k*ratio
// when a block completes at pos (k+1)*ratio-1.
func compressedWritePos(n, ratio int32) []int32 {
	if n <= 0 {
		return nil
	}
	if ratio <= 0 {
		ratio = 1
	}
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(i) * ratio
	}
	return out
}

func tokenPositions(offsets *mlx.Array, L int32) *mlx.Array {
	if L <= 0 {
		return mlx.Zeros(mlx.DTypeInt32, int(offsets.Dim(0)), 0)
	}
	seq := make([]int32, L)
	for i := range seq {
		seq[i] = int32(i)
	}
	t := mlx.FromValues(seq, int(L))
	// [B,1] + [1,L] → [B,L]
	return mlx.Add(mlx.ExpandDims(offsets, 1), mlx.ExpandDims(t, 0))
}

func applyRoPE(x, offsets *mlx.Array, cfg *Config, pack ropePack, inverse bool) *mlx.Array {
	// Why NORM not NEOX: llama.cpp LLM_ARCH_DEEPSEEK4 is LLAMA_ROPE_TYPE_NORM
	// (consecutive pairs). Offset n_nope still means the last qk_rope_head_dim.
	// NEOX half-split on those 64 dims is a different function and dumps
	// number-salad logits on a real pack.
	return applyNormRoPELast(x, offsets, cfg, pack, inverse)
}

// applyNormRoPELast rotates the last qk_rope_head_dim dims as GPT-J pairs.
func applyNormRoPELast(x, offsets *mlx.Array, cfg *Config, pack ropePack, inverse bool) *mlx.Array {
	ropeDim := int(cfg.QKRopeHeadDim)
	dims := x.Dims()
	last := dims[len(dims)-1]
	L := int32(dims[len(dims)-2])
	if ropeDim <= 0 || ropeDim > last || ropeDim%2 != 0 {
		return x
	}
	start := make([]int32, len(dims))
	stopNope := make([]int32, len(dims))
	stopAll := make([]int32, len(dims))
	startRot := make([]int32, len(dims))
	for i, d := range dims {
		stopNope[i] = int32(d)
		stopAll[i] = int32(d)
	}
	stopNope[len(dims)-1] = int32(last - ropeDim)
	startRot[len(dims)-1] = int32(last - ropeDim)
	nope := mlx.SliceStartStop(x, start, stopNope)
	rot := mlx.SliceStartStop(x, startRot, stopAll)

	var pos *mlx.Array
	if offsets != nil && offsets.NumDims() >= 2 {
		// Absolute [B, L] (compressed write-pos is stride ratio, not 0..L-1).
		pos = offsets.AsType(mlx.DTypeFloat32)
	} else {
		pos = tokenPositions(offsets, L).AsType(mlx.DTypeFloat32)
	}
	for len(pos.Dims()) < len(dims)-1 {
		pos = mlx.ExpandDims(pos, 1)
	}
	half := ropeDim / 2
	var freq *mlx.Array
	if pack.freqs != nil {
		freq = pack.freqs
	} else {
		vals := make([]float32, half)
		for i := range vals {
			vals[i] = float32(math.Pow(float64(pack.base), float64(2*i)/float64(ropeDim)))
		}
		freq = mlx.FromValues(vals, half)
	}
	inv := mlx.Div(mlx.NewScalarArray(1), freq)
	theta := mlx.Mul(mlx.ExpandDims(pos, -1), inv)
	cos := mlx.Cos(theta)
	sin := mlx.Sin(theta)
	if inverse {
		sin = mlx.Neg(sin)
	}
	if pack.mscale != 1 {
		cos = mlx.MulScalar(cos, pack.mscale)
		sin = mlx.MulScalar(sin, pack.mscale)
	}

	pairShape := make([]int32, len(rot.Dims())+1)
	for i, d := range rot.Dims() {
		pairShape[i] = int32(d)
	}
	pairShape[len(rot.Dims())-1] = int32(half)
	pairShape[len(rot.Dims())] = 2
	paired := mlx.Reshape(rot, pairShape...)
	x0 := mlx.Squeeze(mlx.SliceStartStop(paired, zerosRank(paired), lastAxisStop(paired, 1)), -1)
	x1start := zerosRank(paired)
	x1start[len(x1start)-1] = 1
	x1 := mlx.Squeeze(mlx.SliceStartStop(paired, x1start, lastAxisStop(paired, 2)), -1)

	o0 := mlx.Sub(mlx.Mul(x0, cos), mlx.Mul(x1, sin))
	o1 := mlx.Add(mlx.Mul(x0, sin), mlx.Mul(x1, cos))
	rotOut := mlx.Reshape(mlx.Concatenate([]*mlx.Array{mlx.ExpandDims(o0, -1), mlx.ExpandDims(o1, -1)}, -1), shapeI32(rot)...)
	return mlx.Concatenate([]*mlx.Array{nope, rotOut}, -1)
}

func shapeI32(a *mlx.Array) []int32 {
	d := a.Dims()
	out := make([]int32, len(d))
	for i, v := range d {
		out[i] = int32(v)
	}
	return out
}

func zerosRank(a *mlx.Array) []int32 {
	return make([]int32, a.NumDims())
}

func lastAxisStop(a *mlx.Array, last int32) []int32 {
	out := make([]int32, a.NumDims())
	for i := range out {
		out[i] = int32(a.Dim(i))
	}
	out[len(out)-1] = last
	return out
}
