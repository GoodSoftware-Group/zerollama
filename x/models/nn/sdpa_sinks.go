package nn

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// scaledDotProductAttentionWithSinksRef implements attention with per-head sinks
// using explicit matmul+softmax. The MLX Metal fast SDPA+sinks kernel panics
// for GQA layouts (e.g. GPT-OSS 64Q/8KV); this path matches mlx_ref_attn.
func scaledDotProductAttentionWithSinksRef(q, k, v, sinks *mlx.Array, scale float32, d sdpaDispatch) *mlx.Array {
	if sinks == nil || !sinks.Valid() {
		panic("scaledDotProductAttentionWithSinksRef: sinks required")
	}
	if sinks.DType() != mlx.DTypeFloat32 {
		sinks = sinks.AsType(mlx.DTypeFloat32)
	}

	q = mlx.MulScalar(q, scale)

	B := int32(q.Dim(0))
	nQHeads := int32(q.Dim(1))
	L := int32(q.Dim(2))
	headDim := int32(q.Dim(3))
	nKVHeads := int32(k.Dim(1))
	kL := int32(k.Dim(2))

	if nQHeads%nKVHeads != 0 {
		panic("scaledDotProductAttentionWithSinksRef: n_q_heads must divide n_kv_heads")
	}
	nRepeats := nQHeads / nKVHeads

	if nRepeats > 1 {
		q = mlx.Reshape(q, B, nKVHeads, nRepeats, L, headDim)
		k = mlx.ExpandDims(k, 2)
		v = mlx.ExpandDims(v, 2)
	}

	var kT *mlx.Array
	if nRepeats > 1 {
		kT = mlx.Transpose(k, 0, 1, 2, 4, 3)
	} else {
		kT = mlx.Transpose(k, 0, 1, 3, 2)
	}
	scores := mlx.Matmul(q, kT)

	switch d.mode {
	case "causal":
		offset := int(kL - L)
		qCol := mlx.FromValues(int32Range(int(L), offset), int(L), 1)
		kRow := mlx.FromValues(int32Range(int(kL), 0), 1, int(kL))
		mask := kRow.LessEqual(qCol)
		if nRepeats > 1 {
			mask = mlx.Reshape(mask, 1, 1, 1, L, kL)
		} else {
			mask = mlx.Reshape(mask, 1, 1, L, kL)
		}
		scores = mlx.Where(mask, scores, mlx.FromValue(-math.MaxFloat32))
	case "array":
		if d.arr != nil && d.arr.Valid() {
			scores = mlx.Add(scores, d.arr)
		}
	}

	scoreDims := scores.Dims()
	if len(scoreDims) == 0 {
		panic("scaledDotProductAttentionWithSinksRef: matmul produced empty scores")
	}
	sinkShape := make([]int32, len(scoreDims))
	for i, d := range scoreDims {
		sinkShape[i] = int32(d)
	}
	sinkShape[len(sinkShape)-1] = 1
	if nRepeats > 1 {
		sinks = mlx.Reshape(sinks, 1, nKVHeads, nRepeats, 1, 1)
	} else {
		sinks = mlx.Reshape(sinks, 1, nQHeads, 1, 1)
	}
	sinks = mlx.BroadcastTo(sinks, sinkShape...)
	scores = mlx.Concatenate([]*mlx.Array{sinks, scores}, -1)
	scores = mlx.SoftmaxAxis(scores, -1, true)
	scores = sliceLastAxisFrom(scores, 1)

	out := mlx.Matmul(scores, v)
	if nRepeats > 1 {
		out = mlx.Reshape(out, B, nQHeads, L, headDim)
	}
	return out
}

func int32Range(n, offset int) []int32 {
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(i + offset)
	}
	return out
}

func sliceLastAxisFrom(a *mlx.Array, start int) *mlx.Array {
	switch len(a.Dims()) {
	case 4:
		return a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(start, mlx.End))
	case 5:
		return a.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(), mlx.Slice(start, mlx.End))
	default:
		panic("sliceLastAxisFrom: unsupported rank")
	}
}
