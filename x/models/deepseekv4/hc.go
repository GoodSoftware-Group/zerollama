package deepseekv4

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// HyperConn holds per-block hyper-connection mix tensors (attn or ffn).
// Why layout detect on Fn: ggml is {hc_dim, mix_dim}; mlx-lm is usually
// PyTorch [mix, hc*d]. Wrong transpose silently mixes streams.
type HyperConn struct {
	Fn    *mlx.Array // [mix_dim, hc*d] pytorch
	Base  *mlx.Array // [mix_dim]
	Scale *mlx.Array // [3] for block, [1] for head
}

type hcMix struct {
	post *mlx.Array // [B, L, hc]
	comb *mlx.Array // [B, L, hc, hc] PyTorch view; hcPost applies comb.T
}

func hcMixDim(hc int32) int32 {
	return (2 + hc) * hc
}

// sinkhorn matches HuggingFace DeepseekV4HyperConnection (this pack is mlx-lm,
// not GGUF). softmax(-1), then /sum(-2), then (iters-1) of /sum(-1), /sum(-2).
func sinkhorn(comb *mlx.Array, iters int32, eps float32) *mlx.Array {
	e := mlx.NewScalarArray(eps)
	comb = mlx.SoftmaxAxis(comb, -1, true)
	comb = mlx.Add(comb, e)
	comb = mlx.Div(comb, mlx.Add(mlx.Sum(comb, -2, true), e))
	for i := int32(1); i < iters; i++ {
		comb = mlx.Div(comb, mlx.Add(mlx.Sum(comb, -1, true), e))
		comb = mlx.Div(comb, mlx.Add(mlx.Sum(comb, -2, true), e))
	}
	return comb
}

func hcApplyFn(flat, fn *mlx.Array, inDim, outDim int32) *mlx.Array {
	// Why: mlx-lm [out,in] vs ggml [in,out] — see docs/mlx-deepseek-v4-flash-findings.md.
	d0, d1 := int32(fn.Dim(0)), int32(fn.Dim(1))
	switch {
	case d0 == outDim && d1 == inDim:
		return mlx.Matmul(flat, mlx.Transpose(fn, 1, 0))
	case d0 == inDim && d1 == outDim:
		return mlx.Matmul(flat, fn)
	default:
		return mlx.Matmul(flat, mlx.Transpose(fn, 1, 0))
	}
}

func hcAffine(x, scale, base *mlx.Array) *mlx.Array {
	return mlx.Add(mlx.Mul(x, scale), base)
}

// hcPre mixes hc streams into a single hidden [B,L,d] and returns post/comb for hcPost.
func hcPre(x *mlx.Array, hc *HyperConn, cfg *Config) (*mlx.Array, hcMix) {
	dims := x.Dims()
	B, L, nHC, d := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	flat := mlx.Reshape(x, B*L, nHC*d)
	flat = mlx.RMSNormFn(flat, nil, cfg.RMSNormEps)
	mixes := hcApplyFn(flat, hc.Fn, nHC*d, hcMixDim(nHC))
	mixes = mlx.Reshape(mixes, B, L, hcMixDim(nHC))

	scalePre := mlx.SliceStartStop(hc.Scale, []int32{0}, []int32{1})
	scalePost := scalePre
	if hc.Scale.Dims()[0] > 1 {
		scalePost = mlx.SliceStartStop(hc.Scale, []int32{1}, []int32{2})
	}
	basePre := mlx.SliceStartStop(hc.Base, []int32{0}, []int32{nHC})
	basePost := mlx.SliceStartStop(hc.Base, []int32{nHC}, []int32{2 * nHC})

	pre := mlx.SliceStartStop(mixes, []int32{0, 0, 0}, []int32{B, L, nHC})
	pre = hcAffine(pre, scalePre, basePre)
	pre = mlx.Add(mlx.Sigmoid(pre), mlx.NewScalarArray(cfg.HCEps))

	post := mlx.SliceStartStop(mixes, []int32{0, 0, nHC}, []int32{B, L, 2 * nHC})
	post = hcAffine(post, scalePost, basePost)
	post = mlx.MulScalar(mlx.Sigmoid(post), 2)

	var comb *mlx.Array
	if hc.Scale.Dims()[0] > 2 {
		scaleComb := mlx.SliceStartStop(hc.Scale, []int32{2}, []int32{3})
		baseComb := mlx.SliceStartStop(hc.Base, []int32{2 * nHC}, []int32{nHC * nHC + 2*nHC})
		comb = mlx.SliceStartStop(mixes, []int32{0, 0, 2 * nHC}, []int32{B, L, hcMixDim(nHC)})
		comb = hcAffine(comb, scaleComb, baseComb)
		comb = mlx.Reshape(comb, B, L, nHC, nHC)
		comb = sinkhorn(comb, cfg.HCSinkhornIters, cfg.HCEps)
	}

	// weighted sum of streams: [B,L,hc,d] * [B,L,hc,1]
	preExp := mlx.ExpandDims(pre, -1)
	mixed := mlx.Sum(mlx.Mul(x, preExp), 2, false)
	return mixed, hcMix{post: post, comb: comb}
}

func hcPost(delta, residual *mlx.Array, mix hcMix, _ *Config) *mlx.Array {
	dims := residual.Dims()
	B, L, nHC, d := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	post := mlx.ExpandDims(mix.post, -1) // [B,L,hc,1]
	out := mlx.Mul(mlx.ExpandDims(delta, 2), post)
	if mix.comb != nil {
		// HF/mlx-lm: comb.T @ residual (sum_j comb[j,k] * residual[j]).
		combT := mlx.Transpose(mix.comb, 0, 1, 3, 2)
		src := mlx.Reshape(residual, B, L, 1, nHC, d)
		w := mlx.Reshape(combT, B, L, nHC, nHC, 1)
		out = mlx.Add(out, mlx.Sum(mlx.Mul(src, w), 3, false))
	}
	return mlx.Reshape(out, B, L, nHC, d)
}

func hcHead(x *mlx.Array, hc *HyperConn, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L, nHC, d := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	flat := mlx.Reshape(x, B*L, nHC*d)
	flat = mlx.RMSNormFn(flat, nil, cfg.RMSNormEps)
	mixes := hcApplyFn(flat, hc.Fn, nHC*d, nHC)
	mixes = mlx.Reshape(mixes, B, L, nHC)
	pre := hcAffine(mixes, hc.Scale, hc.Base)
	pre = mlx.Add(mlx.Sigmoid(pre), mlx.NewScalarArray(cfg.HCEps))
	preExp := mlx.ExpandDims(pre, -1)
	return mlx.Sum(mlx.Mul(x, preExp), 2, false)
}

func broadcastHC(h *mlx.Array, hc int32) *mlx.Array {
	h = mlx.ExpandDims(h, 2)
	return mlx.Tile(h, []int32{1, 1, hc, 1})
}
