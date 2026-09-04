package deepseekv4

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Compressor is the Flash CSA/HCA (and indexer) token compressor.
//
// Why overlap (CSA): llama.cpp build_overlap_compressed_kv_from_state mixes
// prev-window first half with cur-window second half of the 2×head_dim
// projection (coff=2). HCA is softmax over `ratio` tokens (coff=1).
// Why keep tokKV on the struct: mlxrunner KVCache only stores attention K/V;
// compressor write-pos maps are not ported yet.
type Compressor struct {
	WKV, WGate nn.LinearLayer
	APE        *mlx.Array
	Norm       *nn.RMSNorm
	Ratio      int32
	Overlap    bool // CSA: coff=2 overlap windows; HCA: coff=1 softmax over ratio
	HeadDim    int32
	tokKV      *mlx.Array
	tokScore   *mlx.Array
	compKV     *mlx.Array
}

func loadCompressor(tensors map[string]*mlx.Array, linears modelLinear, prefix string, ratio, headDim int32, overlap bool) *Compressor {
	wkv := linears.Make(prefix + ".wkv")
	wgate := linears.Make(prefix + ".wgate")
	if wkv == nil || wgate == nil {
		return nil
	}
	c := &Compressor{WKV: wkv, WGate: wgate, Ratio: ratio, Overlap: overlap, HeadDim: headDim}
	if a := tensors[prefix+".ape"]; a != nil {
		c.APE = normalizeAPE(a, ratio)
	}
	if w := tensors[prefix+".norm.weight"]; w != nil {
		c.Norm = nn.NewRMSNorm(w, 0)
	}
	return c
}

type modelLinear interface {
	Make(path string) nn.LinearLayer
}

func normalizeAPE(ape *mlx.Array, ratio int32) *mlx.Array {
	if ape.NumDims() != 2 {
		return ape
	}
	if int32(ape.Dim(0)) == ratio {
		return ape
	}
	if int32(ape.Dim(1)) == ratio {
		return mlx.Transpose(ape, 1, 0)
	}
	return ape
}

func (c *Compressor) reset() {
	c.tokKV, c.tokScore, c.compKV = nil, nil, nil
}

func (c *Compressor) append(x, positions *mlx.Array, cfg *Config) *mlx.Array {
	kv := c.WKV.Forward(x)
	score := c.WGate.Forward(x)
	if c.APE != nil {
		idx := posMod(positions, c.Ratio)
		flat := mlx.Reshape(idx, int32(idx.Dim(0))*int32(idx.Dim(1)))
		ape := mlx.Take(c.APE, flat, 0)
		ape = mlx.Reshape(ape, int32(x.Dim(0)), int32(x.Dim(1)), int32(c.APE.Dim(1)))
		score = mlx.Add(score, ape)
	}
	c.tokKV = catTime(c.tokKV, kv)
	c.tokScore = catTime(c.tokScore, score)
	T := c.tokKV.Dim(1)
	want := T / int(c.Ratio)
	have := 0
	if c.compKV != nil {
		have = c.compKV.Dim(1)
	}
	if want > have {
		c.compKV = catTime(c.compKV, c.compressBlocks(have, want, cfg))
	}
	return c.compKV
}

func catTime(prev, cur *mlx.Array) *mlx.Array {
	if prev == nil {
		return cur
	}
	return mlx.Concatenate([]*mlx.Array{prev, cur}, 1)
}

func posMod(pos *mlx.Array, ratio int32) *mlx.Array {
	q := mlx.FloorDivideScalar(pos, ratio)
	return mlx.Sub(pos, mlx.Mul(q, mlx.FromValue(int(ratio))))
}

func (c *Compressor) compressBlocks(from, to int, cfg *Config) *mlx.Array {
	B := int32(c.tokKV.Dim(0))
	r := c.Ratio
	blocks := make([]*mlx.Array, 0, to-from)
	for i := from; i < to; i++ {
		blocks = append(blocks, c.compressOne(B, int32(i), r, cfg))
	}
	return mlx.Concatenate(blocks, 1)
}

func (c *Compressor) compressOne(B, block, r int32, cfg *Config) *mlx.Array {
	if c.Overlap {
		return c.compressOverlap(B, block, r, cfg)
	}
	lo := block * r
	hi := lo + r
	kv := mlx.SliceStartStop(c.tokKV, []int32{0, lo, 0}, []int32{B, hi, int32(c.tokKV.Dim(2))})
	sc := mlx.SliceStartStop(c.tokScore, []int32{0, lo, 0}, []int32{B, hi, int32(c.tokScore.Dim(2))})
	w := mlx.SoftmaxAxis(sc, 1, true)
	comp := mlx.Sum(mlx.Mul(kv, w), 1, true) // [B,1,D]
	if c.Norm != nil {
		comp = c.Norm.Forward(comp, cfg.RMSNormEps)
	}
	return comp
}

func (c *Compressor) compressOverlap(B, block, r int32, cfg *Config) *mlx.Array {
	d := c.HeadDim
	split := func(t *mlx.Array, second bool) *mlx.Array {
		if second {
			return mlx.SliceStartStop(t, []int32{0, 0, d}, []int32{int32(t.Dim(0)), int32(t.Dim(1)), int32(t.Dim(2))})
		}
		return mlx.SliceStartStop(t, []int32{0, 0, 0}, []int32{int32(t.Dim(0)), int32(t.Dim(1)), d})
	}
	curLo := block * r
	curHi := curLo + r
	kvCur := split(mlx.SliceStartStop(c.tokKV, []int32{0, curLo, 0}, []int32{B, curHi, int32(c.tokKV.Dim(2))}), true)
	scCur := split(mlx.SliceStartStop(c.tokScore, []int32{0, curLo, 0}, []int32{B, curHi, int32(c.tokScore.Dim(2))}), true)
	var kv, sc *mlx.Array
	if block == 0 {
		z := mlx.Zeros(c.tokKV.DType(), int(B), int(r), int(d))
		ns := mlx.Zeros(c.tokScore.DType(), int(B), int(r), int(d))
		// llama pads missing prev scores with -inf so softmax ignores them
		ns = mlx.Add(ns, mlx.NewScalarArray(float32(-1e9)))
		kv = mlx.Concatenate([]*mlx.Array{z, kvCur}, 1)
		sc = mlx.Concatenate([]*mlx.Array{ns, scCur}, 1)
	} else {
		prevLo := (block - 1) * r
		prevHi := curLo
		kvPrev := split(mlx.SliceStartStop(c.tokKV, []int32{0, prevLo, 0}, []int32{B, prevHi, int32(c.tokKV.Dim(2))}), false)
		scPrev := split(mlx.SliceStartStop(c.tokScore, []int32{0, prevLo, 0}, []int32{B, prevHi, int32(c.tokScore.Dim(2))}), false)
		kv = mlx.Concatenate([]*mlx.Array{kvPrev, kvCur}, 1)
		sc = mlx.Concatenate([]*mlx.Array{scPrev, scCur}, 1)
	}
	w := mlx.SoftmaxAxis(sc, 1, true)
	comp := mlx.Sum(mlx.Mul(kv, w), 1, true)
	if c.Norm != nil {
		comp = c.Norm.Forward(comp, cfg.RMSNormEps)
	}
	return comp
}

func indexerTopK(scores *mlx.Array, k int) *mlx.Array {
	// scores [B, Lq, Lk]
	lk := scores.Dim(2)
	if k > lk {
		k = lk
	}
	neg := mlx.Neg(scores)
	inds := mlx.Argpartition(neg, k-1, -1)
	return mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(scores.Dim(0)), int32(scores.Dim(1)), int32(k)})
}
