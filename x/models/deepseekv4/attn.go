package deepseekv4

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// Attention is Flash MLA: LoRA Q, single-head KV, grouped wo_a/wo_b, sinks.
//
// Why not one MHA for every layer: compress_ratio 0 is raw SWA only; 4 is CSA
// (overlap compressor + indexer top-k); 128 is HCA (softmax compressor).
// Concat(raw, compressed) is the Flash attention, not full-history MHA.
//
// Why reset compressors when cache.Offset()==0: layer-local tok/comp buffers
// are not in cache.Snapshot; a new prefill must not reuse a previous sequence.
type Attention struct {
	WQA    nn.LinearLayer
	QNorm  *nn.RMSNorm
	WQB    nn.LinearLayer
	WKV    nn.LinearLayer
	KVNorm *nn.RMSNorm
	WoA    nn.LinearLayer
	WoB    nn.LinearLayer
	Sinks  *mlx.Array
	Ratio  int32
	Comp   *Compressor
	IdxComp *Compressor
	IdxWQB nn.LinearLayer
	IdxProj nn.LinearLayer
}

func (a *Attention) Forward(x *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config, pack ropePack) *mlx.Array {
	if c != nil && c.Offset() == 0 {
		if a.Comp != nil {
			a.Comp.reset()
		}
		if a.IdxComp != nil {
			a.IdxComp.reset()
		}
	}

	qr := a.WQA.Forward(x)
	if a.QNorm != nil {
		qr = a.QNorm.Forward(qr, cfg.RMSNormEps)
	}
	q := a.WQB.Forward(qr)
	q = mlx.Reshape(q, B, L, cfg.NumAttentionHeads, cfg.HeadDim)
	q = mlx.RMSNormFn(q, nil, cfg.RMSNormEps)
	q = mlx.Transpose(q, 0, 2, 1, 3)

	kv := a.WKV.Forward(x)
	if a.KVNorm != nil {
		kv = a.KVNorm.Forward(kv, cfg.RMSNormEps)
	}
	kv = mlx.Reshape(kv, B, L, cfg.NumKeyValueHeads, cfg.HeadDim)
	kv = mlx.Transpose(kv, 0, 2, 1, 3)

	compress := a.Ratio != 0
	q = applyRoPE(q, positions, cfg, pack, false)
	kv = applyRoPE(kv, positions, cfg, pack, false)

	k, v := kv, kv
	out := a.attend(x, qr, q, k, v, b, c, positions, B, L, cfg, pack, compress)

	out = applyRoPE(out, positions, cfg, pack, true)
	out = mlx.Transpose(out, 0, 2, 1, 3) // [B,L,H,D]
	nHeadsGroup := cfg.NumAttentionHeads / cfg.OGroups
	oGroupDim := nHeadsGroup * cfg.HeadDim
	out = mlx.Reshape(out, B, L, cfg.OGroups, oGroupDim)
	oa := groupedWoA(a.WoA, out, cfg.OGroups, cfg.OLoraRank)
	oa = mlx.Reshape(oa, B, L, cfg.OGroups*cfg.OLoraRank)
	return a.WoB.Forward(oa)
}

func (a *Attention) attend(x, qr, q, k, v *mlx.Array, b *batch.Batch, c cache.Cache, positions *mlx.Array, B, L int32, cfg *Config, pack ropePack, compress bool) *mlx.Array {
	window := int(cfg.SlidingWindow)
	var kvOpt nn.SDPAOption
	var histK, histV *mlx.Array
	if c != nil {
		history := c.(cache.Attention).Update(b, k, v)
		histK, histV = history.K(), history.V()
		kvOpt = nn.WithKVHistory(history)
	} else {
		histK, histV = k, v
		kvOpt = nn.WithKV(k, v, b.SeqQueryLens)
	}

	if !compress || a.Comp == nil {
		mask := nn.CausalMask()
		if window > 0 {
			klen := histK.Dim(2)
			if sw := nn.SlidingWindowMask(b, klen, window, q.DType()); !sw.IsZero() {
				mask = mask.Intersect(sw)
			}
		}
		opts := []nn.SDPAOption{kvOpt, nn.WithMask(mask)}
		if a.Sinks != nil {
			opts = append(opts, nn.WithSinks(a.Sinks))
		}
		return nn.ScaledDotProductAttention(b, q, cfg.Scale, opts...)
	}

	tokPos := tokenPositions(positions, L)
	comp := a.Comp.append(x, tokPos, cfg)
	if comp != nil {
		comp = ropeCompressed(comp, cfg, pack, a.Ratio)
	}

	var idxComp *mlx.Array
	if a.IdxComp != nil {
		idxComp = a.IdxComp.append(x, tokPos, cfg)
		if idxComp != nil {
			idxComp = ropeCompressed(idxComp, cfg, pack, a.Ratio)
		}
	}

	// Why roll: CSA concat bypasses RotatingKVCache's mask applier. Decode
	// after SWA wrap stores raw K in ring order; compressMask and SDPA need
	// oldest-first or Q attends the wrong 128 tokens (chat templates are
	// already longer than sliding_window).
	histK, histV = kvOldestFirst(c, histK, histV)

	kAll, vAll := histK, histV
	if comp != nil {
		ck := mlx.Reshape(comp, B, 1, int32(comp.Dim(1)), cfg.HeadDim)
		kAll = mlx.Concatenate([]*mlx.Array{histK, ck}, 2)
		vAll = mlx.Concatenate([]*mlx.Array{histV, ck}, 2)
	}

	mask := compressMask(b, q, histK, comp, idxComp, qr, x, a, cfg, pack, positions, window)
	kLens := make([]int32, int(B))
	for i := range kLens {
		kLens[i] = int32(kAll.Dim(2))
	}
	opts := []nn.SDPAOption{nn.WithKV(kAll, vAll, kLens), nn.WithMask(mask)}
	if a.Sinks != nil {
		opts = append(opts, nn.WithSinks(a.Sinks))
	}
	return nn.ScaledDotProductAttention(b, q, cfg.Scale, opts...)
}

func ropeCompressed(comp *mlx.Array, cfg *Config, pack ropePack, ratio int32) *mlx.Array {
	if comp == nil {
		return nil
	}
	B := int32(comp.Dim(0))
	n := int32(comp.Dim(1))
	if n <= 0 {
		return comp
	}
	x := mlx.Reshape(comp, B, 1, n, int32(comp.Dim(2)))
	starts := compressedWritePos(n, ratio)
	pos := mlx.FromValues(starts, int(n))
	pos = mlx.ExpandDims(pos, 0)
	if B > 1 {
		pos = mlx.BroadcastTo(pos, B, n)
	}
	return applyRoPE(x, pos, cfg, pack, false)
}

func compressMask(b *batch.Batch, q, rawK, comp, idxComp, qr, x *mlx.Array, a *Attention, cfg *Config, pack ropePack, positions *mlx.Array, window int) nn.AttentionMask {
	B := q.Dim(0)
	L := q.Dim(2)
	Kr := rawK.Dim(2)
	Kc := 0
	if comp != nil {
		Kc = comp.Dim(1)
	}
	K := Kr + Kc
	neg := float32(math.Inf(-1))
	vals := make([]float32, B*L*K)
	// raw: causal + SWA
	for i := 0; i < B; i++ {
		off := int(b.SeqOffsets[i])
		oldest := Kr
		if window > 0 {
			oldest = max(0, off+L-window)
		}
		rawOldestPos := max(0, off+L-Kr)
		for qpos := 0; qpos < L; qpos++ {
			absQ := off + qpos
			row := (i*L + qpos) * K
			for k := 0; k < Kr; k++ {
				absK := rawOldestPos + k
				if absK > absQ || absK < oldest {
					vals[row+k] = neg
				}
			}
			for k := 0; k < Kc; k++ {
				// compressed block k ends at token (k+1)*ratio - 1
				absK := (k+1)*int(a.Ratio) - 1
				if absK > absQ {
					vals[row+Kr+k] = neg
				}
			}
		}
	}
	arr := mlx.FromValues(vals, B, 1, L, K).AsType(q.DType())
	if a.Ratio == 4 && a.IdxWQB != nil && idxComp != nil && Kc > 0 {
		arr = applyIndexerMask(arr, qr, x, idxComp, a, cfg, pack, positions, B, L, Kr, Kc)
	}
	return nn.ArrayMask(arr)
}

type ringWriter interface {
	RingWriteIndex() int
}

func kvOldestFirst(c cache.Cache, k, v *mlx.Array) (*mlx.Array, *mlx.Array) {
	if c == nil || k == nil {
		return k, v
	}
	r, ok := c.(ringWriter)
	if !ok {
		return k, v
	}
	idx := r.RingWriteIndex()
	return rollTimeToOldest(k, idx), rollTimeToOldest(v, idx)
}

func rollTimeToOldest(t *mlx.Array, ringIdx int) *mlx.Array {
	if t == nil {
		return nil
	}
	klen := t.Dim(2)
	if klen <= 0 {
		return t
	}
	idx := ringIdx % klen
	if idx == 0 {
		return t
	}
	tail := mlx.SliceStartStop(t, []int32{0, 0, int32(idx), 0}, []int32{int32(t.Dim(0)), int32(t.Dim(1)), int32(klen), int32(t.Dim(3))})
	head := mlx.SliceStartStop(t, []int32{0, 0, 0, 0}, []int32{int32(t.Dim(0)), int32(t.Dim(1)), int32(idx), int32(t.Dim(3))})
	return mlx.Concatenate([]*mlx.Array{tail, head}, 2)
}

func applyIndexerMask(base, qr, x, idxComp *mlx.Array, a *Attention, cfg *Config, pack ropePack, positions *mlx.Array, B, L, Kr, Kc int) *mlx.Array {
	iq := a.IdxWQB.Forward(qr)
	iq = mlx.Reshape(iq, int32(B), int32(L), cfg.IndexNHeads, cfg.IndexHeadDim)
	iq = mlx.Transpose(iq, 0, 2, 1, 3) // [B,H,L,D]
	iq = applyRoPE(iq, positions, cfg, pack, false)
	ik := mlx.Reshape(idxComp, int32(B), 1, int32(Kc), cfg.IndexHeadDim)
	// scores: relu(q @ k^T) * weights, sum heads
	kt := mlx.Transpose(ik, 0, 1, 3, 2) // [B,1,D,Kc]
	logits := mlx.Matmul(iq, kt)        // [B,H,L,Kc]
	logits = mlx.Maximum(logits, mlx.NewScalarArray(0))
	w := a.IdxProj.Forward(x)
	scale := float32(1.0 / math.Sqrt(float64(cfg.IndexHeadDim*cfg.IndexNHeads)))
	w = mlx.MulScalar(w, scale)
	w = mlx.Reshape(w, int32(B), int32(L), cfg.IndexNHeads)
	w = mlx.Transpose(w, 0, 2, 1) // [B,H,L]
	w = mlx.ExpandDims(w, -1)
	score := mlx.Sum(mlx.Mul(logits, w), 1, false) // [B,L,Kc]
	topk := indexerTopK(score, int(cfg.IndexTopK))
	allowed := mlx.Zeros(mlx.DTypeFloat32, B, 1, L, Kc)
	idx := mlx.Reshape(topk, int32(B), 1, int32(L), int32(topk.Dim(2)))
	ones := mlx.BroadcastTo(mlx.NewScalarArray(1), int32(B), 1, int32(L), int32(topk.Dim(2)))
	allowed = allowed.PutAlongAxis(idx, ones, -1)
	neg := mlx.NewScalarArray(float32(math.Inf(-1)))
	compMask := mlx.Where(allowed.Greater(mlx.NewScalarArray(0)), mlx.NewScalarArray(0), neg)
	rawPart := mlx.SliceStartStop(base, []int32{0, 0, 0, 0}, []int32{int32(B), 1, int32(L), int32(Kr)})
	return mlx.Concatenate([]*mlx.Array{rawPart, mlx.Add(compMask, mlx.SliceStartStop(base, []int32{0, 0, 0, int32(Kr)}, []int32{int32(B), 1, int32(L), int32(Kr+Kc)}))}, -1)
}
