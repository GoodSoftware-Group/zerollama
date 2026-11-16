package siglip

import (
	"math"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

type visionTower struct {
	PatchWeight *mlx.Array
	PatchBias   *mlx.Array
	PatchLinear nn.LinearLayer
	Positions   *nn.Embedding
	Blocks      []*encoderBlock
	PostNorm    *nn.LayerNorm
	Head        *mapHead
	Projection  nn.LinearLayer
	cfg         *VisionConfig
	act         func(*mlx.Array) *mlx.Array
}

type mapHead struct {
	Probe    *mlx.Array
	InWeight *mlx.Array
	InBias   *mlx.Array
	Q        nn.LinearLayer
	K        nn.LinearLayer
	V        nn.LinearLayer
	Out      nn.LinearLayer
	Norm     *nn.LayerNorm
	FC1      nn.LinearLayer
	FC2      nn.LinearLayer
	heads    int32
	act      func(*mlx.Array) *mlx.Array
}

type textTower struct {
	Tokens    nn.EmbeddingLayer
	Positions *nn.Embedding
	Blocks    []*encoderBlock
	FinalNorm *nn.LayerNorm
	Head      nn.LinearLayer
	cfg       *TextConfig
	act       func(*mlx.Array) *mlx.Array
}

func (t *visionTower) encode(plan imagePlan) *mlx.Array {
	h := t.tokens(plan)
	if t.Head != nil {
		h = t.Head.forward(h)
	} else {
		h = mlx.Mean(h, 1, false)
	}
	if t.Projection != nil {
		h = t.Projection.Forward(h)
	}
	return asRow(h)
}

// tokens is the encoder output after post-layernorm: [1, patches, hidden].
// The MAP head and visual projection are not applied.
func (t *visionTower) tokens(plan imagePlan) *mlx.Array {
	var h *mlx.Array
	if t.PatchLinear != nil {
		patches := mlx.FromValues(plan.Patches, 1, plan.GridH*plan.GridW, len(plan.Patches)/(plan.GridH*plan.GridW))
		h = t.PatchLinear.Forward(patches.AsType(t.posDtype()))
	} else {
		pixels := mlx.FromValues(plan.Pixels, 1, plan.Height, plan.Width, int(t.cfg.NumChannels))
		pixels = pixels.AsType(t.PatchWeight.DType())
		h = mlx.Conv2d(pixels, t.PatchWeight, t.cfg.PatchSize, t.cfg.PatchSize, 0, 0, 1, 1, 1)
		if t.PatchBias != nil {
			h = mlx.Add(h, mlx.Reshape(t.PatchBias, 1, 1, 1, t.cfg.HiddenSize))
		}
		h = mlx.Reshape(h, 1, int32(plan.GridH*plan.GridW), t.cfg.HiddenSize)
	}
	return t.run(h, plan.GridH, plan.GridW)
}

func (t *visionTower) run(h *mlx.Array, gridH, gridW int) *mlx.Array {
	h = mlx.Add(h, t.positions(gridH, gridW).AsType(h.DType()))
	for _, block := range t.Blocks {
		h = block.forward(h, t.cfg.NumHeads, t.act)
	}
	return t.PostNorm.Forward(h)
}

func (t *visionTower) posDtype() mlx.DType {
	if t.Positions != nil && t.Positions.Weight != nil {
		return t.Positions.Weight.DType()
	}
	if t.PatchWeight != nil {
		return t.PatchWeight.DType()
	}
	return mlx.DTypeFloat32
}

func (t *visionTower) positions(gridH, gridW int) *mlx.Array {
	n := int32(gridH * gridW)
	side := int(math.Round(math.Sqrt(float64(t.cfg.NumPatches))))
	if gridH == side && gridW == side {
		ids := mlx.Arange(0, float64(n), 1, mlx.DTypeInt32)
		return mlx.ExpandDims(t.Positions.Forward(ids), 0)
	}
	idx, wt := bilinearPosWeights(side, gridH, gridW)
	pos := mlx.Zeros(mlx.DTypeFloat32, gridH*gridW, int(t.cfg.HiddenSize))
	table := t.Positions.Weight.AsType(mlx.DTypeFloat32)
	for c := range 4 {
		at := make([]int32, n)
		w := make([]float32, n)
		for i := range int(n) {
			at[i] = idx[4*i+c]
			w[i] = wt[4*i+c]
		}
		rows := mlx.Take(table, mlx.FromValues(at, int(n)), 0)
		pos = mlx.Add(pos, mlx.Mul(rows, mlx.FromValues(w, int(n), 1)))
	}
	return mlx.ExpandDims(pos, 0)
}

func (h *mapHead) forward(x *mlx.Array) *mlx.Array {
	B := int32(x.Dim(0))
	probe := mlx.Tile(h.Probe, []int32{B, 1, 1})
	var q, k, v *mlx.Array
	if h.InWeight != nil {
		q = projectPacked(h.InWeight, h.InBias, probe, 0)
		k = projectPacked(h.InWeight, h.InBias, x, 1)
		v = projectPacked(h.InWeight, h.InBias, x, 2)
	} else {
		q, k, v = h.Q.Forward(probe), h.K.Forward(x), h.V.Forward(x)
	}
	hidden := int32(x.Dim(2))
	headDim := hidden / h.heads
	shapeQ := func(y *mlx.Array, length int32) *mlx.Array {
		return mlx.Transpose(mlx.Reshape(y, B, length, h.heads, headDim), 0, 2, 1, 3)
	}
	q = shapeQ(q, 1)
	k = shapeQ(k, int32(x.Dim(1)))
	v = shapeQ(v, int32(x.Dim(1)))
	scale := float32(1 / math.Sqrt(float64(headDim)))
	out := mlx.FastScaledDotProductAttention(q, k, v, scale, "", nil)
	out = mlx.Reshape(mlx.Transpose(out, 0, 2, 1, 3), B, 1, hidden)
	out = h.Out.Forward(out)
	residual := out
	out = mlx.Add(residual, h.FC2.Forward(h.act(h.FC1.Forward(h.Norm.Forward(out)))))
	return mlx.Squeeze(out, 1)
}

func projectPacked(weight, bias, x *mlx.Array, which int) *mlx.Array {
	hidden := x.Dim(x.NumDims() - 1)
	start := which * hidden
	w := weight.Slice(mlx.Slice(start, start+hidden), mlx.Slice())
	var b *mlx.Array
	if bias != nil {
		b = bias.Slice(mlx.Slice(start, start+hidden))
	}
	return nn.NewLinear(w, b).Forward(x)
}

func (t *textTower) forward(ids *mlx.Array, pad int32) *mlx.Array {
	max := int(t.cfg.MaxPositionEmbeddings)
	ids = padTokenIDs(ids, max, pad)
	h := t.Tokens.Forward(ids)
	posIDs := mlx.Arange(0, float64(ids.Dim(1)), 1, mlx.DTypeInt32)
	pos := mlx.ExpandDims(t.Positions.Forward(posIDs), 0)
	h = mlx.Add(h, pos.AsType(h.DType()))
	for _, block := range t.Blocks {
		h = block.forward(h, t.cfg.NumHeads, t.act)
	}
	h = t.FinalNorm.Forward(h)
	// Training pads to max length and pools that final (pad) position.
	last := h.Slice(mlx.Slice(), mlx.Slice(-1, mlx.End), mlx.Slice())
	last = mlx.Squeeze(last, 1)
	if t.Head != nil {
		last = t.Head.Forward(last)
	}
	return asRow(last)
}

func padTokenIDs(ids *mlx.Array, max int, pad int32) *mlx.Array {
	if max <= 0 {
		return ids
	}
	L := ids.Dim(1)
	if L == max {
		return ids
	}
	if L > max {
		return ids.Slice(mlx.Slice(), mlx.Slice(0, max))
	}
	pads := mlx.Tile(mlx.FromValues([]int32{pad}, 1, 1), []int32{1, int32(max - L)})
	return mlx.Concatenate([]*mlx.Array{ids, pads}, 1)
}
