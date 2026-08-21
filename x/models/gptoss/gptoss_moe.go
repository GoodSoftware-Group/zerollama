package gptoss

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/models/nn"
)

type stackedExpertWeights struct {
	Weight    *mlx.Array
	Scales    *mlx.Array
	Biases    *mlx.Array
	FPBias    *mlx.Array
	Bits      int
	GroupSize int
	Mode      string
}

// SwitchMLP executes the selected expert MLPs with stacked expert weights.
type SwitchMLP struct {
	GateWeight *mlx.Array
	UpWeight   *mlx.Array
	DownWeight *mlx.Array

	GateWeightQ, GateScales, GateBiases *mlx.Array
	UpWeightQ, UpScales, UpBiases       *mlx.Array
	DownWeightQ, DownScales, DownBiases *mlx.Array

	GateFPBias *mlx.Array
	UpFPBias   *mlx.Array
	DownFPBias *mlx.Array

	GateBits, UpBits, DownBits                int
	GateGroupSize, UpGroupSize, DownGroupSize int
	GateMode, UpMode, DownMode                string

	UseQuantized bool
}

// SparseMoE routes each token to the top-k experts.
type SparseMoE struct {
	Router    nn.LinearLayer
	SwitchMLP *SwitchMLP
}

func supportsGatherQMM(mode string, bits int) bool {
	switch mode {
	case "affine":
		return bits == 3 || bits == 4 || bits == 6 || bits == 8
	case "mxfp8":
		return bits == 8
	case "nvfp4", "mxfp4":
		return bits == 4
	default:
		return false
	}
}

func transposeExpertWeightForGatherMM(w *mlx.Array) *mlx.Array {
	if w == nil || !w.Valid() || w.NumDims() != 3 {
		return w
	}
	t := mlx.Transpose(w, 0, 2, 1)
	cloned := t.Clone()
	mlx.Eval(cloned)
	return cloned
}

func loadStackedProjection(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, base string) *stackedExpertWeights {
	key := base + ".weight"
	w := tensors[key]
	if w == nil {
		return nil
	}

	out := &stackedExpertWeights{FPBias: tensors[base+".bias"]}

	scales := tensors[key+"_scale"]
	if scales == nil {
		out.Weight = w
		return out
	}

	qbiases := tensors[key+"_qbias"]
	groupSize, bits, mode := model.ResolveLinearQuantParams(
		cfg.QuantGroupSize, cfg.QuantBits, cfg.QuantMode, cfg.TensorQuant,
		key, w, scales,
	)
	if useQuantized && supportsGatherQMM(mode, bits) {
		out.Weight = w
		out.Scales = scales
		out.Biases = qbiases
		out.Bits = bits
		out.GroupSize = groupSize
		out.Mode = mode
		return out
	}

	out.Weight = mlx.Dequantize(w, scales, qbiases, groupSize, bits, mode, nil)
	out.Bits = bits
	out.GroupSize = groupSize
	out.Mode = mode
	return out
}

func loadLayerExperts(tensors map[string]*mlx.Array, cfg *Config, useQuantized bool, layerPrefix string) (*SwitchMLP, error) {
	gateW := loadStackedProjection(tensors, cfg, useQuantized, layerPrefix+".mlp.experts.gate_proj")
	upW := loadStackedProjection(tensors, cfg, useQuantized, layerPrefix+".mlp.experts.up_proj")
	downW := loadStackedProjection(tensors, cfg, useQuantized, layerPrefix+".mlp.experts.down_proj")
	if gateW == nil || upW == nil || downW == nil {
		return nil, fmt.Errorf("missing stacked expert weights (gate=%v up=%v down=%v)", gateW != nil, upW != nil, downW != nil)
	}

	switchMLP := &SwitchMLP{
		GateFPBias: gateW.FPBias,
		UpFPBias:   upW.FPBias,
		DownFPBias: downW.FPBias,
	}

	if gateW.Scales != nil && upW.Scales != nil && downW.Scales != nil {
		switchMLP.UseQuantized = true
		switchMLP.GateWeightQ = gateW.Weight
		switchMLP.GateScales = gateW.Scales
		switchMLP.GateBiases = gateW.Biases
		switchMLP.GateBits = gateW.Bits
		switchMLP.GateGroupSize = gateW.GroupSize
		switchMLP.GateMode = gateW.Mode
		switchMLP.UpWeightQ = upW.Weight
		switchMLP.UpScales = upW.Scales
		switchMLP.UpBiases = upW.Biases
		switchMLP.UpBits = upW.Bits
		switchMLP.UpGroupSize = upW.GroupSize
		switchMLP.UpMode = upW.Mode
		switchMLP.DownWeightQ = downW.Weight
		switchMLP.DownScales = downW.Scales
		switchMLP.DownBiases = downW.Biases
		switchMLP.DownBits = downW.Bits
		switchMLP.DownGroupSize = downW.GroupSize
		switchMLP.DownMode = downW.Mode
		return switchMLP, nil
	}

	switchMLP.GateWeight = transposeExpertWeightForGatherMM(gateW.Weight)
	switchMLP.UpWeight = transposeExpertWeightForGatherMM(upW.Weight)
	switchMLP.DownWeight = transposeExpertWeightForGatherMM(downW.Weight)
	return switchMLP, nil
}

func addGatheredExpertBias(out, bias, idxFlat *mlx.Array) *mlx.Array {
	if bias == nil || !bias.Valid() {
		return out
	}
	selected := mlx.Take(bias, mlx.Flatten(idxFlat), 0)
	dims := out.Dims()
	if len(dims) == 3 {
		return mlx.Add(out, mlx.Reshape(selected, int32(dims[0]), 1, int32(dims[2])))
	}
	shape := make([]int32, len(dims))
	for i, d := range dims {
		shape[i] = int32(d)
	}
	return mlx.Add(out, mlx.Reshape(selected, shape...))
}

func (s *SwitchMLP) Forward(x *mlx.Array, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok

	xFlat := mlx.Reshape(x, B*L, 1, 1, cfg.HiddenSize)
	idxFlat := mlx.Reshape(indices, B*L, topK)

	doSort := B*L >= 64
	var invOrder *mlx.Array
	n := B * L * topK

	if doSort {
		idxAll := mlx.Flatten(idxFlat)
		order := mlx.Argsort(idxAll, 0)
		invOrder = mlx.Argsort(order, 0)
		xFlat = mlx.ExpandDims(mlx.Take(mlx.Squeeze(xFlat, 1), mlx.FloorDivideScalar(order, topK), 0), 1)
		idxFlat = mlx.Reshape(mlx.Take(idxAll, order, 0), n, 1)
	}

	var gate, up, hidden, down *mlx.Array
	if s.UseQuantized {
		gate = mlx.GatherQMM(xFlat, s.GateWeightQ, s.GateScales, s.GateBiases,
			nil, idxFlat, true, s.GateGroupSize, s.GateBits, s.GateMode, doSort)
		up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
			nil, idxFlat, true, s.UpGroupSize, s.UpBits, s.UpMode, doSort)
		gate = addGatheredExpertBias(gate, s.GateFPBias, idxFlat)
		up = addGatheredExpertBias(up, s.UpFPBias, idxFlat)
		hidden = mlx.SwiGLUOAI(gate, up)
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, s.DownMode, doSort)
		down = addGatheredExpertBias(down, s.DownFPBias, idxFlat)
	} else {
		gate = mlx.GatherMM(xFlat, s.GateWeight, nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, s.UpWeight, nil, idxFlat, doSort)
		gate = addGatheredExpertBias(gate, s.GateFPBias, idxFlat)
		up = addGatheredExpertBias(up, s.UpFPBias, idxFlat)
		hidden = mlx.SwiGLUOAI(gate, up)
		down = mlx.GatherMM(hidden, s.DownWeight, nil, idxFlat, doSort)
		down = addGatheredExpertBias(down, s.DownFPBias, idxFlat)
	}

	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}

	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func (moe *SparseMoE) route(x *mlx.Array, cfg *Config) (inds, scores *mlx.Array) {
	dims := x.Dims()
	if len(dims) != 3 {
		panic(fmt.Sprintf("gptoss moe: expected rank-3 hidden states, got %v", dims))
	}
	B, L, H := int32(dims[0]), int32(dims[1]), int32(dims[2])
	BL := B * L
	topK := cfg.NumExpertsPerTok
	if topK <= 0 {
		panic(fmt.Sprintf("gptoss moe: invalid num_experts_per_tok=%d", topK))
	}
	if cfg.NumLocalExperts < topK {
		panic(fmt.Sprintf("gptoss moe: num_local_experts=%d < topK=%d", cfg.NumLocalExperts, topK))
	}

	// Flatten like Gemma4 MoE. Keep routing on [B*L, E] so a bad router
	// quant (empty logits) fails with a clear shape error instead of
	// panicking on shape[0] after argpartition.
	x2d := mlx.Reshape(x, BL, H)
	logits := moe.Router.Forward(x2d)
	logitDims := logits.Dims()
	if len(logitDims) != 2 || int32(logitDims[0]) != BL {
		panic(fmt.Sprintf("gptoss moe: router logits shape %v, want [%d, %d]", logitDims, BL, cfg.NumLocalExperts))
	}

	kth := int(topK) - 1
	inds = mlx.Argpartition(mlx.Neg(logits), kth, -1)
	shape := inds.Dims()
	if len(shape) < 2 {
		panic(fmt.Sprintf("gptoss moe: argpartition returned invalid shape %v (logits=%v kth=%d experts=%d)",
			shape, logitDims, kth, cfg.NumLocalExperts))
	}
	inds = mlx.SliceStartStop(inds, []int32{0, 0}, []int32{BL, topK})

	selected := mlx.TakeAlongAxis(logits, inds, -1)
	scores = mlx.SoftmaxAxis(selected, -1, true)

	inds = mlx.Reshape(inds, B, L, topK)
	scores = mlx.Reshape(scores, B, L, topK)
	return inds, scores
}

func (moe *SparseMoE) Forward(x *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])

	inds, scores := moe.route(x, cfg)
	expertOut := moe.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)
	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}
