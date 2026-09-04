package deepseekv4

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

type SwitchMLP struct {
	GateWeightQ, GateScales, GateBiases *mlx.Array
	UpWeightQ, UpScales, UpBiases       *mlx.Array
	DownWeightQ, DownScales, DownBiases *mlx.Array
	GateWeight, UpWeight, DownWeight    *mlx.Array
	GateBits, UpBits, DownBits          int
	GateGroupSize, UpGroupSize, DownGroupSize int
	UseQuantized                        bool
}

func (s *SwitchMLP) Forward(x, indices *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	topK := cfg.NumExpertsPerTok
	xExpanded := mlx.ExpandDims(mlx.ExpandDims(x, -2), -2)
	xFlat := mlx.Reshape(xExpanded, B*L, 1, 1, cfg.HiddenSize)
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
			nil, idxFlat, true, s.GateGroupSize, s.GateBits, cfg.QuantMode, doSort)
		up = mlx.GatherQMM(xFlat, s.UpWeightQ, s.UpScales, s.UpBiases,
			nil, idxFlat, true, s.UpGroupSize, s.UpBits, cfg.QuantMode, doSort)
		hidden = swigluLimit(gate, up, cfg.SwigluLimit)
		down = mlx.GatherQMM(hidden, s.DownWeightQ, s.DownScales, s.DownBiases,
			nil, idxFlat, true, s.DownGroupSize, s.DownBits, cfg.QuantMode, doSort)
	} else {
		gate = mlx.GatherMM(xFlat, mlx.Transpose(s.GateWeight, 0, 2, 1), nil, idxFlat, doSort)
		up = mlx.GatherMM(xFlat, mlx.Transpose(s.UpWeight, 0, 2, 1), nil, idxFlat, doSort)
		hidden = swigluLimit(gate, up, cfg.SwigluLimit)
		down = mlx.GatherMM(hidden, mlx.Transpose(s.DownWeight, 0, 2, 1), nil, idxFlat, doSort)
	}
	if doSort {
		down = mlx.Reshape(mlx.Take(mlx.Squeeze(mlx.Squeeze(down, 2), 1), invOrder, 0), B*L, topK, cfg.HiddenSize)
	} else {
		down = mlx.Squeeze(down, 2)
	}
	return mlx.Reshape(down, B, L, topK, cfg.HiddenSize)
}

func swigluLimit(gate, up *mlx.Array, limit float32) *mlx.Array {
	if limit > 0 {
		gate = mlx.Clamp(gate, -limit, limit)
		up = mlx.Clamp(up, -limit, limit)
	}
	return mlx.SwiGLU(gate, up)
}

type SharedExperts struct {
	GateProj nn.LinearLayer
	UpProj   nn.LinearLayer
	DownProj nn.LinearLayer
}

func (s *SharedExperts) Forward(x *mlx.Array, limit float32) *mlx.Array {
	return s.DownProj.Forward(swigluLimit(s.GateProj.Forward(x), s.UpProj.Forward(x), limit))
}

type MoEGate struct {
	Gate nn.LinearLayer
	Bias *mlx.Array
}

func sqrtSoftplus(x *mlx.Array) *mlx.Array {
	return mlx.Softplus(x).Sqrt()
}

func (g *MoEGate) Forward(x *mlx.Array, cfg *Config) (*mlx.Array, *mlx.Array) {
	// Why split bias: llama.cpp uses exp_probs_b only for top-k selection;
	// mix weights are unbiased sqrtsoftplus (noaux_tc).
	logits := g.Gate.Forward(x)
	unbiased := sqrtSoftplus(logits)
	selection := unbiased
	if g.Bias != nil {
		selection = mlx.Add(unbiased, g.Bias)
	}
	neg := mlx.Neg(selection)
	topK := cfg.NumExpertsPerTok
	inds := mlx.Argpartition(neg, int(topK)-1, -1)
	d := inds.Dims()
	inds = mlx.SliceStartStop(inds, []int32{0, 0, 0}, []int32{int32(d[0]), int32(d[1]), topK})
	picked := mlx.TakeAlongAxis(unbiased, inds, -1)
	if topK > 1 && cfg.NormTopKProb {
		picked = mlx.Div(picked, mlx.Sum(picked, -1, true))
	}
	if cfg.RoutedScalingFactor != 0 && cfg.RoutedScalingFactor != 1 {
		picked = mlx.MulScalar(picked, cfg.RoutedScalingFactor)
	}
	return inds, picked
}

func hashExperts(tid2eid, tokenIDs *mlx.Array, topK int32) *mlx.Array {
	// Why topk-on-dim0 not "smaller dim": llama.cpp is {n_expert_used, n_vocab}.
	// llama.cpp stores {n_expert_used, n_vocab}; mlx-lm usually [vocab, topk].
	dims := tid2eid.Dims()
	ids := mlx.Reshape(tokenIDs, int32(tokenIDs.Dims()[0]), int32(tokenIDs.Dims()[1]))
	if len(dims) == 2 && int32(dims[0]) == topK && int32(dims[1]) != topK {
		tid2eid = mlx.Transpose(tid2eid, 1, 0)
	}
	flatIDs := mlx.Reshape(ids, int32(ids.Dims()[0])*int32(ids.Dims()[1]))
	return mlx.Take(tid2eid, flatIDs, 0)
}

type MoE struct {
	Gate          *MoEGate
	Tid2eid       *mlx.Array
	SwitchMLP     *SwitchMLP
	SharedExperts *SharedExperts
	Hash          bool
}

func (m *MoE) Forward(x, tokenIDs *mlx.Array, cfg *Config) *mlx.Array {
	dims := x.Dims()
	B, L := int32(dims[0]), int32(dims[1])
	var inds, scores *mlx.Array
	if m.Hash && m.Tid2eid != nil && tokenIDs != nil {
		// Why not uniform 1/k: tid2eid selects experts; gate scores still weight them.
		flat := hashExperts(m.Tid2eid, tokenIDs, cfg.NumExpertsPerTok)
		inds = mlx.Reshape(flat, B, L, cfg.NumExpertsPerTok)
		unbiased := sqrtSoftplus(m.Gate.Gate.Forward(x))
		scores = mlx.TakeAlongAxis(unbiased, inds, -1)
		if cfg.NumExpertsPerTok > 1 && cfg.NormTopKProb {
			scores = mlx.Div(scores, mlx.Sum(scores, -1, true))
		}
		if cfg.RoutedScalingFactor != 0 && cfg.RoutedScalingFactor != 1 {
			scores = mlx.MulScalar(scores, cfg.RoutedScalingFactor)
		}
	} else {
		inds, scores = m.Gate.Forward(x, cfg)
	}
	expertOut := m.SwitchMLP.Forward(x, inds, cfg)
	y := mlx.Sum(mlx.Mul(expertOut, mlx.ExpandDims(scores, -1)), 2, false)
	if m.SharedExperts != nil {
		// Why 0: mlx-lm DeepseekV4 shared experts do not apply swiglu_limit
		// (routed experts still clamp). Clamping the shared residual mixed
		// the trunk into generic Chinese assistant completions.
		y = mlx.Add(y, m.SharedExperts.Forward(x, 0))
	}
	return mlx.Reshape(y, B, L, cfg.HiddenSize)
}
