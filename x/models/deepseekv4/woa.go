package deepseekv4

import (
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
)

// groupedWoA applies wo_a as G independent maps [o_group_dim] → [o_lora_rank].
// Why not one 2-D linear on [B,L,G,D]: llama.cpp reshapes the file
// (n_head*head_dim/o_groups, o_lora*o_groups) to [in, o_lora, groups].
func groupedWoA(layer nn.LinearLayer, x *mlx.Array, groups, oLora int32) *mlx.Array {
	dims := x.Dims()
	B, L, G, D := int32(dims[0]), int32(dims[1]), int32(dims[2]), int32(dims[3])
	if G != groups {
		// fall back: treat last two dims as a single matmul
		return layer.Forward(mlx.Reshape(x, B, L, G*D))
	}
	switch w := unwrapLinear(layer).(type) {
	case *nn.QuantizedLinear:
		outs := make([]*mlx.Array, groups)
		for g := int32(0); g < groups; g++ {
			xg := mlx.Squeeze(mlx.SliceStartStop(x,
				[]int32{0, 0, g, 0},
				[]int32{B, L, g + 1, D},
			), 2)
			wg := sliceOutRows(w.Weight, g*oLora, (g+1)*oLora)
			sg := sliceOutRows(w.Scales, g*oLora, (g+1)*oLora)
			var qb *mlx.Array
			if w.QBiases != nil {
				qb = sliceOutRows(w.QBiases, g*oLora, (g+1)*oLora)
			}
			outs[g] = mlx.QuantizedMatmul(xg, wg, sg, qb, true, w.GroupSize, w.Bits, w.Mode)
		}
		return mlx.Concatenate(outs, -1)
	case *nn.Linear:
		outs := make([]*mlx.Array, groups)
		for g := int32(0); g < groups; g++ {
			xg := mlx.Squeeze(mlx.SliceStartStop(x,
				[]int32{0, 0, g, 0},
				[]int32{B, L, g + 1, D},
			), 2)
			wg := sliceOutRows(w.Weight, g*oLora, (g+1)*oLora)
			outs[g] = mlx.Matmul(xg, mlx.Transpose(wg, 1, 0))
		}
		return mlx.Concatenate(outs, -1)
	default:
		flat := mlx.Reshape(x, B, L, G*D)
		return layer.Forward(flat)
	}
}

func unwrapLinear(layer nn.LinearLayer) nn.LinearLayer {
	if d, ok := layer.(*nn.DecodeQuantLinear); ok {
		return d.Dense
	}
	return layer
}

func sliceOutRows(w *mlx.Array, lo, hi int32) *mlx.Array {
	if w == nil {
		return nil
	}
	nd := w.NumDims()
	start := make([]int32, nd)
	stop := make([]int32, nd)
	for i := range stop {
		stop[i] = int32(w.Dim(i))
	}
	start[0], stop[0] = lo, hi
	return mlx.SliceStartStop(w, start, stop)
}
