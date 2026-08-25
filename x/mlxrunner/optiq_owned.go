package mlxrunner

import (
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	qwen35 "github.com/ollama/ollama/x/models/qwen3_5"
	"github.com/ollama/ollama/x/uma"
)

// installOptiqOwnedHook registers L25 InProjZ → broker GEMV when GRAPH_DECODE=owned (F0685).
func (r *Runner) installOptiqOwnedHook() {
	nn.SetQuantizedForwardHook(nil)
	if !uma.OptiqDecodeOwned() {
		return
	}
	m, ok := r.Model.(*qwen35.Model)
	if !ok || m == nil || len(m.Layers) <= 25 {
		return
	}
	lin := m.Layers[25].Linear
	if lin == nil || lin.InProjZ == nil {
		return
	}
	ql, ok := lin.InProjZ.(*nn.QuantizedLinear)
	if !ok {
		slog.Warn("uma: optiq owned: L25 InProjZ is not QuantizedLinear", "type", lin.InProjZ)
		return
	}
	uma.RegisterOwnedLinearTarget(ql)
	_ = uma.EnsureOptiqDecodeSession()
	nn.SetQuantizedForwardHook(func(q *nn.QuantizedLinear, x *mlx.Array) (*mlx.Array, bool) {
		if !uma.OwnedTargetMatch(q) {
			return nil, false
		}
		xf := x.AsType(mlx.DTypeFloat32)
		mlx.Eval(xf)
		xs := xf.Floats()
		ys, err := uma.OwnedInProjZGemv(xs)
		dims := x.Dims()
		if err != nil {
			uma.SetOwnedForwardErr(err)
			slog.Error("uma: optiq owned InProjZ GEMV failed", "error", err)
			// Fail closed: do not fall through to MLX QMM.
			return mlx.Zeros(mlx.DTypeFloat32, dims...), true
		}
		switch len(dims) {
		case 1:
			return mlx.FromValues(ys, len(ys)), true
		case 2:
			return mlx.FromValues(ys, dims[0], len(ys)), true
		case 3:
			B, L, N := dims[0], dims[1], len(ys)
			out := make([]float32, B*L*N)
			if L > 1 && len(xs) >= B*L*dims[2] {
				// keep prior positions from MLX input path unused; z is per-position
				// For L>1 prefill, run last-row replace only (decode is L=1).
			}
			copy(out[(L-1)*N:], ys)
			return mlx.FromValues(out, B, L, N), true
		default:
			return mlx.FromValues(ys, len(ys)), true
		}
	})
	slog.Info("uma: optiq owned Forward hook on L25 InProjZ")
}
