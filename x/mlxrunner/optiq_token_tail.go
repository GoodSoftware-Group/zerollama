package mlxrunner

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	qwen35 "github.com/ollama/ollama/x/models/qwen3_5"
	"github.com/ollama/ollama/x/uma"
)

// beginOptiqTokenTail arms GRAPH token-tail for this Completion.
// Default live path: MLX final Norm stays on; GRAPH runs GEMV→ARGMAX on
// post-norm last-hidden (freeze rematch). Set ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE=
// norm_gemv_argmax to also SkipFinalNorm and run full F0666 NORM→GEMV→ARGMAX.
func (r *Runner) beginOptiqTokenTail() (cleanup func(), err error) {
	cleanup = func() {}
	if !uma.OptiqTokenTailEnabled() {
		return cleanup, nil
	}
	if err := uma.EnsureOptiqTokenTailSession(); err != nil {
		slog.Error("uma: optiq token-tail session failed", "error", err)
		if uma.OptiqTokenTailRequire() {
			return cleanup, err
		}
		slog.Warn("uma: optiq token-tail soft-fail; MLX Unembed+sample")
		return cleanup, nil
	}
	recipe := os.Getenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE")
	if recipe == "" {
		// Live default: post-norm x → GEMV→ARGMAX (Model.Forward already Norms).
		_ = os.Setenv("ZEROLLAMA_UMA_OPTIQ_TOKEN_RECIPE", "gemv_argmax")
		recipe = "gemv_argmax"
	}
	if recipe == "norm_gemv_argmax" || recipe == "norm" {
		if _, ok := r.Model.(*qwen35.Model); !ok {
			msg := "uma optiq token-tail NORM recipe requires qwen3_5 Model"
			slog.Warn(msg)
			if uma.OptiqTokenTailRequire() {
				return cleanup, fmt.Errorf("%s", msg)
			}
			return cleanup, nil
		}
		qwen35.SetSkipFinalNorm(true)
		cleanup = func() { qwen35.SetSkipFinalNorm(false) }
		slog.Info("uma: optiq GRAPH token-tail armed (SkipFinalNorm + NORM→GEMV→ARGMAX)")
		return cleanup, nil
	}
	slog.Info("uma: optiq GRAPH token-tail armed (MLX Norm + GEMV→ARGMAX)")
	return cleanup, nil
}

// maybeOwnedGraphToken — F0700: replace Unembed+argmax with broker GEMV→ARGMAX
// (or NORM→GEMV→ARGMAX when recipe says so) on last-row hidden.
func maybeOwnedGraphToken(hidden *mlx.Array) (sample.Result, bool, error) {
	if !uma.OptiqTokenTailEnabled() || !uma.OptiqTokenTailSessionReady() {
		return sample.Result{}, false, nil
	}
	hf := hidden.AsType(mlx.DTypeFloat32)
	mlx.Eval(hf)
	mlx.Synchronize()
	xs := hf.Floats()
	dims := hf.Dims()
	if len(dims) == 3 {
		tlen, D := dims[1], dims[2]
		xs = xs[(tlen-1)*D : tlen*D]
	} else if len(dims) == 2 && dims[0] > 1 {
		D := dims[1]
		xs = xs[(dims[0]-1)*D:]
	}
	tok, err := graphTokenTailOutsideLease(xs)
	if err != nil {
		if uma.OptiqTokenTailRequire() {
			return sample.Result{}, true, fmt.Errorf("uma optiq token-tail: %w", err)
		}
		slog.Warn("uma: optiq token-tail soft-fail; MLX Unembed", "error", err)
		return sample.Result{}, false, nil
	}
	slog.Debug("uma: optiq GRAPH token-tail", "tok", tok, "steps", uma.OptiqTokenTailSteps())
	token := mlx.FromValues([]int32{tok}, 1)
	mlx.Eval(token) // host token must be ready before pipelined next Forward
	return sample.Result{Token: token}, true, nil
}

// graphTokenTailOutsideLease drops HOLD_GPU around GRAPH so the broker can
// run GEMV (HOLD∩GRAPH otherwise stalls until lease_ttl).
func graphTokenTailOutsideLease(xs []float32) (int32, error) {
	uma.LeaseEnd()
	tok, err := uma.OptiqTokenTailArgmax(xs)
	if err2 := uma.LeaseBegin("token-tail"); err2 != nil && err == nil {
		err = err2
	}
	return tok, err
}
