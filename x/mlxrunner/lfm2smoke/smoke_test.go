package lfm2smoke_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/lfm2"
	"github.com/ollama/ollama/x/models/nn"
)

func TestLFM2ForwardSmoke(t *testing.T) {
	if os.Getenv("LFM2_MLX_TEST") == "" {
		t.Skip("set LFM2_MLX_TEST=1 to run")
	}
	ctx := context.Background()
	th, err := mlxthread.Start("lfm2", func() error {
		mlx.EnableCompile()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer th.Stop(ctx, nil)

	var r mlxrunner.Runner
	if err := th.Do(ctx, func() error { return r.Load("lfm2-350m-mlx:4bit") }); err != nil {
		t.Fatal(err)
	}

	err = th.Do(ctx, func() error {
		m := r.Model.(*lfm2.Model)
		t.Logf("layers=%d full=%v convL=%d defaults gs=%d bits=%d mode=%q",
			m.NumLayers(), m.FullAttnIdxs, m.ConvLCache, m.QuantGroupSize, m.QuantBits, m.QuantMode)
		w1 := m.Layers[0].FeedForward.W1
		if ql, ok := w1.(*nn.QuantizedLinear); ok {
			t.Logf("w1 bits=%d gs=%d mode=%q w=%v scales=%v", ql.Bits, ql.GroupSize, ql.Mode, ql.Weight.Dims(), ql.Scales.Dims())
		} else {
			t.Logf("w1 type=%T", w1)
		}
		if m.Layers[0].Conv != nil {
			t.Logf("conv0 weight=%v groups=%d", m.Layers[0].Conv.Conv.Weight.Dims(), m.Layers[0].Conv.Conv.Groups)
		}

		tids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		b := &batch.Batch{
			InputIDs:     mlx.FromValues(tids, 1, len(tids)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tids))},
		}
		caches := m.NewCaches()

		h := m.EmbedTokens.Forward(b.InputIDs)
		mlx.Eval(h)
		t.Logf("embed ok dims=%v", h.Dims())

		normed := m.Layers[0].OperatorNorm.Forward(h, m.NormEps)
		mlx.Eval(normed)
		t.Logf("opnorm ok")
		r0 := m.Layers[0].Conv.Forward(normed, b, caches[0], 1, int32(len(tids)), m.Config)
		mlx.Eval(r0)
		t.Logf("shortconv ok dims=%v", r0.Dims())

		h1 := mlx.Add(h, r0)
		ffnIn := m.Layers[0].FFNNorm.Forward(h1, m.NormEps)
		mlx.Eval(ffnIn)
		t.Logf("ffnnorm ok")

		gate := m.Layers[0].FeedForward.W1.Forward(ffnIn)
		mlx.Eval(gate)
		t.Logf("w1 ok dims=%v dtype=%v valid=%v", gate.Dims(), gate.DType(), gate.Valid())
		up := m.Layers[0].FeedForward.W3.Forward(ffnIn)
		mlx.Eval(up)
		t.Logf("w3 ok dims=%v", up.Dims())

		sw := mlx.SwiGLU(gate, up)
		mlx.Eval(sw)
		t.Logf("swiglu ok")

		out := m.Forward(b, m.NewCaches())
		if out == nil || !out.Valid() {
			return fmt.Errorf("invalid model output")
		}
		mlx.Eval(out)
		t.Logf("full forward ok dims=%v", out.Dims())
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
