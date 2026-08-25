package qwen35smoke_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	qwen35 "github.com/ollama/ollama/x/models/qwen3_5"
)

func TestBonsaiOrnithForwardSmoke(t *testing.T) {
	modelName := os.Getenv("QWEN35_SMOKE_MODEL")
	if modelName == "" {
		t.Skip("set QWEN35_SMOKE_MODEL=bonsai:27b-mlx (or ornith-9b-optiq) to run")
	}
	ctx := context.Background()
	th, err := mlxthread.Start("qwen35-smoke", func() error {
		mlx.EnableCompile()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer th.Stop(ctx, nil)

	var r mlxrunner.Runner
	if err := th.Do(ctx, func() error { return r.Load(modelName) }); err != nil {
		t.Fatal(err)
	}

	err = th.Do(ctx, func() error {
		m := r.Model.(*qwen35.Model)
		t.Logf("layers=%d defaults gs=%d bits=%d mode=%q linear_conv=%d",
			m.NumLayers(), m.QuantGroupSize, m.QuantBits, m.QuantMode, m.LinearConvKernelDim)

		var linIdx int
		var lin *qwen35.GatedDeltaNet
		for i, layer := range m.Layers {
			if layer.IsLinear && layer.Linear != nil {
				linIdx, lin = i, layer.Linear
				break
			}
		}
		if lin == nil {
			return fmt.Errorf("no linear layer found")
		}
		t.Logf("first linear layer=%d conv_dims=%v groups=%d", linIdx, lin.Conv1D.Weight.Dims(), lin.Conv1D.Groups)

		dumpQ := func(name string, l nn.LinearLayer) {
			if l == nil {
				t.Logf("%s=<nil>", name)
				return
			}
			if ql, ok := l.(*nn.QuantizedLinear); ok {
				t.Logf("%s quant bits=%d gs=%d mode=%q w=%v scales=%v",
					name, ql.Bits, ql.GroupSize, ql.Mode, ql.Weight.Dims(), ql.Scales.Dims())
			} else {
				t.Logf("%s type=%T", name, l)
			}
		}
		dumpQ("in_proj_qkv", lin.InProjQKV)
		dumpQ("in_proj_z", lin.InProjZ)
		dumpQ("in_proj_qkvz", lin.InProjQKVZ)
		dumpQ("out_proj", lin.OutProj)

		tids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		b := &batch.Batch{
			InputIDs:     mlx.FromValues(tids, 1, len(tids)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tids))},
		}
		caches := m.NewCaches()

		h := m.EmbedTokens.Forward(b.InputIDs)
		mlx.Eval(h)
		t.Logf("embed ok dims=%v dtype=%v valid=%v", h.Dims(), h.DType(), h.Valid())
		if !h.Valid() {
			return fmt.Errorf("embed invalid")
		}

		layer := m.Layers[linIdx]
		if layer.InputNorm != nil && layer.InputNorm.Weight != nil {
			t.Logf("input_norm weight dims=%v dtype=%v valid=%v eps=%v",
				layer.InputNorm.Weight.Dims(), layer.InputNorm.Weight.DType(), layer.InputNorm.Weight.Valid(), m.RMSNormEps)
		}
		normed := layer.InputNorm.Forward(h, m.RMSNormEps)
		mlx.Eval(normed)
		t.Logf("input_norm ok dims=%v valid=%v dtype=%v", normed.Dims(), normed.Valid(), normed.DType())
		if !normed.Valid() {
			// Try raw RMSNormFn for diagnosis
			raw := mlx.RMSNormFn(h, layer.InputNorm.Weight, m.RMSNormEps)
			mlx.Eval(raw)
			t.Logf("raw RMSNormFn dims=%v valid=%v", raw.Dims(), raw.Valid())
			return fmt.Errorf("input_norm invalid")
		}

		g := lin
		var qkv *mlx.Array
		if g.InProjQKV != nil {
			qkv = g.InProjQKV.Forward(normed)
		} else {
			mixedQKVZ := g.InProjQKVZ.Forward(normed)
			mlx.Eval(mixedQKVZ)
			t.Logf("in_proj_qkvz ok dims=%v valid=%v", mixedQKVZ.Dims(), mixedQKVZ.Valid())
			return fmt.Errorf("combined qkvz path — inspect manually (unexpected for bonsai/ornith)")
		}
		mlx.Eval(qkv)
		t.Logf("in_proj_qkv ok dims=%v dtype=%v valid=%v", qkv.Dims(), qkv.DType(), qkv.Valid())
		if !qkv.Valid() {
			return fmt.Errorf("in_proj_qkv invalid (bad quant)")
		}

		convTail := int(m.LinearConvKernelDim - 1)
		rc := caches[linIdx].(*cache.RecurrentCache)
		opts := []nn.RecurrentOption{nn.WithRecurrentHistory(rc.Get(b, normed.DType()))}
		convOut, convStates := nn.CausalConv1D(b, qkv, g.Conv1D, convTail, opts...)
		mlx.Eval(convOut)
		t.Logf("causal_conv ok dims=%v valid=%v n_states=%d", convOut.Dims(), convOut.Valid(), len(convStates))
		if !convOut.Valid() {
			return fmt.Errorf("causal_conv invalid")
		}

		silu := mlx.SiLU(convOut)
		mlx.Eval(silu)
		t.Logf("silu ok dims=%v valid=%v", silu.Dims(), silu.Valid())

		// Full model forward
		out := m.Forward(b, m.NewCaches())
		if out == nil || !out.Valid() {
			return fmt.Errorf("full forward invalid")
		}
		mlx.Eval(out)
		t.Logf("full forward ok dims=%v", out.Dims())
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
