package gemma4smoke_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/gemma4"
	"github.com/ollama/ollama/x/models/nn"
)

func TestGemma4OptiqForwardSmoke(t *testing.T) {
	modelName := os.Getenv("GEMMA4_SMOKE_MODEL")
	if modelName == "" {
		t.Skip("set GEMMA4_SMOKE_MODEL=gemma4:26b-optiq to run")
	}
	ctx := context.Background()
	th, err := mlxthread.Start("gemma4-smoke", func() error {
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
		m := r.Model.(*gemma4.Model)
		t.Logf("layers=%d defaults gs=%d bits=%d mode=%q",
			m.NumLayers(), m.QuantGroupSize, m.QuantBits, m.QuantMode)

		if ql, ok := m.EmbedTokens.(*nn.QuantizedEmbedding); ok {
			t.Logf("embed quant bits=%d gs=%d mode=%q", ql.Bits, ql.GroupSize, ql.Mode)
		} else {
			t.Logf("embed type=%T", m.EmbedTokens)
		}

		tids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
		b := &batch.Batch{
			InputIDs:     mlx.FromValues(tids, 1, len(tids)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tids))},
		}
		h := m.EmbedTokens.Forward(b.InputIDs)
		mlx.Eval(h)
		t.Logf("embed ok dims=%v valid=%v", h.Dims(), h.Valid())
		if !h.Valid() {
			return fmt.Errorf("embed invalid")
		}
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
