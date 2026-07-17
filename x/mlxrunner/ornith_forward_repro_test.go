package mlxrunner_test

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/models/nn"
	qwen35 "github.com/ollama/ollama/x/models/qwen3_5"
)

// Regression: ornith-9b-optiq mixed-precision down_proj packing is ambiguous
// between (bits=4,gs=64) and (bits=8,gs=32). Wrong resolution made layer-8+
// QuantizedMatmul return an invalid array and panicked inside compiled SiLU.
func TestOrnithOptiqForward(t *testing.T) {
	if os.Getenv("ORNITH_OPTIQ_TEST") == "" {
		t.Skip("set ORNITH_OPTIQ_TEST=1 to run")
	}
	ctx := context.Background()
	th, err := mlxthread.Start("ornith", func() error {
		mlx.EnableCompile()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer th.Stop(ctx, nil)

	var r mlxrunner.Runner
	if err := th.Do(ctx, func() error { return r.Load("ornith-9b-optiq") }); err != nil {
		t.Fatal(err)
	}

	err = th.Do(ctx, func() error {
		m := r.Model.(*qwen35.Model)
		for _, li := range []int{8, 9} {
			ql := m.Layers[li].MLP.(*qwen35.DenseMLP).DownProj.(*nn.QuantizedLinear)
			if ql.Bits != 4 || ql.GroupSize != 64 {
				return fmt.Errorf("layer %d down_proj got bits=%d gs=%d, want 4/64", li, ql.Bits, ql.GroupSize)
			}
		}
		tids := r.Tokenizer.Encode("Say hello in one word.", false)
		out := m.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues(tids, 1, len(tids)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tids))},
		}, m.NewCaches())
		if out == nil || !out.Valid() {
			return fmt.Errorf("invalid model output")
		}
		mlx.Eval(out)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
