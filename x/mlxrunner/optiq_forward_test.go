package mlxrunner_test

import (
	"context"
	"os"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestGemma4OptiqForward(t *testing.T) {
	if os.Getenv("GEMMA4_OPTIQ_TEST") == "" {
		t.Skip("set GEMMA4_OPTIQ_TEST=1 to run")
	}

	ctx := context.Background()
	th, err := mlxthread.Start("test", func() error {
		mlx.EnableCompile()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer th.Stop(ctx, nil)

	var r mlxrunner.Runner
	if err := th.Do(ctx, func() error {
		return r.Load("gemma4:26b-optiq")
	}); err != nil {
		t.Fatal(err)
	}

	err = th.Do(ctx, func() error {
		tids := r.Tokenizer.Encode("Say hello", false)
		caches := make([]cache.Cache, r.Model.NumLayers())
		for i := range caches {
			caches[i] = cache.NewKVCache()
		}
		out := r.Model.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues(tids, 1, len(tids)),
			SeqOffsets:   []int32{0},
			SeqQueryLens: []int32{int32(len(tids))},
		}, caches)
		if !out.Valid() {
			t.Fatal("invalid forward output")
		}
		dims := out.Dims()
		if len(dims) != 3 || dims[2] != 2816 {
			t.Fatalf("unexpected output dims %v", dims)
		}
		mlx.Eval(out)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
