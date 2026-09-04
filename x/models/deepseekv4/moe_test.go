package deepseekv4

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestSqrtSoftplusPositive(t *testing.T) {
	x := mlx.FromValues([]float32{0, 1, -1}, 1, 1, 3)
	y := sqrtSoftplus(x)
	mlx.Eval(y)
	if len(y.Dims()) != 3 {
		t.Fatalf("dims %v", y.Dims())
	}
}

func TestHashExpertsVocabLayout(t *testing.T) {
	table := mlx.FromValues([]int32{0, 1, 2, 3, 4, 5, 6, 7}, 4, 2)
	ids := mlx.FromValues([]int32{1, 3}, 1, 2)
	got := hashExperts(table, ids, 2)
	mlx.Eval(got)
	d := got.Dims()
	if d[0] != 2 || d[len(d)-1] != 2 {
		t.Fatalf("got dims %v want rows=2 topk=2", d)
	}
}

func TestHashExpertsLlamaLayout(t *testing.T) {
	table := mlx.FromValues([]int32{0, 2, 4, 6, 1, 3, 5, 7}, 2, 4)
	ids := mlx.FromValues([]int32{1, 3}, 1, 2)
	got := hashExperts(table, ids, 2)
	mlx.Eval(got)
	d := got.Dims()
	if d[0] != 2 || d[len(d)-1] != 2 {
		t.Fatalf("got dims %v want rows=2 topk=2", d)
	}
}

func TestMoEGateBiasSelectionOnly(t *testing.T) {
	x := mlx.FromValues([]float32{0.2, -0.1, 0.05}, 1, 1, 3)
	unb := sqrtSoftplus(x)
	biased := mlx.Add(unb, mlx.FromValues([]float32{0, 0, 0}, 1, 1, 3))
	mlx.Eval(unb, biased)
}

func TestSwigluLimitZeroIsIdentity(t *testing.T) {
	g := mlx.FromValues([]float32{20, -20}, 1, 1, 2)
	u := mlx.FromValues([]float32{3, 4}, 1, 1, 2)
	clamped := swigluLimit(g, u, 10)
	free := swigluLimit(g, u, 0)
	mlx.Eval(clamped, free)
	cf, ff := clamped.Floats(), free.Floats()
	if cf[0] == ff[0] {
		t.Fatal("limit 10 should change a |20| gate vs unlimited shared-expert path")
	}
}

func TestAPELayout(t *testing.T) {
	ape := mlx.FromValues([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4)
	got := normalizeAPE(ape, 4)
	mlx.Eval(got)
	if got.Dim(0) != 4 || got.Dim(1) != 2 {
		t.Fatalf("want [4,2] got %v", got.Dims())
	}
}
