package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestRemapLoadedTensorsAffineScales(t *testing.T) {
	raw := map[string]*mlx.Array{
		"draft.fc.weight": dummyArray(),
		"draft.fc.scales": dummyArray(),
		"draft.fc.biases": dummyArray(),
	}
	got := remapLoadedTensors(raw)
	if _, ok := got["draft.fc.weight_scale"]; !ok {
		t.Fatalf("keys %v", keys(got))
	}
	if _, ok := got["draft.fc.weight_qbias"]; !ok {
		t.Fatalf("missing weight_qbias in %v", keys(got))
	}
	if _, ok := got["draft.fc.scales"]; ok {
		t.Fatal("raw .scales should be remapped away")
	}
}

func dummyArray() *mlx.Array { return &mlx.Array{} }

func keys(m map[string]*mlx.Array) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
