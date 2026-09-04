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

func TestRemapLoadedTensorsKeepsHCScale(t *testing.T) {
	raw := map[string]*mlx.Array{
		"model.hc_head.fn":    dummyArray(),
		"model.hc_head.base":  dummyArray(),
		"model.hc_head.scale": dummyArray(),
		"linear.weight":       dummyArray(),
		"linear.weight.scale": dummyArray(),
		"linear.weight.bias":  dummyArray(),
	}
	got := remapLoadedTensors(raw)
	if _, ok := got["model.hc_head.scale"]; !ok {
		t.Fatalf("hc_head.scale dropped; keys %v", keys(got))
	}
	if _, ok := got["linear.weight_scale"]; !ok {
		t.Fatalf("weight.scale not remapped; keys %v", keys(got))
	}
	if _, ok := got["linear.weight.scale"]; ok {
		t.Fatal("raw weight.scale should be remapped away")
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
