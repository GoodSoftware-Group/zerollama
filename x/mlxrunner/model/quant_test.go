package model

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestResolveLinearQuantParamsShapeOverridesMetadata(t *testing.T) {
	weight := mlx.FromValues(make([]uint32, 704), 1, 704)
	scales := mlx.FromValues(make([]float32, 44), 1, 44)

	tq := map[string]*TensorQuantInfo{
		"language_model.model.embed_tokens.weight": {
			QuantType: "int4",
			GroupSize: 64,
		},
	}

	groupSize, bits, mode := ResolveLinearQuantParams(
		64, 4, "affine",
		tq,
		"language_model.model.embed_tokens.weight",
		weight,
		scales,
	)
	if mode != "affine" {
		t.Fatalf("mode = %q, want affine", mode)
	}
	if bits != 8 {
		t.Fatalf("bits = %d, want 8", bits)
	}
	if groupSize != 64 {
		t.Fatalf("groupSize = %d, want 64", groupSize)
	}
}

func TestAffinePackedShapeMatches(t *testing.T) {
	weight := mlx.FromValues(make([]uint32, 704), 1, 704)
	scales := mlx.FromValues(make([]float32, 44), 1, 44)
	if !affinePackedShapeMatches(weight, scales, 64, 8) {
		t.Fatal("expected int8 packed shape to match")
	}
	if affinePackedShapeMatches(weight, scales, 64, 4) {
		t.Fatal("int4 metadata should not match int8-packed weights")
	}
}
