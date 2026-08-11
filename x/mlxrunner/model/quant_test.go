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
	// Metadata says int4/gs64 which does not pack; with bits=4 hint the
	// matching packing is gs128 (4/128 and 8/64 are both shape-legal).
	if bits != 4 {
		t.Fatalf("bits = %d, want 4", bits)
	}
	if groupSize != 128 {
		t.Fatalf("groupSize = %d, want 128", groupSize)
	}
}

func TestResolveLinearQuantParamsConfigBeatsAmbiguousBlob(t *testing.T) {
	// bonsai-style: 2/128 and 4/64 share packed shapes; config defaults win
	// even if blob TensorQuant guessed int4.
	weight := mlx.FromValues(make([]uint32, 320), 1, 320)
	scales := mlx.FromValues(make([]float32, 40), 1, 40)

	tq := map[string]*TensorQuantInfo{
		"language_model.model.embed_tokens.weight": {
			QuantType: "int4",
			GroupSize: 64,
		},
	}

	groupSize, bits, mode := ResolveLinearQuantParams(
		128, 2, "affine",
		tq,
		"language_model.model.embed_tokens.weight",
		weight,
		scales,
	)
	if mode != "affine" || bits != 2 || groupSize != 128 {
		t.Fatalf("got gs=%d bits=%d mode=%q, want 128/2/affine", groupSize, bits, mode)
	}
}

func TestResolveLinearQuantParamsTensorMetaWhenDefaultsMismatch(t *testing.T) {
	// OptIQ: defaults 4/64, tensor is int8/64 (same shape as 4/128).
	weight := mlx.FromValues(make([]uint32, 1024), 1, 1024)
	scales := mlx.FromValues(make([]float32, 64), 1, 64)

	tq := map[string]*TensorQuantInfo{
		"language_model.model.embed_tokens.weight": {
			QuantType: "int8",
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
	if mode != "affine" || bits != 8 || groupSize != 64 {
		t.Fatalf("got gs=%d bits=%d mode=%q, want 64/8/affine", groupSize, bits, mode)
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
