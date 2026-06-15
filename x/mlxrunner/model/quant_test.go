package model

import (
	"encoding/json"
	"testing"
)

func TestInferQuantTypeFromShapesGemma3QAT(t *testing.T) {
	header := map[string]json.RawMessage{
		"language_model.model.layers.0.mlp.gate_proj.weight":       mustRaw(`{"shape":[21504,672],"dtype":"U32"}`),
		"language_model.model.layers.0.mlp.gate_proj.weight.scale": mustRaw(`{"shape":[21504,84],"dtype":"BF16"}`),
	}
	qt, gs := inferQuantTypeFromShapes(header, "language_model.model.layers.0.mlp.gate_proj.weight", "")
	if qt != "INT4" || gs != 64 {
		t.Fatalf("got %q gs=%d, want INT4 gs=64", qt, gs)
	}
}

func TestInferAffineQuantParamsGemma3QAT(t *testing.T) {
	// Mirrors 21504x672 weight / 21504x84 scales without loading MLX arrays.
	gs, bits, ok := inferAffineFromColCounts(672, 84, 0)
	if !ok || gs != 64 || bits != 4 {
		t.Fatalf("got gs=%d bits=%d ok=%v, want 64/4/true", gs, bits, ok)
	}
}

func inferAffineFromColCounts(weightCols, scalesCols, hintBits int) (int, int, bool) {
	groupSize4 := weightCols * 8 / scalesCols
	groupSize8 := weightCols * 4 / scalesCols
	switch {
	case groupSize4 == 32:
		return 32, 4, true
	case groupSize8 == 64:
		return 64, 8, true
	case groupSize4 == 64 && groupSize8 == 32:
		if hintBits == 8 {
			return 32, 8, true
		}
		return 64, 4, true
	}
	return 0, 0, false
}

func mustRaw(s string) json.RawMessage { return json.RawMessage(s) }
