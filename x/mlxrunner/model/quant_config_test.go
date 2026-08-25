package model

import "testing"

func TestApplyQuantizationFromConfig(t *testing.T) {
	groupSize, bits, mode := 0, 0, ""
	tensorQuant := map[string]*TensorQuantInfo{}

	ApplyQuantizationFromConfig([]byte(`{
		"quantization_config": {"bits": 3, "group_size": 64, "mode": "affine"}
	}`), &QuantConfigFields{
		QuantGroupSize: &groupSize,
		QuantBits:      &bits,
		QuantMode:      &mode,
		TensorQuant:    tensorQuant,
	})
	if bits != 3 || groupSize != 64 || mode != "affine" {
		t.Fatalf("quantization_config: bits=%d group=%d mode=%q", bits, groupSize, mode)
	}

	groupSize, bits, mode = 0, 0, ""
	ApplyQuantizationFromConfig([]byte(`{
		"quantization": {"bits": 4, "group_size": 64}
	}`), &QuantConfigFields{
		QuantGroupSize: &groupSize,
		QuantBits:      &bits,
		QuantMode:      &mode,
		TensorQuant:    tensorQuant,
	})
	if bits != 4 || groupSize != 64 || mode != "affine" {
		t.Fatalf("quantization alias: bits=%d group=%d mode=%q", bits, groupSize, mode)
	}
}

// gpt-oss mxfp4-q8: global mode mxfp4, but mlp.router overrides are bits=8 with
// no mode — must resolve to int8/affine, not inherit mxfp4 (wrong packing → empty logits).
func TestApplyQuantizationFromConfigMixedMxfp4Q8Router(t *testing.T) {
	groupSize, bits, mode := 0, 0, ""
	tensorQuant := map[string]*TensorQuantInfo{}

	ApplyQuantizationFromConfig([]byte(`{
		"quantization": {
			"group_size": 32,
			"bits": 4,
			"mode": "mxfp4",
			"model.layers.0.mlp.router": {"group_size": 64, "bits": 8},
			"model.layers.0.self_attn.q_proj": {"group_size": 32, "bits": 8, "mode": "affine"}
		}
	}`), &QuantConfigFields{
		QuantGroupSize: &groupSize,
		QuantBits:      &bits,
		QuantMode:      &mode,
		TensorQuant:    tensorQuant,
	})
	if mode != "mxfp4" || bits != 4 || groupSize != 32 {
		t.Fatalf("global: bits=%d group=%d mode=%q", bits, groupSize, mode)
	}
	router := tensorQuant["model.layers.0.mlp.router.weight"]
	if router == nil {
		t.Fatal("missing router tensor quant")
	}
	if router.QuantType != "int8" || router.GroupSize != 64 {
		t.Fatalf("router = %+v, want int8 gs=64", router)
	}
	q := tensorQuant["model.layers.0.self_attn.q_proj.weight"]
	if q == nil || q.QuantType != "int8" {
		t.Fatalf("q_proj = %+v, want int8", q)
	}
}
