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
