package base

import "testing"

func TestArchitectureFromConfig(t *testing.T) {
	arch, err := architectureFromConfig([]byte(`{"model_type":"siglip","vision_config":{"model_type":"siglip_vision_model"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if arch != "SiglipModel" {
		t.Fatalf("arch = %q, want SiglipModel", arch)
	}

	arch, err = architectureFromConfig([]byte(`{"architectures":["Siglip2Model"],"model_type":"siglip"}`))
	if err != nil {
		t.Fatal(err)
	}
	if arch != "Siglip2Model" {
		t.Fatalf("explicit architectures lost: %q", arch)
	}

	if _, err := architectureFromConfig([]byte(`{"model_type":"llama"}`)); err == nil {
		t.Fatal("expected missing architecture to fail")
	}
}
