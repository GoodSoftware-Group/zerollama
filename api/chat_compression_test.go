package api

import "testing"

func TestApplyStickyChatCompression(t *testing.T) {
	req := &ChatRequest{}
	ApplyStickyChatCompression(req, nil)
	if req.Compression != nil {
		t.Fatal("nil meta should not allocate compression")
	}

	ApplyStickyChatCompression(req, &ChatCompressionMeta{ElideFrom: 3})
	if req.Compression != nil {
		t.Fatal("empty Mode means compression did not run")
	}

	ApplyStickyChatCompression(req, &ChatCompressionMeta{Mode: "placeholder", ElideFrom: 0})
	if req.Compression == nil || req.Compression.ElideFrom == nil || *req.Compression.ElideFrom != 0 {
		t.Fatalf("want sticky 0, got %+v", req.Compression)
	}

	explicit := 9
	req.Compression.ElideFrom = &explicit
	ApplyStickyChatCompression(req, &ChatCompressionMeta{Mode: "placeholder", ElideFrom: 2})
	if *req.Compression.ElideFrom != 9 {
		t.Fatal("explicit ElideFrom must win")
	}
}
