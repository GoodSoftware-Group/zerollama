package tensorproto

import "testing"

func TestRequestValidate(t *testing.T) {
	t.Parallel()
	if err := (Request{}).Validate(); err == nil {
		t.Fatal("expected error")
	}
	r := Request{Digest: "sha256-x", TensorRef: "layer.0.ffn", Offset: -1}
	if err := r.Validate(); err == nil {
		t.Fatal("expected negative offset error")
	}
	r.Offset = 0
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
}
