package weightedit

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestCanonicalPyJSONNoHTMLEscape(t *testing.T) {
	v := map[string]any{
		"b": "a&b<c>d",
		"a": 1,
	}
	got, err := CanonicalPyJSON(v)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"a":1,"b":"a&b<c>d"}`
	if string(got) != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

func TestCanonicalPyJSONPatchIDStable(t *testing.T) {
	man := map[string]any{
		"format":  "geodesic-patch",
		"version": 1,
		"name":    "x",
		"source": map[string]any{
			"fingerprint": "abc",
			"config":      map[string]any{"config.json": "deadbeef"},
		},
		"tensors": []any{},
	}
	raw, err := CanonicalPyJSON(man)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	id := hex.EncodeToString(sum[:])
	raw2, err := CanonicalPyJSON(man)
	if err != nil {
		t.Fatal(err)
	}
	sum2 := sha256.Sum256(raw2)
	if hex.EncodeToString(sum2[:]) != id {
		t.Fatal("unstable canonicalization")
	}
}
