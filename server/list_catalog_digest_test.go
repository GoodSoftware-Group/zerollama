package server

import "testing"

func TestListCatalogDigest(t *testing.T) {
	a := listCatalogDigest("eliza:acme/foo")
	b := listCatalogDigest("eliza:acme/foo")
	c := listCatalogDigest("eliza:acme/bar")
	if len(a) != 64 {
		t.Fatalf("len=%d want 64", len(a))
	}
	if a != b {
		t.Fatalf("not stable: %q vs %q", a, b)
	}
	if a == c {
		t.Fatalf("collision across seeds")
	}
	// Stock ollama clients slice [:12]; empty would panic.
	if a[:12] == "" {
		t.Fatal("unexpected empty prefix")
	}
}
