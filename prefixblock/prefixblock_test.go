package prefixblock

import "testing"

func TestHashMatchesPythonGolden(t *testing.T) {
	scope := ModelScope("testhash", "")
	tokens := make([]int32, 16)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	blocks := Iter(tokens, 8, scope, 0)
	if len(blocks) != 2 {
		t.Fatalf("blocks=%d want 2", len(blocks))
	}
	want0 := "4f43c6d184cc797987773c7f9400baac32f22f3248e0f1b2dd95f158ef869cf1"
	want1 := "5ca1fab98e8158937ae2c0dfb410065cb2474c363551c8fc262f60a7c561a03b"
	if blocks[0].Hash != want0 {
		t.Fatalf("h0=%s want %s", blocks[0].Hash, want0)
	}
	if blocks[1].Hash != want1 {
		t.Fatalf("h1=%s want %s", blocks[1].Hash, want1)
	}
	if blocks[0].ParentHash != RootHash {
		t.Fatalf("parent0=%s", blocks[0].ParentHash)
	}
	if blocks[1].ParentHash != want0 {
		t.Fatalf("parent1=%s", blocks[1].ParentHash)
	}
}

func TestModelScopeSaltMatchesPython(t *testing.T) {
	scope := ModelScope("testhash", "tenant")
	tokens := make([]int32, 8)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	got := Hash(scope, RootHash, 0, tokens)
	want := "3d0acd81214c6d8a78e081d8d41ad0d3f3230269f623e7ef27d9d77a740f8798"
	if got != want {
		t.Fatalf("got %s want %s", got, want)
	}
}

func TestHashesHelper(t *testing.T) {
	scope := ModelScope("testhash", "")
	tokens := make([]int32, 16)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	hs := Hashes(tokens, 8, scope, 0)
	if len(hs) != 2 {
		t.Fatalf("len=%d", len(hs))
	}
}
