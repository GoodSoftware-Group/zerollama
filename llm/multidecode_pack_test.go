package llm

import "testing"

func TestLongestCommonPrefixLen(t *testing.T) {
	if got := LongestCommonPrefixLen([][]int{{1, 2, 3}, {1, 2, 9}, {1, 2, 3, 4}}); got != 2 {
		t.Fatalf("lcp=%d want 2", got)
	}
	if got := LongestCommonPrefixLen([][]int{{1}, {2}}); got != 0 {
		t.Fatalf("lcp=%d want 0", got)
	}
}

func TestPackForestFromTokenLists(t *testing.T) {
	seqs := [][]int{
		{10, 11, 12, 20},
		{10, 11, 12, 30, 31},
	}
	f, err := PackForestFromTokenLists(seqs)
	if err != nil {
		t.Fatal(err)
	}
	if f.PrefixLen != 3 {
		t.Fatalf("prefix=%d", f.PrefixLen)
	}
	// prefix 3 + branch0 (1) + branch1 (2) = 6
	if len(f.Tokens) != 6 {
		t.Fatalf("tokens=%v", f.Tokens)
	}
	if f.Leaves[0] != 3 || f.Leaves[1] != 5 {
		t.Fatalf("leaves=%v", f.Leaves)
	}
	// Identical prompts share leaf
	f2, err := PackForestFromTokenLists([][]int{{1, 2, 3}, {1, 2, 3}})
	if err != nil {
		t.Fatal(err)
	}
	if f2.Leaves[0] != f2.Leaves[1] || f2.Leaves[0] != 2 {
		t.Fatalf("identical leaves=%v", f2.Leaves)
	}
}
