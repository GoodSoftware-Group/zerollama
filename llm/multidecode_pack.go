package llm

import "fmt"

// PackForest is a one-shot MultiDecode forest (tokens/pos/parent/leaves).
type PackForest struct {
	Tokens []int32
	Pos    []int32
	Parent []int32
	Leaves []int32
	// PrefixLen is the longest common prefix length across input sequences.
	PrefixLen int
}

// LongestCommonPrefixLen returns the shared prefix length of token sequences.
func LongestCommonPrefixLen(seqs [][]int) int {
	if len(seqs) == 0 {
		return 0
	}
	lcp := len(seqs[0])
	for _, s := range seqs[1:] {
		n := len(s)
		if n < lcp {
			lcp = n
		}
		for i := 0; i < lcp; i++ {
			if s[i] != seqs[0][i] {
				lcp = i
				break
			}
		}
	}
	return lcp
}

// PackForestFromTokenLists builds a shared-prefix forest for MultiDecode.
//
// Shared prefix is emitted once; each sequence's unique suffix is a branch.
// Identical sequences share one leaf index. Requires at least one non-empty sequence.
func PackForestFromTokenLists(seqs [][]int) (PackForest, error) {
	if len(seqs) == 0 {
		return PackForest{}, fmt.Errorf("pack forest: empty sequences")
	}
	for i, s := range seqs {
		if len(s) == 0 {
			return PackForest{}, fmt.Errorf("pack forest: sequence %d is empty", i)
		}
	}
	lcp := LongestCommonPrefixLen(seqs)
	out := PackForest{PrefixLen: lcp}
	// Shared prefix
	for i := 0; i < lcp; i++ {
		out.Tokens = append(out.Tokens, int32(seqs[0][i]))
		out.Pos = append(out.Pos, int32(i))
		if i == 0 {
			out.Parent = append(out.Parent, -1)
		} else {
			out.Parent = append(out.Parent, int32(i-1))
		}
	}
	prefixLeaf := int32(lcp - 1) // -1 when lcp==0
	out.Leaves = make([]int32, len(seqs))
	for si, seq := range seqs {
		curParent := prefixLeaf
		curPos := lcp
		if len(seq) == lcp {
			if lcp == 0 {
				return PackForest{}, fmt.Errorf("pack forest: empty after lcp")
			}
			out.Leaves[si] = prefixLeaf
			continue
		}
		for j := lcp; j < len(seq); j++ {
			idx := int32(len(out.Tokens))
			out.Tokens = append(out.Tokens, int32(seq[j]))
			out.Pos = append(out.Pos, int32(curPos))
			out.Parent = append(out.Parent, curParent)
			curParent = idx
			curPos++
		}
		out.Leaves[si] = curParent
	}
	return out, nil
}

// DefaultMultiDecodeLCPThreshold is the minimum shared prefix (tokens) before
// silent Hermes batch packing prefers MultiDecode over Python generate_batch.
const DefaultMultiDecodeLCPThreshold = 32
