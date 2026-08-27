package mlxrunner

import (
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/tokenizer"
)

func suppressReservedEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_SUPPRESS_RESERVED"))) {
	case "0", "off", "false", "no":
		return false
	default:
		return true
	}
}

// reservedSampleBanIDs lists tokenizer ids that must never be sampled
// (mlx-serve reservedOutputIds). Thinking tags and tool markers stay legal.
func reservedSampleBanIDs(tok *tokenizer.Tokenizer) []int32 {
	if tok == nil {
		return nil
	}
	seen := make(map[int32]struct{})
	var out []int32
	add := func(id int32) {
		if id < 0 || tok.IsEOS(id) {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	for name, id := range tok.SpecialTokens() {
		if banReservedSampleText(name) {
			add(id)
		}
	}
	add(tok.PAD())
	add(tok.BOS())
	slices.Sort(out)
	return out
}

func banReservedSampleText(s string) bool {
	l := strings.ToLower(strings.TrimSpace(s))
	if l == "" {
		return false
	}
	if strings.Contains(l, "think") || strings.Contains(l, "tool") ||
		strings.Contains(l, "im_end") || strings.Contains(l, "im_start") ||
		strings.Contains(l, "eot") || strings.Contains(l, "end_of_turn") {
		return false
	}
	if strings.Contains(l, "fim") || strings.Contains(l, "hole") ||
		strings.Contains(l, "unused") || strings.Contains(l, "reserved") {
		return true
	}
	switch l {
	case "<unk>", "<pad>", "<mask>", "<|unk|>", "<|pad|>", "<|mask|>", "[unk]", "[pad]", "[mask]":
		return true
	}
	return false
}
