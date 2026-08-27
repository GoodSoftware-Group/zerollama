package mlxrunner

import (
	"encoding/json"
	"os"
	"strings"
)

// Prompt Lookup Decoding (PLD) follows mlx-serve src/pld_index.zig:
// n-gram match in prompt+generated, default-on, no extra weights.
// Geometry and gates are fixed to mlx-serve shipping defaults (key 3,
// draft 5, prompt 3-gram threshold 0.01). The only operator knob is
// ZEROLLAMA_MLX_PLD=off for apples-to-apples AR benches.

const (
	pldKeyLen         = 3
	pldDraftLen       = 5
	pldSpecGateNgram  = 3
	pldSpecGateThresh = 0.01

	pldRuntimeMinRounds = 5
	pldRuntimeMinAccept = 0.30
	mtpRuntimeMinRounds = 8
	mtpRuntimeMinAccept = 0.70
	pldMaxNgramLen      = 8

	// Mid-request re-enable (mlx-serve tailMatchFraction): echo of the
	// prompt in the generated tail, even when the prompt itself scored novel.
	pldReenableWindow = 16
	pldReenableThresh = 0.7
)

func pldEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("ZEROLLAMA_MLX_PLD"))) {
	case "0", "off", "false", "no":
		return false
	default:
		return true
	}
}

func pldRequested(explicit *bool) bool {
	if explicit != nil {
		return *explicit
	}
	return pldEnabled()
}

func mtpRequested(explicit *bool, sparseMoE bool) bool {
	if explicit != nil {
		return *explicit
	}
	// mlx-serve defaultEnableMtp: MoE parks the draft head unless the request
	// (or --mtp) turns it on. Dense checkpoints still use a loaded companion.
	if sparseMoE {
		return false
	}
	return true
}

func jsonInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case int64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	default:
		return 0
	}
}

func mapSparseMoE(raw map[string]any) bool {
	if raw == nil {
		return false
	}
	for _, key := range []string{"num_experts", "num_local_experts", "n_routed_experts"} {
		if jsonInt(raw[key]) > 1 {
			return true
		}
	}
	if s, ok := raw["model_type"].(string); ok && strings.Contains(strings.ToLower(s), "moe") {
		return true
	}
	if arr, ok := raw["architectures"].([]any); ok {
		for _, a := range arr {
			s, ok := a.(string)
			if ok && strings.Contains(strings.ToLower(s), "moe") {
				return true
			}
		}
	}
	for _, nest := range []string{"text_config", "language_config", "llm_config"} {
		m, ok := raw[nest].(map[string]any)
		if ok && mapSparseMoE(m) {
			return true
		}
	}
	return false
}

// configSparseMoE reports a mixture-of-experts checkpoint from HF config.json
// (num_experts / *Moe* architecture). Used only for MTP's default-off.
func configSparseMoE(data []byte) bool {
	if len(data) == 0 {
		return false
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return false
	}
	return mapSparseMoE(raw)
}

// pldFindMatch returns up to maxDraft tokens after the latest occurrence of
// key in committed, excluding the trailing key (the query itself).
func pldFindMatch(committed []int32, key []int32, maxDraft int) []int32 {
	if len(key) == 0 || maxDraft <= 0 {
		return nil
	}
	if len(committed) <= len(key) {
		return nil
	}
	lastStart := len(committed) - len(key)
	for i := lastStart - 1; i >= 0; i-- {
		if !equalTokens(committed[i:i+len(key)], key) {
			continue
		}
		draftStart := i + len(key)
		if draftStart >= len(committed) {
			return nil
		}
		take := min(maxDraft, len(committed)-draftStart)
		if take <= 0 {
			return nil
		}
		out := make([]int32, take)
		copy(out, committed[draftStart:draftStart+take])
		return out
	}
	return nil
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ngramRepeatScore is the fraction of distinct n-grams that appear at least
// twice. Novel prompts score near 0; echo/code loops score high.
func ngramRepeatScore(tokens []int32, ngramLen int) float64 {
	if ngramLen <= 0 || ngramLen > pldMaxNgramLen || len(tokens) < ngramLen {
		return 0
	}
	n := len(tokens) - ngramLen + 1
	counts := make(map[[8]int32]int, n)
	for i := 0; i < n; i++ {
		var k [8]int32
		for j := 0; j < ngramLen; j++ {
			k[j] = tokens[i+j]
		}
		counts[k]++
	}
	if len(counts) == 0 {
		return 0
	}
	repeated := 0
	for _, c := range counts {
		if c >= 2 {
			repeated++
		}
	}
	return float64(repeated) / float64(len(counts))
}

// tailMatchFraction is the fraction of the last window positions whose
// trailing keyLen-gram also appears earlier (echo of the prompt, not
// internal tail repeats). Unused by the sticky runtime gate; kept for
// rematch against mlx-serve.
func tailMatchFraction(committed []int32, window, keyLen int) float64 {
	if keyLen <= 0 || window <= 0 || len(committed) <= keyLen {
		return 0
	}
	last := len(committed)
	first := last - window
	if first < keyLen {
		first = keyLen
	}
	if first >= last {
		return 0
	}
	matched, total := 0, 0
	for i := first; i < last; i++ {
		key := committed[i-keyLen : i]
		total++
		scanEnd := i - keyLen
		for j := 0; j < scanEnd; j++ {
			if equalTokens(committed[j:j+keyLen], key) {
				matched++
				break
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(matched) / float64(total)
}
