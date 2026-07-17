// Package fleet — multi-node routing helpers.
//
// ScoreCandidates ranks peers for a model request (warmth, queue, session affinity).
// Why: warm-model-first alone ignores loading state and agent thread stickiness;
// filter-then-score keeps assign deterministic and testable. Lower score wins.
package fleet

import (
	"strings"
	"time"
)

// Routing score weights (lower total wins). Tunable constants — not env yet.
const (
	scoreWarmBonus       = -10_000
	scoreAffinityBonus   = -5_000
	scoreRadixBonus      = -2_000 // L3-R8/LA13: prefer peers with warm prefix block pool + radix_share
	scoreRadixPerBlock   = -2     // soft density hint (capped)
	scoreRadixBlockCap   = 500
	scoreRadixHashBlock  = -80    // L3-R9: per leading matched content-hash block
	scoreRadixHashCap    = 64
	scoreRadixDigestBonus = -400 // L3-R11: peer can HTTP-serve slot blobs for matched prefix
	scoreQueueWeight     = 100
	scoreLoadingPenalty  = 500
	scoreColdPenalty     = 2_000 // applied only when prefer_warm and no warm nodes remain after filter
	scoreResidentPenalty = 300   // per other loaded model on the node
	scoreLoadedCtxPerK   = 5     // per 1024 effective ctx from non-request residents
)

// ScoreRequest is POST /internal/score (and the scoring input for assign).
type ScoreRequest struct {
	Model             string   `json:"model"`
	PreferWarm        *bool    `json:"prefer_warm,omitempty"`
	WarmOnly          bool     `json:"warm_only,omitempty"`
	Exclude           []string `json:"exclude,omitempty"`
	SessionKey        string   `json:"session_key,omitempty"`
	PromptCacheKey    string   `json:"prompt_cache_key,omitempty"`
	PrefixBlockHashes []string `json:"prefix_block_hashes,omitempty"` // L3-R9 / LA13 ordered hashes from token 0
}

// ScoredNode is one ranked fleet peer for a model request.
type ScoredNode struct {
	NodeSnapshot
	Score    float64  `json:"score"`
	Warm     bool     `json:"warm"`
	Affinity bool     `json:"affinity,omitempty"`
	Reasons  []string `json:"reasons,omitempty"`
}

// ScoreResponse is POST /internal/score.
type ScoreResponse struct {
	Model       string       `json:"model"`
	Candidates  []ScoredNode `json:"candidates"`
	Best        *ScoredNode  `json:"best,omitempty"`
	GeneratedAt time.Time    `json:"generated_at"`
}

// ScoreCandidates ranks available nodes for a model. Lower score is better.
func ScoreCandidates(nodes []NodeSnapshot, req ScoreRequest, cache *PrefixCache) ScoreResponse {
	model := strings.TrimSpace(req.Model)
	now := time.Now().UTC()

	preferWarm := true
	if req.PreferWarm != nil {
		preferWarm = *req.PreferWarm
	}

	excluded := excludedSet(req.Exclude)
	candidates := filterCandidates(nodes, excluded)

	sessionKey := strings.TrimSpace(req.SessionKey)
	if sessionKey == "" {
		sessionKey = strings.TrimSpace(req.PromptCacheKey)
	}

	affinityID := ""
	if cache != nil && sessionKey != "" {
		if id, ok := cache.PreferredNode(model, sessionKey); ok {
			affinityID = id
		}
	}

	scored := make([]ScoredNode, 0, len(candidates))
	for _, n := range candidates {
		affinity := affinityID != "" && strings.EqualFold(n.ID, affinityID)
		warm := nodeHasModel(n, model)
		score, reasons := nodeScore(n, model, warm, affinity, preferWarm, sessionKey != "", req.PrefixBlockHashes)
		scored = append(scored, ScoredNode{
			NodeSnapshot: n,
			Score:        score,
			Warm:         warm,
			Affinity:     affinity,
			Reasons:      reasons,
		})
	}

	scored = filterScoredForPolicy(scored, preferWarm, req.WarmOnly)
	sortScored(scored)

	var best *ScoredNode
	if len(scored) > 0 {
		best = &scored[0]
	}

	return ScoreResponse{
		Model:       model,
		Candidates:  scored,
		Best:        best,
		GeneratedAt: now,
	}
}

func nodeScore(n NodeSnapshot, assignModel string, warm, affinity, preferWarm, sessionHint bool, prefixHashes []string) (float64, []string) {
	var score float64
	reasons := make([]string, 0, 8)

	if warm {
		score += scoreWarmBonus
		reasons = append(reasons, "warm")
	} else if preferWarm {
		score += scoreColdPenalty
		reasons = append(reasons, "cold")
	}

	if affinity {
		score += scoreAffinityBonus
		reasons = append(reasons, "affinity")
	}

	if hashBonus, why := nodeRadixHashBonus(n, prefixHashes); hashBonus != 0 {
		score += hashBonus
		reasons = append(reasons, why...)
	} else if sessionHint {
		if bonus, why := nodeRadixResidencyBonus(n); bonus != 0 {
			score += bonus
			reasons = append(reasons, why...)
		}
	}

	if n.QueueDepth > 0 {
		score += float64(n.QueueDepth) * scoreQueueWeight
		reasons = append(reasons, "queue")
	}
	if n.Loading {
		score += scoreLoadingPenalty
		reasons = append(reasons, "loading")
	}

	if capPenalty, capReasons := nodeCapacityPenalty(n, assignModel); capPenalty > 0 {
		score += capPenalty
		reasons = append(reasons, capReasons...)
	}

	return score, reasons
}

// nodeRadixHashBonus prefers peers whose advertised block_hashes cover the
// client's leading prefix chain (L3-R9 / full LA13). Soft — agents still pay
// for cold restore if hashes are stale.
func nodeRadixHashBonus(n NodeSnapshot, want []string) (float64, []string) {
	if !fleetRadixHashScoreEnabled() || len(want) == 0 {
		return 0, nil
	}
	r := n.Inference.Runtime.Radix
	if r == nil || !r.Enabled || len(r.BlockHashes) == 0 {
		return 0, nil
	}
	matched := longestPrefixHashMatch(want, r.BlockHashes)
	if matched <= 0 {
		return 0, nil
	}
	if matched > scoreRadixHashCap {
		matched = scoreRadixHashCap
	}
	bonus := float64(matched) * scoreRadixHashBlock
	// Keep a light residency nudge when radix_share is on (same peer likely can seed).
	if r.RadixShare {
		bonus += float64(scoreRadixBonus) / 4
	}
	// Prefer peers that advertise pullable digests (L3-R10/R11 cold restore path).
	if r.BlobDigestBlocks > 0 || len(r.BlobDigests) > 0 {
		bonus += float64(scoreRadixDigestBonus)
		return bonus, []string{"radix_hash", "radix_blob"}
	}
	return bonus, []string{"radix_hash"}
}

// nodeRadixResidencyBonus prefers peers whose /api/status mirrors a warm Python
// prefix block pool with radix_share (L3-R8). Soft signal when hashes are absent.
func nodeRadixResidencyBonus(n NodeSnapshot) (float64, []string) {
	if !fleetRadixScoreEnabled() {
		return 0, nil
	}
	r := n.Inference.Runtime.Radix
	if r == nil || !r.Enabled || !r.RadixShare || r.EntryCount <= 0 {
		return 0, nil
	}
	bonus := float64(scoreRadixBonus)
	blocks := r.EntryCount
	if blocks > scoreRadixBlockCap {
		blocks = scoreRadixBlockCap
	}
	bonus += float64(blocks) * scoreRadixPerBlock
	return bonus, []string{"radix"}
}

func nodeCapacityPenalty(n NodeSnapshot, assignModel string) (float64, []string) {
	details := n.Inference.Ggml.LoadedModelDetails
	if len(details) == 0 {
		others := 0
		for _, loaded := range n.LoadedModels {
			if !modelMatches(loaded, assignModel) {
				others++
			}
		}
		if others == 0 {
			return 0, nil
		}
		return float64(others) * scoreResidentPenalty, []string{"residents"}
	}

	var penalty float64
	var reasons []string
	others := 0
	totalCtx := 0
	for _, d := range details {
		if modelMatches(d.Name, assignModel) {
			continue
		}
		others++
		ctx := d.NumCtx
		if ctx <= 0 {
			ctx = d.ManifestNumCtx
		}
		if ctx > 0 {
			totalCtx += ctx
		}
	}
	if others > 0 {
		penalty += float64(others) * scoreResidentPenalty
		reasons = append(reasons, "residents")
	}
	if totalCtx > 0 {
		penalty += float64(totalCtx/1024) * scoreLoadedCtxPerK
		if totalCtx >= 8192 {
			reasons = append(reasons, "ctx_pressure")
		}
	}
	return penalty, reasons
}

func filterScoredForPolicy(scored []ScoredNode, preferWarm, warmOnly bool) []ScoredNode {
	if warmOnly {
		out := make([]ScoredNode, 0, len(scored))
		for _, s := range scored {
			if s.Warm {
				out = append(out, s)
			}
		}
		return out
	}
	if !preferWarm {
		return scored
	}
	hasWarm := false
	for _, s := range scored {
		if s.Warm {
			hasWarm = true
			break
		}
	}
	if !hasWarm {
		return scored
	}
	out := make([]ScoredNode, 0, len(scored))
	for _, s := range scored {
		if s.Warm {
			out = append(out, s)
		}
	}
	return out
}

func sortScored(scored []ScoredNode) {
	for i := 0; i < len(scored); i++ {
		for j := i + 1; j < len(scored); j++ {
			if scoredLess(scored[j], scored[i]) {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}
}

func scoredLess(a, b ScoredNode) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.ID < b.ID
}
