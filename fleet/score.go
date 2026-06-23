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
	scoreQueueWeight     = 100
	scoreLoadingPenalty  = 500
	scoreColdPenalty     = 2_000 // applied only when prefer_warm and no warm nodes remain after filter
	scoreResidentPenalty = 300   // per other loaded model on the node
	scoreLoadedCtxPerK   = 5     // per 1024 effective ctx from non-request residents
)

// ScoreRequest is POST /internal/score (and the scoring input for assign).
type ScoreRequest struct {
	Model          string   `json:"model"`
	PreferWarm     *bool    `json:"prefer_warm,omitempty"`
	WarmOnly       bool     `json:"warm_only,omitempty"`
	Exclude        []string `json:"exclude,omitempty"`
	SessionKey     string   `json:"session_key,omitempty"`
	PromptCacheKey string   `json:"prompt_cache_key,omitempty"`
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
		score, reasons := nodeScore(n, model, warm, affinity, preferWarm)
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

func nodeScore(n NodeSnapshot, assignModel string, warm, affinity, preferWarm bool) (float64, []string) {
	var score float64
	reasons := make([]string, 0, 6)

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
