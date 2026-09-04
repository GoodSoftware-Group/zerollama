package mlxrunner

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
)

type specStats struct {
	iterations int
	drafted    int
	accepted   int
	maxDraft   int
	// chosen is the draft depth picked each round, in order; split into time
	// buckets it distinguishes a ramp that holds from one that thrashes shallow.
	chosen []int
}

func (s *specStats) recordRound(depth int) {
	if !slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		return
	}
	s.chosen = append(s.chosen, depth)
}

// depthBuckets is how many equal time slices depthOverTime splits a run into.
const depthBuckets = 8

// depthOverTime reports per-bucket mean/max chosen depth across up to depthBuckets
// equal time buckets, e.g. "0.3/1 2.1/3 4.8/5 5.0/6".
func (s *specStats) depthOverTime() string {
	if len(s.chosen) == 0 {
		return ""
	}
	buckets := min(depthBuckets, len(s.chosen))
	parts := make([]string, 0, buckets)
	for b := range buckets {
		lo := b * len(s.chosen) / buckets
		hi := (b + 1) * len(s.chosen) / buckets
		sum, mx := 0, 0
		for _, d := range s.chosen[lo:hi] {
			sum += d
			mx = max(mx, d)
		}
		parts = append(parts, fmt.Sprintf("%.1f/%d", float64(sum)/float64(hi-lo), mx))
	}
	return strings.Join(parts, " ")
}

func (s *speculationSession) logStats() {
	acceptance := 0.0
	if s.stats.drafted > 0 {
		acceptance = float64(s.stats.accepted) / float64(s.stats.drafted)
	}
	s.logTuneHint(acceptance)
	s.saveLastRun(acceptance)
	if !slog.Default().Enabled(context.TODO(), slog.LevelDebug) {
		return
	}
	avgDraft := 0.0
	avgAccepted := 0.0
	if s.stats.iterations > 0 {
		avgDraft = float64(s.stats.drafted) / float64(s.stats.iterations)
		avgAccepted = float64(s.stats.accepted) / float64(s.stats.iterations)
	}
	slog.Debug("speculative decode stats", "iterations", s.stats.iterations, "drafted", s.stats.drafted, "accepted", s.stats.accepted, "acceptance", fmt.Sprintf("%.2f", acceptance), "avg_draft", fmt.Sprintf("%.2f", avgDraft), "max_draft", s.stats.maxDraft, "avg_accepted", fmt.Sprintf("%.2f", avgAccepted), "greedy_coupled", s.greedyCoupled(), "depth_over_time", s.stats.depthOverTime())

	if s.spec == nil || s.spec.depth == nil {
		return
	}
	d := s.spec.depth
	frontier := d.frontier()
	rates := make([]string, 0, frontier)
	for n := 1; n <= frontier; n++ {
		rates = append(rates, fmt.Sprintf("%d:%.2f", n, d.acc.acceptance(n)))
	}
	limit := frontier + 1
	tps := make([]string, 0, limit+1)
	if d.cost.ready() {
		for n := 0; n <= limit; n++ {
			tps = append(tps, fmt.Sprintf("%d:%.1f", n, 1000*d.acc.expectedCommitted(n)/d.cost.cost(n)))
		}
	}
	slog.Debug("speculation depth controller", "cost", d.cost.sampleString(), "acceptance", strings.Join(rates, " "), "expected_tps", strings.Join(tps, " "), "probe_interval", d.probeInterval)
}

func (s *speculationSession) logTuneHint(acceptance float64) {
	hint := s.tuneHint(acceptance)
	if hint == "" {
		return
	}
	slog.Info("mlx tune", "hint", hint, "acceptance", fmt.Sprintf("%.2f", acceptance), "max_draft", s.stats.maxDraft, "iterations", s.stats.iterations)
}

func (s *speculationSession) tuneHint(acceptance float64) string {
	if s == nil || s.stats.iterations < pldRuntimeMinRounds {
		return ""
	}
	if !s.enabled {
		if s.pld {
			return "spec parked (runtime gate). Novel text is expected AR. ZEROLLAMA_MLX_PLD=off only for a bench; leave on for agents"
		}
		hint := "MTP parked (accept <0.70). Try ZEROLLAMA_MLX_MTP_HISTORY=auto on long prompts, or check the draft companion loaded"
		if fence := greedyTrioMaxContext(); fence > 0 && s.promptTokens >= fence {
			hint += "; T=0 greedy trio is fenced at this prompt length (ZEROLLAMA_MLX_GREEDY_TRIO_MAX_CONTEXT)"
		}
		return hint
	}
	if s.pld && acceptance < pldRuntimeMinAccept {
		return "PLD accept is low this request; runtime gate will freeze PLD. Echo/code prompts should be higher; prose stays AR"
	}
	if s.spec != nil && s.spec.depth != nil && s.spec.depth.scheduled == 0 && s.stats.maxDraft == 0 {
		return "draft width 0. After a few echo requests, mlx-round-cost should show scheduled>0 (`zerollama doctor`)"
	}
	return ""
}

// greedyCoupled is mtplx telemetry: T=0 and the greedy trio fence are both
// live, so draft argmax and equality-accept match the target.
func (s *speculationSession) greedyCoupled() bool {
	if s == nil || s.spec == nil || s.spec.r == nil || s.spec.r.Sampler == nil {
		return false
	}
	if !s.spec.r.Sampler.Greedy(pipelineSlot) {
		return false
	}
	return greedyDraftChainEnabled(s.promptTokens) && batchedGreedyAcceptEnabled(s.promptTokens)
}
