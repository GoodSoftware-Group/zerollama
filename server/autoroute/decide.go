package autoroute

import (
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/ollama/ollama/types/model"
)

// Availability is the server's dry-run view of one candidate (never triggers a load).
type Availability struct {
	Warm          bool   // resident runner for this model
	CanLoad       bool   // a cold load would be admitted now (loads not paused, not cooling down)
	NeedsEviction bool   // loading would unload another resident model (thrash risk)
	Excluded      string // non-empty: not routable on this path, with the reason
	Detail        string // why CanLoad is false / what NeedsEviction would evict
}

// AvailabilityFunc reports availability for c. Called only for candidates that already
// passed capability and context filters, so expensive probes stay off rejected models.
type AvailabilityFunc func(Candidate) Availability

// Rejection records why a candidate was dropped — surfaced in /v1/route and error bodies
// so a 503 is actionable instead of "no model".
type Rejection struct {
	Model  string `json:"model"`
	Reason string `json:"reason"`
}

// Ranked is one surviving candidate with its deterministic score.
type Ranked struct {
	Model         string   `json:"model"`
	Score         float64  `json:"score"`
	Warm          bool     `json:"warm"`
	NeedsEviction bool     `json:"needs_eviction"`
	Reasons       []string `json:"reasons,omitempty"`
}

// Decision is the router's answer.
type Decision struct {
	Surface  Surface     `json:"surface"`
	Model    string      `json:"model"`
	Reason   string      `json:"reason"` // only-candidate | warm | best-fit
	Ranked   []Ranked    `json:"ranked,omitempty"`
	Rejected []Rejection `json:"filtered,omitempty"`
}

// ErrNoCandidate means every candidate was filtered out. The Decision returned with it
// still carries Rejected so callers can explain why.
var ErrNoCandidate = errors.New("no local model can serve this request")

// ErrAmbiguousPinned means a pinned surface (embed / rerank) has more than one qualifying
// model. The router refuses rather than guess, because a later pull could silently change
// the pick and break vector comparability.
var ErrAmbiguousPinned = errors.New("more than one model qualifies for a pinned surface")

// Weights scale the deterministic score components. Fit (from Jev answers) is applied by
// the caller through Signals.Bonus, so it is not part of this struct.
type Weights struct {
	Warm  float64 `json:"warm"`
	Evict float64 `json:"evict"`
	Size  float64 `json:"size"`
	Speed float64 `json:"speed"`
}

// DefaultWeights are the AR1 deterministic weights. WHY these orders of magnitude:
// residency dominates on single-GPU hosts where a load costs seconds to minutes; eviction
// is the next worst outcome; size (a crude quality proxy for text) and bench speed only
// break ties among equally available models.
var DefaultWeights = Weights{Warm: 4.0, Evict: 2.0, Size: 1.0, Speed: 0.5}

// neutralSpeed is used when a candidate has no bench entry, so un-benched models are
// neither preferred nor penalised against benched ones.
const neutralSpeed = 0.5

// Decide filters cands for sig, scores the survivors and picks one. It is pure apart from
// avail and bench.json reads, and safe for concurrent use.
func Decide(cands []Candidate, sig Signals, avail AvailabilityFunc) (Decision, error) {
	dec := Decision{Surface: sig.Surface}
	kept, rejected := filterCandidates(cands, sig)

	// WHY before availability: pinned surfaces (embed / rerank) must not silently switch
	// models when one of several capability-eligible candidates is briefly cooling down.
	// Vector spaces and rerank scores are incompatible across models — uniqueness is a
	// property of the inventory, not of transient load state.
	if sig.Surface.Pinned() && len(kept) > 1 {
		names := make([]string, len(kept))
		for i, c := range kept {
			names[i] = c.Name
		}
		sort.Strings(names)
		dec.Rejected = rejected
		return dec, fmt.Errorf("%w: %s (%s) — pin one with a router.yaml embedder/reranker or name the model",
			ErrAmbiguousPinned, sig.Surface, strings.Join(names, ", "))
	}

	type entry struct {
		c Candidate
		a Availability
	}
	var live []entry
	for _, c := range kept {
		a := Availability{CanLoad: true}
		if avail != nil {
			a = avail(c)
		}
		switch {
		case a.Excluded != "":
			rejected = append(rejected, Rejection{c.Name, a.Excluded})
		case !a.Warm && !a.CanLoad:
			reason := "cannot load now"
			if a.Detail != "" {
				reason += ": " + a.Detail
			}
			rejected = append(rejected, Rejection{c.Name, reason})
		default:
			live = append(live, entry{c, a})
		}
	}
	dec.Rejected = rejected
	if len(live) == 0 {
		return dec, fmt.Errorf("%w (%s)", ErrNoCandidate, summarize(rejected))
	}

	w := DefaultWeights
	if sig.Weights != nil {
		w = *sig.Weights
	}
	var maxSize int64
	for _, e := range live {
		maxSize = max(maxSize, e.c.SizeBytes)
	}
	speeds := relativeSpeeds(live, func(e entry) Candidate { return e.c })
	preferSize := sig.Surface.IsText()

	dec.Ranked = make([]Ranked, len(live))
	for i, e := range live {
		r := Ranked{Model: e.c.Name, Warm: e.a.Warm, NeedsEviction: e.a.NeedsEviction}
		if e.a.Warm {
			r.Score += w.Warm
			r.Reasons = append(r.Reasons, "warm")
		} else if e.a.NeedsEviction {
			r.Score -= w.Evict
			reason := "needs eviction"
			if e.a.Detail != "" {
				reason += ": " + e.a.Detail
			}
			r.Reasons = append(r.Reasons, reason)
		}
		if preferSize && maxSize > 0 {
			r.Score += w.Size * float64(e.c.SizeBytes) / float64(maxSize)
		}
		r.Score += w.Speed * speeds[i]
		if sig.Bonus != nil {
			if b, why := sig.Bonus(e.c); b != 0 {
				r.Score += b
				if why != "" {
					r.Reasons = append(r.Reasons, why)
				}
			}
		}
		dec.Ranked[i] = r
	}
	// Stable on name order (inventory is sorted) so equal scores pick deterministically.
	sort.SliceStable(dec.Ranked, func(i, j int) bool { return dec.Ranked[i].Score > dec.Ranked[j].Score })

	top := dec.Ranked[0]
	dec.Model = top.Model
	switch {
	case len(dec.Ranked) == 1:
		dec.Reason = "only-candidate"
	case top.Warm:
		dec.Reason = "warm"
	default:
		dec.Reason = "best-fit"
	}
	return dec, nil
}

// filterCandidates applies the request-derived hard filters: surface class, required
// capabilities, then context fit.
//
// WHY context fit is soft: when no candidate's window fits (a very long prompt), dropping
// everything would 503 a request the runner could still serve with truncation or LA22
// compression. In that case the filter is skipped rather than applied.
func filterCandidates(cands []Candidate, sig Signals) ([]Candidate, []Rejection) {
	var kept []Candidate
	var rejected []Rejection
	for _, c := range cands {
		if reason, ok := sig.Admits(c); !ok {
			rejected = append(rejected, Rejection{c.Name, reason})
			continue
		}
		kept = append(kept, c)
	}

	kept, rejected = softFilter(kept, rejected, voiceFilter(sig))

	need := sig.contextNeed()
	if need == 0 {
		return kept, rejected
	}
	var fit []Candidate
	var tooSmall []Rejection
	for _, c := range kept {
		if c.ContextLength > 0 && c.ContextLength < need {
			tooSmall = append(tooSmall, Rejection{c.Name, fmt.Sprintf("context %d < needed %d", c.ContextLength, need)})
			continue
		}
		fit = append(fit, c)
	}
	if len(fit) == 0 {
		return kept, rejected
	}
	return fit, append(rejected, tooSmall...)
}

// voiceFilter drops speech candidates that list voices but not the requested one. A
// candidate with no voice list accepts any voice (remote TTS passes it upstream).
func voiceFilter(sig Signals) func(Candidate) string {
	if sig.Surface != SurfaceSpeech || sig.Voice == "" {
		return nil
	}
	return func(c Candidate) string {
		if len(c.Voices) == 0 || slices.Contains(c.Voices, sig.Voice) {
			return ""
		}
		return fmt.Sprintf("voice %q not offered (has %s)", sig.Voice, strings.Join(c.Voices, ", "))
	}
}

// softFilter applies reject but keeps everything when it would reject every candidate.
// WHY: a voice / context mismatch on every model is better answered by the handler (which
// may fall back to a default voice or truncate) than by a router 503.
func softFilter(kept []Candidate, rejected []Rejection, reject func(Candidate) string) ([]Candidate, []Rejection) {
	if reject == nil {
		return kept, rejected
	}
	var pass []Candidate
	var drop []Rejection
	for _, c := range kept {
		if reason := reject(c); reason != "" {
			drop = append(drop, Rejection{c.Name, reason})
			continue
		}
		pass = append(pass, c)
	}
	if len(pass) == 0 {
		return kept, rejected
	}
	return pass, append(rejected, drop...)
}

func missingCaps(c Candidate, required []model.Capability) []string {
	var out []string
	for _, r := range required {
		if !c.Has(r) {
			out = append(out, string(r))
		}
	}
	return out
}

// relativeSpeeds maps each candidate to [0,1] by bench perf relative to the best:
// tok/s for completion entries, inverse seconds for image/video entries.
func relativeSpeeds[T any](items []T, cand func(T) Candidate) []float64 {
	raw := make([]float64, len(items))
	var best float64
	for i, it := range items {
		e, ok := cand(it).Perf()
		if !ok {
			raw[i] = -1
			continue
		}
		switch {
		case e.TokPerSec > 0:
			raw[i] = e.TokPerSec
		case e.GenSec > 0:
			raw[i] = 1 / e.GenSec
		}
		best = max(best, raw[i])
	}
	out := make([]float64, len(items))
	for i, v := range raw {
		if v < 0 || best <= 0 {
			out[i] = neutralSpeed
			continue
		}
		out[i] = v / best
	}
	return out
}

func summarize(rs []Rejection) string {
	if len(rs) == 0 {
		return "no local models"
	}
	const maxShown = 4
	parts := make([]string, 0, maxShown+1)
	for i, r := range rs {
		if i == maxShown {
			parts = append(parts, fmt.Sprintf("+%d more", len(rs)-maxShown))
			break
		}
		parts = append(parts, r.Model+": "+r.Reason)
	}
	return strings.Join(parts, "; ")
}
