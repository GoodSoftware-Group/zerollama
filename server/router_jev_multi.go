package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/types/model"
)

// Multi-question jev routing (docs/auto-router.md AR5). Opt-in: a classifier: jev router
// with a non-empty questions list. One /v1/decisions batch answers every question; the
// answers become a fit score and an escalation gate over the router's candidates, and the
// deterministic engine (capability filters, availability, residency) makes the final pick.

// jevMultiDefaultWeights are the plan defaults (§7). Fit dominates; residency and eviction
// still matter but no longer swamp a confident quality signal as in implicit "auto".
var jevMultiDefaultWeights = struct {
	fit float64
	autoroute.Weights
}{1.0, autoroute.Weights{Warm: 0.6, Evict: 0.8, Size: 0.1, Speed: 0.2}}

const (
	defaultEscalateBelow    = 0.55
	defaultActEscalateBelow = 0.5
	reasoningNoulThreshold  = 0.5
	tierHigh                = "high"
)

// RouterQuestion is one ordered Jev question in router.yaml.
type RouterQuestion struct {
	ID           string          `yaml:"id" json:"id"`
	Type         string          `yaml:"type" json:"type"` // choice | score | noul
	Instructions string          `yaml:"instructions" json:"instructions"`
	Criteria     orderedCriteria `yaml:"criteria" json:"criteria,omitempty"` // choice: label → description
	Labels       []string        `yaml:"labels" json:"labels,omitempty"`     // score: ordered levels
}

// RouterWeights overrides the multi-question score weights (unset fields keep defaults).
type RouterWeights struct {
	Fit   *float64 `yaml:"fit" json:"fit,omitempty"`
	Warm  *float64 `yaml:"warm" json:"warm,omitempty"`
	Evict *float64 `yaml:"evict" json:"evict,omitempty"`
	Size  *float64 `yaml:"size" json:"size,omitempty"`
	Speed *float64 `yaml:"speed" json:"speed,omitempty"`
}

type criterion struct {
	Label, Description string
}

// orderedCriteria keeps a YAML mapping's key order. WHY: choice criteria order is
// load-bearing for Laya (logit ↔ label alignment); decoding into a Go map would scramble it.
type orderedCriteria []criterion

func (o *orderedCriteria) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.MappingNode {
		return fmt.Errorf("criteria must be a mapping of label: description (line %d)", n.Line)
	}
	out := make(orderedCriteria, 0, len(n.Content)/2)
	for i := 0; i+1 < len(n.Content); i += 2 {
		out = append(out, criterion{Label: n.Content[i].Value, Description: n.Content[i+1].Value})
	}
	*o = out
	return nil
}

func (o orderedCriteria) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, c := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		k, _ := json.Marshal(c.Label)
		v, _ := json.Marshal(c.Description)
		b.Write(k)
		b.WriteByte(':')
		b.Write(v)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func (o orderedCriteria) labels() []string {
	out := make([]string, len(o))
	for i, c := range o {
		out[i] = c.Label
	}
	return out
}

// validateJevQuestions catches config mistakes before any decider call. These are loud
// (503 with the reason) rather than fail-open: a broken router.yaml should be fixed.
func validateJevQuestions(qs []RouterQuestion) error {
	seen := map[string]bool{}
	for i, q := range qs {
		id := strings.TrimSpace(q.ID)
		if id == "" {
			return fmt.Errorf("questions[%d]: id is required", i)
		}
		if seen[id] {
			return fmt.Errorf("questions[%d]: duplicate id %q", i, id)
		}
		seen[id] = true
		if strings.TrimSpace(q.Instructions) == "" {
			return fmt.Errorf("question %q: instructions required", id)
		}
		switch strings.ToLower(q.Type) {
		case "choice":
			if len(q.Criteria) == 0 {
				return fmt.Errorf("question %q: choice needs criteria", id)
			}
		case "score":
			if len(q.Labels) == 0 {
				return fmt.Errorf("question %q: score needs labels (ordered levels)", id)
			}
		case "noul":
		default:
			return fmt.Errorf("question %q: type must be choice|score|noul", id)
		}
	}
	return nil
}

// buildJevMultiBody writes one /v1/decisions batch for every question, by hand so choice
// criteria keep their configured order (see buildJevRouteBody).
func buildJevMultiBody(decider string, qs []RouterQuestion, user string) ([]byte, error) {
	var b bytes.Buffer
	enc := func(v any) {
		raw, _ := json.Marshal(v)
		b.Write(raw)
	}
	b.WriteString(`{"model":`)
	enc(decider)
	b.WriteString(`,"keep_alive":-1,"state":{"request":`)
	enc(trimHeadTail(user, jevStateMaxBytes))
	b.WriteString(`},"questions":{`)
	for i, q := range qs {
		if i > 0 {
			b.WriteByte(',')
		}
		enc(q.ID)
		b.WriteString(`:{"type":`)
		enc(strings.ToLower(q.Type))
		b.WriteString(`,"instructions":`)
		enc(q.Instructions)
		switch strings.ToLower(q.Type) {
		case "choice":
			b.WriteString(`,"criteria":`)
			raw, err := q.Criteria.MarshalJSON()
			if err != nil {
				return nil, err
			}
			b.Write(raw)
		case "score":
			b.WriteString(`,"criteria":`)
			enc(q.Labels)
		}
		b.WriteByte('}')
	}
	b.WriteString(`}}`)
	return b.Bytes(), nil
}

// jevAnswer is the union of choice / score / noul answers as the router reads them.
type jevAnswer struct {
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Score         *float64           `json:"score"`
	Noul          *float64           `json:"noul"`
	Calibrated    *bool              `json:"calibrated"`
	Action        *struct {
		ActProbability *float64 `json:"act_probability"`
	} `json:"action"`
}

func (a jevAnswer) calibrated() bool { return a.Calibrated == nil || *a.Calibrated }

// act returns Laya's act-head probability when the decider emitted one.
func (a jevAnswer) act() (float64, bool) {
	if a.Action == nil || a.Action.ActProbability == nil {
		return 0, false
	}
	return *a.Action.ActProbability, true
}

func parseJevAnswers(qs []RouterQuestion, raw map[string]json.RawMessage) (map[string]jevAnswer, error) {
	out := make(map[string]jevAnswer, len(qs))
	for _, q := range qs {
		r, ok := raw[q.ID]
		if !ok || len(r) == 0 {
			return nil, fmt.Errorf("decider returned no answer for %q", q.ID)
		}
		var a jevAnswer
		if err := json.Unmarshal(r, &a); err != nil {
			return nil, fmt.Errorf("answer %q: %w", q.ID, err)
		}
		out[q.ID] = a
	}
	return out, nil
}

// jevPlan is what the answers mean for routing: per-candidate fit, whether to escalate,
// and a summary for the decision body.
type jevPlan struct {
	Calibrated bool
	Escalated  bool
	Reasons    []string // why escalated
	Signals    map[string]any
	Difficulty float64 // 1-based level; 0 = no score question
	Reasoning  float64 // P(true) of the first noul question; -1 = none
	answers    map[string]jevAnswer
	questions  []RouterQuestion
}

// planJevRouting turns answers into a plan. Pure — tested without a decider.
func planJevRouting(spec RouterSpec, answers map[string]jevAnswer) jevPlan {
	p := jevPlan{Calibrated: true, Signals: map[string]any{}, Reasoning: -1, answers: answers, questions: spec.Questions}
	var firstChoiceTop float64 = -1
	var minAct float64 = -1
	for _, q := range spec.Questions {
		a := answers[q.ID]
		if !a.calibrated() {
			p.Calibrated = false
		}
		if act, ok := a.act(); ok && (minAct < 0 || act < minAct) {
			minAct = act
		}
		switch strings.ToLower(q.Type) {
		case "choice":
			p.Signals[q.ID] = map[string]any{"answer": a.Choice, "probabilities": a.Probabilities}
			if firstChoiceTop < 0 {
				firstChoiceTop = a.Probabilities[a.Choice]
			}
		case "score":
			if a.Score != nil {
				level := *a.Score + 1 // Score is the expected 0-based level index
				p.Signals[q.ID] = map[string]any{"level": round2(level), "of": len(q.Labels)}
				if p.Difficulty == 0 {
					p.Difficulty = level
				}
			}
		case "noul":
			if a.Noul != nil {
				p.Signals[q.ID] = map[string]any{"noul": round2(*a.Noul)}
				if p.Reasoning < 0 {
					p.Reasoning = *a.Noul
				}
			}
		}
	}
	p.Signals["calibrated"] = p.Calibrated

	// WHY calibrated only: thresholds on uncalibrated probabilities (OpenJev default,
	// GLiNER-Decide) are meaningless, so they never escalate.
	if !p.Calibrated {
		return p
	}
	below := spec.EscalateBelow
	if below <= 0 {
		below = defaultEscalateBelow
	}
	actBelow := spec.ActEscalateBelow
	if actBelow <= 0 {
		actBelow = defaultActEscalateBelow
	}
	if firstChoiceTop >= 0 && firstChoiceTop < below {
		p.Reasons = append(p.Reasons, fmt.Sprintf("unsure (top choice %.2f < %.2f)", firstChoiceTop, below))
	}
	if spec.EscalateAt > 0 && p.Difficulty >= spec.EscalateAt {
		p.Reasons = append(p.Reasons, fmt.Sprintf("difficulty %.1f ≥ %.1f", p.Difficulty, spec.EscalateAt))
	}
	if p.Reasoning >= reasoningNoulThreshold {
		p.Reasons = append(p.Reasons, "needs reasoning")
	}
	if minAct >= 0 && minAct < actBelow {
		p.Reasons = append(p.Reasons, fmt.Sprintf("act head %.2f < %.2f", minAct, actBelow))
	}
	p.Escalated = len(p.Reasons) > 0
	return p
}

// fit scores how well a router candidate matches the answers (before the fit weight).
func (p jevPlan) fit(rc RouterCandidate, caps []model.Capability) (float64, []string) {
	var f float64
	var why []string
	for _, q := range p.questions {
		if strings.ToLower(q.Type) != "choice" {
			continue
		}
		want := rc.Strengths[q.ID]
		if len(want) == 0 {
			continue
		}
		a := p.answers[q.ID]
		var v float64
		if p.Calibrated {
			for _, l := range want {
				v += a.Probabilities[l]
			}
		} else if slices.Contains(want, a.Choice) {
			v = 1 // uncalibrated ⇒ argmax only
		}
		if v > 0 {
			f += v
			why = append(why, fmt.Sprintf("%s fit %.2f", q.ID, v))
		}
	}
	if p.Difficulty > 0 {
		if rc.MinDifficulty > 0 && p.Difficulty < rc.MinDifficulty {
			f -= 1
			why = append(why, fmt.Sprintf("difficulty %.1f below band", p.Difficulty))
		}
		if rc.MaxDifficulty > 0 && p.Difficulty > rc.MaxDifficulty {
			f -= 1
			why = append(why, fmt.Sprintf("difficulty %.1f above band", p.Difficulty))
		}
	}
	if p.Reasoning >= reasoningNoulThreshold && slices.Contains(caps, model.CapabilityThinking) {
		f += 0.5
		why = append(why, "reasoning ⇒ thinking")
	}
	return f, why
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }

func multiWeights(rw *RouterWeights) (float64, autoroute.Weights) {
	fit, w := jevMultiDefaultWeights.fit, jevMultiDefaultWeights.Weights
	if rw == nil {
		return fit, w
	}
	set := func(dst *float64, v *float64) {
		if v != nil {
			*dst = *v
		}
	}
	set(&fit, rw.Fit)
	set(&w.Warm, rw.Warm)
	set(&w.Evict, rw.Evict)
	set(&w.Size, rw.Size)
	set(&w.Speed, rw.Speed)
	return fit, w
}

// decideRouterJevMulti asks every question in one batch, then lets the deterministic
// engine pick among the router's candidates with fit as a bonus and escalation as a gate.
// Decider failures return errDeciderUnavailable so resolveRouter can fail open.
func (s *Server) decideRouterJevMulti(ctx context.Context, name string, spec RouterSpec, sig autoroute.Signals) (*RouterDecision, error) {
	start := time.Now()
	if err := validateJevQuestions(spec.Questions); err != nil {
		return nil, fmt.Errorf("router %q: %w", name, err)
	}
	decider := strings.TrimSpace(spec.Decider)
	if decider == "" {
		return &RouterDecision{Router: name, Classifier: "jev"}, fmt.Errorf("%w: router has no decider", errDeciderUnavailable)
	}
	body, err := buildJevMultiBody(decider, spec.Questions, sig.Text)
	if err != nil {
		return nil, err
	}
	tctx, cancel := context.WithTimeout(ctx, deciderTimeout(spec))
	raw, err := s.askJevAnswers(tctx, decider, body)
	cancel()
	if err != nil {
		return &RouterDecision{Router: name, Classifier: "jev", Decider: decider}, fmt.Errorf("%w: %v", errDeciderUnavailable, err)
	}
	answers, err := parseJevAnswers(spec.Questions, raw)
	if err != nil {
		return &RouterDecision{Router: name, Classifier: "jev", Decider: decider}, fmt.Errorf("%w: %v", errDeciderUnavailable, err)
	}
	plan := planJevRouting(spec, answers)

	byModel := map[string]RouterCandidate{}
	allow := map[string]bool{}
	hasHigh := false
	for _, rc := range spec.Candidates {
		key := model.ParseName(rc.Model).DisplayShortest()
		byModel[key] = rc
		allow[key] = true
		hasHigh = hasHigh || strings.EqualFold(rc.Tier, tierHigh)
	}
	if fb := strings.TrimSpace(spec.Fallback); fb != "" {
		allow[model.ParseName(fb).DisplayShortest()] = true
	}

	fitW, w := multiWeights(spec.Weights)
	sig.Weights = &w
	sig.Bonus = func(c autoroute.Candidate) (float64, string) {
		f, why := plan.fit(byModel[c.Name], c.Capabilities)
		return fitW * f, strings.Join(why, "; ")
	}
	escalated := plan.Escalated && hasHigh
	if escalated {
		sig.Check = func(c autoroute.Candidate) string {
			if !strings.EqualFold(byModel[c.Name].Tier, tierHigh) {
				return "escalated: needs tier high (" + strings.Join(plan.Reasons, "; ") + ")"
			}
			return ""
		}
	}

	dec, err := s.decideAuto(ctx, name, sig, allow)
	escalationUnavailable := false
	if escalated && errors.Is(err, autoroute.ErrNoCandidate) {
		// WHY retry: escalation is a preference for quality, not a reason to refuse. If no
		// tier: high model can serve right now, answer with the best available instead.
		sig.Check = nil
		dec, err = s.decideAuto(ctx, name, sig, allow)
		escalated, escalationUnavailable = false, true
	}
	if dec == nil {
		return nil, err
	}
	dec.Classifier = "jev"
	dec.Decider = decider
	// WHY reset: decideAuto times only its own deterministic pick; the decider batch is the
	// expensive part (found on a real node: smart reported 0 ms for ~4.8 s decisions).
	dec.LatencyMs = time.Since(start).Milliseconds()
	dec.Calibrated = &plan.Calibrated
	dec.Escalated = escalated
	dec.Signals = plan.Signals
	if plan.Escalated && len(plan.Reasons) > 0 {
		dec.Signals["escalate_reasons"] = plan.Reasons
	}
	if escalationUnavailable {
		dec.Signals["escalation_unavailable"] = true
	}
	if err == nil && escalated {
		dec.Reason = "escalated"
	}
	return dec, err
}

// askJevAnswers runs a decisions batch in-process and returns the raw answers. Shared with
// the single-question path; warms the decider in the background on a deadline.
func (s *Server) askJevAnswers(ctx context.Context, decider string, body []byte) (map[string]json.RawMessage, error) {
	if isRouterName(decider) {
		return nil, fmt.Errorf("decider %q is itself a router", decider)
	}
	call, err := parseDecisionsCall(body)
	if err != nil {
		return nil, err
	}
	if served, aliased, err := resolveAlias(decider); err != nil {
		return nil, err
	} else if aliased {
		call.Req.Model, call.LLM.Model = served, served
	}
	resp, err := s.runDecisions(ctx, call)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			s.prewarmDecider(decider, body)
		}
		return nil, err
	}
	return resp.Answers, nil
}
