package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// classifier: jev (docs/auto-router.md AR4): a local /v1/decisions model (Laya, CLM, a
// System One score model, OpenJev, GLiNER-Decide) answers one choice question whose
// options are the router's policy labels. Same finish() as the score tier, so candidates
// and fallback behave identically.

const (
	defaultDeciderTimeout = 150 * time.Millisecond
	jevRouteQuestionID    = "route"
	// jevStateMaxBytes bounds the request text sent as state. WHY: Laya is a ModernBERT-class
	// encoder; over-long state truncates silently inside the runner, so keep head and tail.
	jevStateMaxBytes = 8000
)

// errDeciderUnavailable marks a jev decider failure (timeout, load error, bad answer) that
// the caller may fail open from. Wrapped with the cause.
var errDeciderUnavailable = errors.New("jev decider unavailable")

func routerIsJev(spec RouterSpec) bool {
	return strings.EqualFold(strings.TrimSpace(spec.Classifier), "jev")
}

func deciderTimeout(spec RouterSpec) time.Duration {
	if d, err := time.ParseDuration(strings.TrimSpace(spec.DeciderTimeout)); err == nil && d > 0 {
		return d
	}
	return defaultDeciderTimeout
}

// deciderFailsClosed reports on_decider_error: fail — surface decider errors as 503 instead
// of falling back.
func deciderFailsClosed(spec RouterSpec) bool {
	return strings.EqualFold(strings.TrimSpace(spec.OnDeciderError), "fail")
}

func decideRouterJev(ctx context.Context, spec RouterSpec, user string, jev routerJevFn, dec *RouterDecision,
	fallback func(string) (RouterDecision, error), finish func([]string) (RouterDecision, error),
) (RouterDecision, error) {
	decider := strings.TrimSpace(spec.Decider)
	dec.Decider = decider
	labels := policyLabels(spec)
	if len(labels) == 0 {
		return fallback("no policies")
	}

	fail := func(cause error) (RouterDecision, error) {
		err := fmt.Errorf("%w: %v", errDeciderUnavailable, cause)
		if deciderFailsClosed(spec) || spec.Fallback == "" {
			// No fallback configured: resolveRouter may still pick deterministically among
			// the router's candidates unless on_decider_error is fail.
			return *dec, err
		}
		return fallback(err.Error())
	}
	if decider == "" || jev == nil {
		return fail(errors.New("router has no decider"))
	}

	body, err := buildJevRouteBody(decider, spec, user)
	if err != nil {
		return fail(err)
	}
	tctx, cancel := context.WithTimeout(ctx, deciderTimeout(spec))
	defer cancel()
	choice, probs, calibrated, err := jev(tctx, decider, body)
	if err != nil {
		return fail(err)
	}
	dec.Calibrated = &calibrated

	thresh := spec.ActivationThreshold
	if thresh <= 0 {
		thresh = defaultRouterActivation
	}
	var active []string
	dec.Scores = make([]RouterLabelScore, len(labels))
	for i, lab := range labels {
		p := probs[lab]
		// WHY argmax only when uncalibrated: OpenJev (default) / GLiNER-Decide probabilities
		// are not calibrated, so a fixed activation threshold would be meaningless.
		on := lab == choice
		if calibrated {
			on = p >= thresh
		}
		lp := math.Inf(-1)
		if p > 0 {
			lp = math.Log(p)
		}
		dec.Scores[i] = RouterLabelScore{Label: lab, LogProb: lp, Softmax: p, Active: on}
		if on {
			active = append(active, lab)
		}
	}
	if len(active) == 0 && choice != "" {
		active = []string{choice}
	}
	if len(active) == 0 {
		return fail(errors.New("decider returned no choice"))
	}
	return finish(active)
}

func policyLabels(spec RouterSpec) []string {
	labels := make([]string, 0, len(spec.Policies))
	for _, p := range spec.Policies {
		if p.Label != "" {
			labels = append(labels, p.Label)
		}
	}
	return labels
}

// buildJevRouteBody writes the /v1/decisions request by hand.
// WHY not json.Marshal of a map: choice criteria key order is load-bearing for Laya (logit
// ↔ label alignment) and must follow the router's policy order; Go maps do not keep it.
// keep_alive -1 keeps the decider resident once loaded (it serves every routed request).
func buildJevRouteBody(decider string, spec RouterSpec, user string) ([]byte, error) {
	var b bytes.Buffer
	enc := func(v any) error {
		raw, err := json.Marshal(v)
		if err != nil {
			return err
		}
		b.Write(raw)
		return nil
	}
	b.WriteString(`{"model":`)
	if err := enc(decider); err != nil {
		return nil, err
	}
	b.WriteString(`,"keep_alive":-1,"state":{"request":`)
	if err := enc(trimHeadTail(user, jevStateMaxBytes)); err != nil {
		return nil, err
	}
	b.WriteString(`},"questions":{"` + jevRouteQuestionID + `":{"type":"choice","instructions":`)
	if err := enc("Which kind of request is this? Pick the label that best matches the user's intent."); err != nil {
		return nil, err
	}
	b.WriteString(`,"criteria":{`)
	n := 0
	for _, p := range spec.Policies {
		if p.Label == "" {
			continue
		}
		if n > 0 {
			b.WriteByte(',')
		}
		n++
		desc := p.Description
		if desc == "" {
			desc = p.Label
		}
		if err := enc(p.Label); err != nil {
			return nil, err
		}
		b.WriteByte(':')
		if err := enc(desc); err != nil {
			return nil, err
		}
	}
	b.WriteString(`}}}}`)
	return b.Bytes(), nil
}

// trimHeadTail keeps the first and last halves of s when it exceeds max bytes, on rune
// boundaries. The tail usually carries the actual ask; the head carries the setup.
func trimHeadTail(s string, max int) string {
	if len(s) <= max {
		return s
	}
	half := max / 2
	cut := half
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	start := len(s) - half
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[:cut] + "\n…\n" + s[start:]
}

// jevChoice is the subset of a choice answer the router needs. Laya / CLM / System One
// omit "calibrated" (they are calibrated by construction); OpenJev and GLiNER-Decide set it.
type jevChoice struct {
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Calibrated    *bool              `json:"calibrated"`
}

func parseJevChoice(raw json.RawMessage) (string, map[string]float64, bool, error) {
	if len(raw) == 0 {
		return "", nil, false, errors.New("decider returned no answer for the route question")
	}
	var ans jevChoice
	if err := json.Unmarshal(raw, &ans); err != nil {
		return "", nil, false, fmt.Errorf("decider answer: %w", err)
	}
	calibrated := ans.Calibrated == nil || *ans.Calibrated
	return ans.Choice, ans.Probabilities, calibrated, nil
}

// askJevDecider runs one routing question through runDecisions in-process (no HTTP hop).
func (s *Server) askJevDecider(ctx context.Context, decider string, body []byte) (string, map[string]float64, bool, error) {
	answers, err := s.askJevAnswers(ctx, decider, body)
	if err != nil {
		return "", nil, false, err
	}
	return parseJevChoice(answers[jevRouteQuestionID])
}

var deciderWarming sync.Map // decider name → struct{} while a warm-up runs

// prewarmDecider issues one untimed decisions call in the background (singleflight per
// decider). Going through runDecisions warms whichever backend serves the decider —
// a Laya runner, CLM heads, or an external URL — and keep_alive -1 holds it.
func (s *Server) prewarmDecider(decider string, body []byte) {
	if _, busy := deciderWarming.LoadOrStore(decider, struct{}{}); busy {
		return
	}
	go func() {
		defer deciderWarming.Delete(decider)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		call, err := parseDecisionsCall(body)
		if err != nil {
			return
		}
		if served, aliased, err := resolveAlias(decider); err == nil && aliased {
			call.Req.Model, call.LLM.Model = served, served
		}
		if _, err := s.runDecisions(ctx, call); err != nil {
			slog.Warn("jev router: decider warm-up failed", "decider", decider, "error", err)
			return
		}
		slog.Info("jev router: decider warm", "decider", decider)
	}()
}

// prewarmJevDeciders warms every classifier: jev decider at serve start.
func (s *Server) prewarmJevDeciders() {
	file, err := loadRouterFile()
	if err != nil || file == nil {
		return
	}
	for name, spec := range file.Routers {
		decider := strings.TrimSpace(spec.Decider)
		if !routerIsJev(spec) || decider == "" {
			continue
		}
		var (
			body []byte
			err  error
		)
		// Multi-question routers (classifier: jev + Questions, e.g. smart) have no
		// Policies — buildJevRouteBody would emit empty choice criteria and
		// parseDecisionsCall would reject before the decider loads.
		if len(spec.Questions) > 0 {
			body, err = buildJevMultiBody(decider, spec.Questions, "warm-up")
		} else {
			body, err = buildJevRouteBody(decider, spec, "warm-up")
		}
		if err != nil {
			continue
		}
		slog.Info("jev router: warming decider", "router", name, "decider", decider)
		s.prewarmDecider(decider, body)
	}
}
