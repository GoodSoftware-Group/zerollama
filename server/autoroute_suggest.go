package server

import (
	"context"
	"strings"
	"sync"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/server/autoroute"
)

// Cross-modal advisory (docs/auto-router.md §9, AR8). When a chat turn really asks for an
// image, speech, video or music, the router suggests the endpoint, a local model that can
// serve it, and a ready request body — via X-Zerollama-Route-Suggest and /v1/route. The
// chat response itself never changes shape (handoff is AR9, opt-in).
//
// Sources, best first: a jev router's output_modality question (asked in the batch it
// already makes, or alone on sticky / cached turns because intent changes per turn), then
// the keyword fallback autoroute.GuessOutputModality.

const (
	defaultModalityQuestion = "output_modality"
	defaultSuggestThreshold = 0.5
	suggestSourceJev        = "jev"
	suggestSourceHeuristic  = "heuristic"
	envSuggestOff           = "off"
	envSuggestJevOnly       = "jev"
)

// ModalitySuggestion tells a client the turn would be better served by another surface.
type ModalitySuggestion struct {
	Modality    string         `json:"modality"`              // image | speech | video | music
	Source      string         `json:"source"`                // jev | heuristic
	Probability float64        `json:"probability,omitempty"` // jev only
	Endpoint    string         `json:"endpoint"`
	Model       string         `json:"model,omitempty"`       // local model the target surface would route to
	Unavailable string         `json:"unavailable,omitempty"` // why no local model can serve it
	Body        map[string]any `json:"body,omitempty"`        // ready request for Endpoint
}

var suggestTargets = map[string]struct {
	endpoint string
	surface  autoroute.Surface
}{
	autoroute.ModalityImage:  {"/v1/images/generations", autoroute.SurfaceImage},
	autoroute.ModalitySpeech: {"/v1/audio/speech", autoroute.SurfaceSpeech},
	autoroute.ModalityVideo:  {"/v1/videos", autoroute.SurfaceVideoGen},
	autoroute.ModalityMusic:  {"/v1/audio/generations", autoroute.SurfaceMusic},
}

// suggestMode reads ZEROLLAMA_ROUTER_SUGGEST: "" / "auto" (jev, else heuristic), "jev"
// (jev only), "off".
func suggestMode() string {
	return strings.ToLower(strings.TrimSpace(envconfig.Var("ZEROLLAMA_ROUTER_SUGGEST")))
}

// attachSuggestion adds dec.Suggestion for chat turns. Never fails the request: any error
// simply means no suggestion.
func (s *Server) attachSuggestion(ctx context.Context, name string, sig autoroute.Signals, dec *RouterDecision, write bool) {
	if dec == nil || sig.Surface != autoroute.SurfaceChat || suggestMode() == envSuggestOff {
		return
	}
	modality, prob, source := s.detectModality(ctx, name, sig, dec)
	if modality == "" || modality == autoroute.ModalityText {
		return
	}
	target, ok := suggestTargets[modality]
	if !ok {
		return
	}
	sug := &ModalitySuggestion{Modality: modality, Source: source, Probability: prob, Endpoint: target.endpoint}
	tsig := autoroute.Signals{Surface: target.surface, Text: sig.Text}
	if d, err := s.decideAuto(ctx, implicitAutoRouter, tsig, nil); err == nil && d != nil {
		sug.Model = d.Candidate
	} else if err != nil {
		sug.Unavailable = err.Error()
	}
	sug.Body = suggestionBody(modality, sug.Model, sig)
	dec.Suggestion = sug
	if write {
		metricsRouteSuggest(modality, source)
	}
}

// detectModality returns the suggested modality with its probability and source.
func (s *Server) detectModality(ctx context.Context, name string, sig autoroute.Signals, dec *RouterDecision) (string, float64, string) {
	spec, isYAML := lookupRouter(name)
	qid := defaultModalityQuestion
	if isYAML && strings.TrimSpace(spec.ModalityQuestion) != "" {
		qid = strings.TrimSpace(spec.ModalityQuestion)
	}
	threshold := spec.SuggestThreshold
	if threshold <= 0 {
		threshold = defaultSuggestThreshold
	}

	// 1. This turn's multi-question batch already answered it.
	if ans, ok := dec.Signals[qid].(map[string]any); ok {
		choice, _ := ans["answer"].(string)
		probs, _ := ans["probabilities"].(map[string]float64)
		calibrated := dec.Calibrated == nil || *dec.Calibrated
		return modalityFromChoice(choice, probs, calibrated, threshold)
	}
	// 2. A jev router with that question, but the decider was skipped (sticky / cached /
	// fail-open): ask just this question — cross-modal intent changes turn to turn.
	// Skipped after a decider failure (decider-fallback): asking again would pay a second
	// timeout on a decider we already know is down.
	if isYAML && routerIsJev(spec) && dec.Reason != "decider-fallback" {
		if q, ok := findQuestion(spec.Questions, qid); ok && strings.TrimSpace(spec.Decider) != "" {
			if body, err := buildJevMultiBody(spec.Decider, []RouterQuestion{q}, sig.Text); err == nil {
				tctx, cancel := context.WithTimeout(ctx, deciderTimeout(spec))
				raw, err := s.askJevAnswers(tctx, spec.Decider, body)
				cancel()
				if err == nil {
					if answers, err := parseJevAnswers([]RouterQuestion{q}, raw); err == nil {
						a := answers[qid]
						return modalityFromChoice(a.Choice, a.Probabilities, a.calibrated(), threshold)
					}
				}
			}
		}
	}
	// 3. Keyword fallback.
	if suggestMode() == envSuggestJevOnly {
		return "", 0, ""
	}
	if m, ok := autoroute.GuessOutputModality(sig.Text); ok {
		return m, 0, suggestSourceHeuristic
	}
	return "", 0, ""
}

// modalityFromChoice applies the threshold for calibrated deciders (P(choice) ≥ threshold)
// and plain argmax otherwise.
func modalityFromChoice(choice string, probs map[string]float64, calibrated bool, threshold float64) (string, float64, string) {
	if choice == "" || choice == autoroute.ModalityText {
		return "", 0, ""
	}
	p := probs[choice]
	if calibrated && p < threshold {
		return "", 0, ""
	}
	return choice, p, suggestSourceJev
}

func findQuestion(qs []RouterQuestion, id string) (RouterQuestion, bool) {
	for _, q := range qs {
		if q.ID == id {
			return q, true
		}
	}
	return RouterQuestion{}, false
}

// suggestionBody builds a request the client can send to the suggested endpoint as-is.
// Speech reads back the previous assistant reply when there is one ("read that out loud").
func suggestionBody(modality, model string, sig autoroute.Signals) map[string]any {
	body := map[string]any{}
	if model != "" {
		body["model"] = model
	}
	switch modality {
	case autoroute.ModalityImage, autoroute.ModalityVideo:
		body["prompt"] = sig.Text
	case autoroute.ModalitySpeech:
		input := sig.Text
		if strings.TrimSpace(sig.LastAssistant) != "" {
			input = sig.LastAssistant
		}
		body["input"] = input
	case autoroute.ModalityMusic:
		body["instructions"] = sig.Text // Music 3 caption; lyrics (input) left to the client
	}
	return body
}

var (
	suggestMetricsMu sync.Mutex
	suggestCounts    = map[[2]string]uint64{} // {modality, source}
)

func metricsRouteSuggest(modality, source string) {
	suggestMetricsMu.Lock()
	suggestCounts[[2]string{modality, source}]++
	suggestMetricsMu.Unlock()
}
