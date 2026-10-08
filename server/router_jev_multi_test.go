package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/ollama/ollama/server/autoroute"
	"github.com/ollama/ollama/types/model"
)

func f64(v float64) *float64 { return &v }

func multiSpec() RouterSpec {
	var spec RouterSpec
	raw := `
classifier: jev
decider: clm
escalate_at: 4
questions:
  - id: task
    type: choice
    instructions: What kind of work?
    criteria:
      code: write or fix code
      chat: conversation
  - id: difficulty
    type: score
    instructions: How hard?
    labels: ["1", "2", "3", "4", "5"]
  - id: reasoning
    type: noul
    instructions: Needs multi-step reasoning?
candidates:
  - model: plain
    tier: low
    strengths: {task: [chat]}
    max_difficulty: 2
  - model: tooly
    tier: high
    strengths: {task: [code]}
`
	if err := yaml.Unmarshal([]byte(raw), &spec); err != nil {
		panic(err)
	}
	return spec
}

func TestOrderedCriteriaKeepsYAMLOrder(t *testing.T) {
	spec := multiSpec()
	if got := spec.Questions[0].Criteria.labels(); strings.Join(got, ",") != "code,chat" {
		t.Fatalf("labels = %v", got)
	}
	var q RouterQuestion
	if err := yaml.Unmarshal([]byte("id: x\ncriteria: {zeta: z, alpha: a}\n"), &q); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(q.Criteria)
	if string(b) != `{"zeta":"z","alpha":"a"}` {
		t.Fatalf("json = %s", b)
	}
	if err := yaml.Unmarshal([]byte("criteria: [a, b]\n"), &q); err == nil {
		t.Fatal("list criteria must be rejected")
	}
}

func TestValidateJevQuestions(t *testing.T) {
	if err := validateJevQuestions(multiSpec().Questions); err != nil {
		t.Fatal(err)
	}
	for name, qs := range map[string][]RouterQuestion{
		"missing id":  {{Type: "noul", Instructions: "x"}},
		"duplicate":   {{ID: "a", Type: "noul", Instructions: "x"}, {ID: "a", Type: "noul", Instructions: "y"}},
		"no criteria": {{ID: "a", Type: "choice", Instructions: "x"}},
		"no levels":   {{ID: "a", Type: "score", Instructions: "x"}},
		"bad type":    {{ID: "a", Type: "rank", Instructions: "x"}},
	} {
		if validateJevQuestions(qs) == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestBuildJevMultiBodyIsValidBatch(t *testing.T) {
	body, err := buildJevMultiBody("clm", multiSpec().Questions, "fix my loop")
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)
	if strings.Index(s, `"code"`) > strings.Index(s, `"chat"`) || !strings.Contains(s, `"criteria":["1","2","3","4","5"]`) {
		t.Fatalf("body = %s", s)
	}
	call, err := parseDecisionsCall(body)
	if err != nil {
		t.Fatalf("not a valid decisions batch: %v", err)
	}
	if len(call.Req.Questions) != 3 || call.Req.Questions["reasoning"].Type != "noul" {
		t.Fatalf("questions = %+v", call.Req.Questions)
	}
}

func answersOf(task string, probs map[string]float64, level0 float64, noul float64, calibrated *bool, act *float64) map[string]jevAnswer {
	a := map[string]jevAnswer{
		"task":       {Choice: task, Probabilities: probs, Calibrated: calibrated},
		"difficulty": {Score: f64(level0), Calibrated: calibrated},
		"reasoning":  {Noul: f64(noul), Calibrated: calibrated},
	}
	if act != nil {
		t := a["task"]
		t.Action = &struct {
			ActProbability *float64 `json:"act_probability"`
		}{act}
		a["task"] = t
	}
	return a
}

func TestPlanJevRoutingEscalation(t *testing.T) {
	spec := multiSpec()

	p := planJevRouting(spec, answersOf("chat", map[string]float64{"chat": 0.9, "code": 0.1}, 0, 0.1, nil, nil))
	if p.Escalated || p.Difficulty != 1 || !p.Calibrated {
		t.Fatalf("easy chat: %+v", p)
	}

	p = planJevRouting(spec, answersOf("code", map[string]float64{"code": 0.5, "chat": 0.5}, 0, 0.1, nil, nil))
	if !p.Escalated || !strings.Contains(strings.Join(p.Reasons, ";"), "unsure") {
		t.Fatalf("unsure: %+v", p.Reasons)
	}

	p = planJevRouting(spec, answersOf("code", map[string]float64{"code": 0.9}, 3, 0.1, nil, nil)) // level 4 ≥ escalate_at 4
	if !p.Escalated || !strings.Contains(strings.Join(p.Reasons, ";"), "difficulty 4.0") {
		t.Fatalf("difficulty: %+v", p.Reasons)
	}

	p = planJevRouting(spec, answersOf("chat", map[string]float64{"chat": 0.9}, 0, 0.8, nil, nil))
	if !p.Escalated || !strings.Contains(strings.Join(p.Reasons, ";"), "reasoning") {
		t.Fatalf("reasoning: %+v", p.Reasons)
	}

	p = planJevRouting(spec, answersOf("chat", map[string]float64{"chat": 0.9}, 0, 0.1, nil, f64(0.2)))
	if !p.Escalated || !strings.Contains(strings.Join(p.Reasons, ";"), "act head") {
		t.Fatalf("act head: %+v", p.Reasons)
	}

	no := false
	p = planJevRouting(spec, answersOf("code", map[string]float64{"code": 0.3}, 4, 0.9, &no, f64(0.1)))
	if p.Escalated || p.Calibrated {
		t.Fatalf("uncalibrated must never escalate: %+v", p)
	}
}

func TestJevPlanFit(t *testing.T) {
	spec := multiSpec()
	plain, tooly := spec.Candidates[0], spec.Candidates[1]
	thinking := []model.Capability{model.CapabilityCompletion, model.CapabilityThinking}

	p := planJevRouting(spec, answersOf("code", map[string]float64{"code": 0.8, "chat": 0.2}, 2, 0.9, nil, nil)) // level 3
	if f, _ := p.fit(tooly, nil); f != 0.8 {
		t.Fatalf("tooly fit = %v, want 0.8", f)
	}
	if f, why := p.fit(plain, nil); f != 0.2-1 || !strings.Contains(strings.Join(why, ";"), "above band") {
		t.Fatalf("plain fit = %v %v, want 0.2 - 1 (difficulty 3 > max 2)", f, why)
	}
	if f, _ := p.fit(tooly, thinking); f != 1.3 {
		t.Fatalf("reasoning bonus: fit = %v, want 1.3", f)
	}

	no := false
	p = planJevRouting(spec, answersOf("code", map[string]float64{"code": 0.6, "chat": 0.4}, 0, 0, &no, nil))
	if f, _ := p.fit(tooly, nil); f != 1 {
		t.Fatalf("uncalibrated argmax fit = %v, want 1", f)
	}
	if f, _ := p.fit(plain, nil); f != 0 {
		t.Fatalf("uncalibrated non-argmax fit = %v, want 0", f)
	}
}

func TestMultiWeightsOverrides(t *testing.T) {
	fit, w := multiWeights(nil)
	if fit != 1 || w != (autoroute.Weights{Warm: 0.6, Evict: 0.8, Size: 0.1, Speed: 0.2}) {
		t.Fatalf("defaults = %v %+v", fit, w)
	}
	fit, w = multiWeights(&RouterWeights{Fit: f64(2), Warm: f64(0)})
	if fit != 2 || w.Warm != 0 || w.Evict != 0.8 {
		t.Fatalf("overrides = %v %+v", fit, w)
	}
}

// fakeCLM answers decisions batches like a calibrated decider, from a per-test script.
func fakeCLM(t *testing.T, answer func(req map[string]any) map[string]any) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		w.Header().Set("content-type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "clm", "answers": answer(req)})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestResolveRouterJevMultiEndToEnd(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	var script atomic.Value
	script.Store(map[string]any{})
	srv, calls := fakeCLM(t, func(map[string]any) map[string]any { return script.Load().(map[string]any) })
	t.Setenv("ZEROLLAMA_CLM_URL", srv.URL)
	t.Setenv("ZEROLLAMA_CLM_HEADS", "")

	yamlSpec := `
routers:
  smart:
    classifier: jev
    decider: clm
    decider_timeout: 5s
    escalate_at: 4
    questions:
      - id: task
        type: choice
        instructions: What kind of work?
        criteria: {code: write or fix code, chat: conversation}
      - id: difficulty
        type: score
        instructions: How hard?
        labels: ["1", "2", "3", "4", "5"]
    candidates:
      - {model: plain, tier: low, strengths: {task: [chat]}, max_difficulty: 2}
      - {model: tooly, tier: high, strengths: {task: [code]}}
`
	resetRouterFile(t, yamlSpec)
	sig := autoroute.Signals{Surface: autoroute.SurfaceChat, Text: "hello"}

	script.Store(map[string]any{
		"task":       map[string]any{"type": "choice", "choice": "chat", "probabilities": map[string]float64{"chat": 0.95, "code": 0.05}},
		"difficulty": map[string]any{"type": "score", "score": 0.0},
	})
	dec, _, err := s.resolveRouter(t.Context(), "smart", sig)
	if err != nil || !strings.HasPrefix(dec.Candidate, "plain") || dec.Escalated || dec.Classifier != "jev" {
		t.Fatalf("easy chat: dec=%+v err=%v", dec, err)
	}
	if dec.Calibrated == nil || !*dec.Calibrated || dec.Signals["task"] == nil {
		t.Fatalf("signals missing: %+v", dec)
	}

	script.Store(map[string]any{
		"task":       map[string]any{"type": "choice", "choice": "chat", "probabilities": map[string]float64{"chat": 0.9, "code": 0.1}},
		"difficulty": map[string]any{"type": "score", "score": 3.5}, // level 4.5 ≥ escalate_at 4
	})
	dec, _, err = s.resolveRouter(t.Context(), "smart", sig)
	if err != nil || !strings.HasPrefix(dec.Candidate, "tooly") || !dec.Escalated || dec.Reason != "escalated" {
		t.Fatalf("escalation: dec=%+v err=%v", dec, err)
	}

	script.Store(map[string]any{
		"task":       map[string]any{"type": "choice", "choice": "code", "probabilities": map[string]float64{"code": 0.6, "chat": 0.4}, "calibrated": false},
		"difficulty": map[string]any{"type": "score", "score": 4.0, "calibrated": false},
	})
	dec, _, err = s.resolveRouter(t.Context(), "smart", sig)
	if err != nil || !strings.HasPrefix(dec.Candidate, "tooly") || dec.Escalated || *dec.Calibrated {
		t.Fatalf("uncalibrated argmax: dec=%+v err=%v", dec, err)
	}
	if calls.Load() != 3 {
		t.Fatalf("decider calls = %d, want one batch per decision", calls.Load())
	}

	// Decider down ⇒ fail open to a deterministic pick among the router's candidates.
	srv.Close()
	dec, _, err = s.resolveRouter(t.Context(), "smart", sig)
	if err != nil || dec.Reason != "decider-fallback" || dec.Decider != "clm" {
		t.Fatalf("fail-open: dec=%+v err=%v", dec, err)
	}
}
