package server

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/server/autoroute"
)

func jevSpec() RouterSpec {
	return RouterSpec{
		Classifier: "jev",
		Decider:    "laya",
		Policies: []RouterPolicy{
			{Label: "code", Description: "programming"},
			{Label: "chat", Description: "small talk"},
			{Label: "math"},
		},
		Candidates: []RouterCandidate{
			{Model: "coder", Labels: []string{"code"}},
			{Model: "general", Labels: []string{"code", "chat", "math"}},
		},
	}
}

func TestBuildJevRouteBodyKeepsPolicyOrder(t *testing.T) {
	spec := jevSpec()
	spec.Policies = []RouterPolicy{{Label: "zeta"}, {Label: "alpha", Description: "first letter"}, {Label: "mu"}}
	body, err := buildJevRouteBody("laya", spec, "hello")
	if err != nil {
		t.Fatal(err)
	}
	s := string(body)
	if !(strings.Index(s, `"zeta"`) < strings.Index(s, `"alpha"`) && strings.Index(s, `"alpha"`) < strings.Index(s, `"mu"`)) {
		t.Fatalf("criteria order lost: %s", s)
	}
	if !strings.Contains(s, `"keep_alive":-1`) || !strings.Contains(s, `"alpha":"first letter"`) || !strings.Contains(s, `"mu":"mu"`) {
		t.Fatalf("body = %s", s)
	}
	call, err := parseDecisionsCall(body)
	if err != nil {
		t.Fatalf("body must be a valid decisions request: %v", err)
	}
	// keep_alive -1 decodes to "forever" (api.Duration maps negatives to the max duration).
	if call.Req.Model != "laya" || call.Req.KeepAlive == nil || call.Req.KeepAlive.Duration < 1000*time.Hour {
		t.Fatalf("call = %+v", call.Req)
	}
}

func TestTrimHeadTailRuneSafe(t *testing.T) {
	s := strings.Repeat("é", 100) // 2 bytes each
	got := trimHeadTail(s, 51)
	if !utf8.ValidString(got) || !strings.Contains(got, "…") {
		t.Fatalf("got %q", got)
	}
	if trimHeadTail("short", 100) != "short" {
		t.Fatal("short strings must pass through")
	}
}

func TestParseJevChoiceCalibration(t *testing.T) {
	_, _, cal, err := parseJevChoice(json.RawMessage(`{"choice":"code","probabilities":{"code":0.9}}`))
	if err != nil || !cal {
		t.Fatalf("missing calibrated (Laya) should mean calibrated: %v %v", cal, err)
	}
	choice, probs, cal, err := parseJevChoice(json.RawMessage(`{"choice":"chat","probabilities":{"chat":0.6},"calibrated":false}`))
	if err != nil || cal || choice != "chat" || probs["chat"] != 0.6 {
		t.Fatalf("openjev answer: %q %v %v %v", choice, probs, cal, err)
	}
	if _, _, _, err := parseJevChoice(nil); err == nil {
		t.Fatal("empty answer must error")
	}
}

func fakeJev(choice string, probs map[string]float64, calibrated bool) routerJevFn {
	return func(context.Context, string, []byte) (string, map[string]float64, bool, error) {
		return choice, probs, calibrated, nil
	}
}

func TestDecideRouterJevActivation(t *testing.T) {
	probs := map[string]float64{"code": 0.7, "chat": 0.2, "math": 0.1}

	dec, err := decideRouterWith(t.Context(), "agent", jevSpec(), "fix my loop", routerBackends{jev: fakeJev("code", probs, true)})
	if err != nil {
		t.Fatal(err)
	}
	// Calibrated: code (0.7) and chat (0.2) clear the 0.15 default → only general covers both.
	if dec.Candidate != "general" || dec.Calibrated == nil || !*dec.Calibrated || dec.Decider != "laya" {
		t.Fatalf("calibrated: %+v", dec)
	}

	dec, err = decideRouterWith(t.Context(), "agent", jevSpec(), "fix my loop", routerBackends{jev: fakeJev("code", probs, false)})
	if err != nil || dec.Candidate != "coder" {
		t.Fatalf("uncalibrated should use argmax only: %+v %v", dec, err)
	}
	if len(dec.Scores) != 3 || !dec.Scores[0].Active || dec.Scores[1].Active {
		t.Fatalf("scores = %+v", dec.Scores)
	}
}

func TestDecideRouterJevFailures(t *testing.T) {
	boom := func(context.Context, string, []byte) (string, map[string]float64, bool, error) {
		return "", nil, false, errors.New("runner exploded")
	}

	spec := jevSpec()
	spec.Fallback = "general"
	dec, err := decideRouterWith(t.Context(), "agent", spec, "x", routerBackends{jev: boom})
	if err != nil || dec.Candidate != "general" || !dec.Fallback {
		t.Fatalf("with fallback: %+v %v", dec, err)
	}

	_, err = decideRouterWith(t.Context(), "agent", jevSpec(), "x", routerBackends{jev: boom})
	if !errors.Is(err, errDeciderUnavailable) {
		t.Fatalf("no fallback: err = %v, want errDeciderUnavailable", err)
	}

	spec = jevSpec()
	spec.Fallback = "general"
	spec.OnDeciderError = "fail"
	if _, err := decideRouterWith(t.Context(), "agent", spec, "x", routerBackends{jev: boom}); !errors.Is(err, errDeciderUnavailable) {
		t.Fatalf("on_decider_error fail must not use fallback: %v", err)
	}

	spec = jevSpec()
	spec.DeciderTimeout = "20ms"
	slow := func(ctx context.Context, _ string, _ []byte) (string, map[string]float64, bool, error) {
		<-ctx.Done()
		return "", nil, false, ctx.Err()
	}
	start := time.Now()
	_, err = decideRouterWith(t.Context(), "agent", spec, "x", routerBackends{jev: slow})
	if !errors.Is(err, errDeciderUnavailable) || time.Since(start) > time.Second {
		t.Fatalf("timeout: err=%v after %s", err, time.Since(start))
	}
}

func TestResolveRouterJevFailsOpenToCandidates(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	resetRouterFile(t, `
routers:
  agent:
    classifier: jev
    decider: no-such-decider
    policies:
      - label: chat
    candidates:
      - model: plain
        labels: [chat]
  strict:
    classifier: jev
    decider: no-such-decider
    on_decider_error: fail
    policies:
      - label: chat
    candidates:
      - model: plain
        labels: [chat]
  loop:
    classifier: jev
    decider: auto
    fallback: plain
    policies:
      - label: chat
    candidates:
      - model: plain
        labels: [chat]
`)
	sig := autoroute.Signals{Surface: autoroute.SurfaceChat, Text: "hello"}

	dec, _, err := s.resolveRouter(t.Context(), "agent", sig)
	if err != nil || !strings.HasPrefix(dec.Candidate, "plain") || dec.Reason != "decider-fallback" || dec.Decider != "no-such-decider" {
		t.Fatalf("fail-open: dec=%+v err=%v", dec, err)
	}

	if _, _, err := s.resolveRouter(t.Context(), "strict", sig); !errors.Is(err, errDeciderUnavailable) {
		t.Fatalf("strict: err = %v", err)
	}

	dec, _, err = s.resolveRouter(t.Context(), "loop", sig)
	if err != nil || dec.Candidate != "plain" || !dec.Fallback {
		t.Fatalf("router-as-decider must be refused and fall back: dec=%+v err=%v", dec, err)
	}
}
