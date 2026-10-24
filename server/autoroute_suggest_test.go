package server

import (
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestSuggestionHeuristicAdvisoryOnly(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)

	code, hdr, body := routeChat(t, s, api.ChatRequest{
		Model: "auto", Messages: []api.Message{{Role: "user", Content: "Can you make me a picture of a fox in the snow?"}},
	})
	if code != http.StatusOK {
		t.Fatalf("chat must still be served as text: %d %s", code, body)
	}
	if hdr.Get("X-Zerollama-Route-Suggest") != "image" || hdr.Get("X-Zerollama-Route-Suggest-Endpoint") != "/v1/images/generations" {
		t.Fatalf("headers = %v", hdr)
	}
	if hdr.Get("X-Zerollama-Route-Suggest-Model") != "" {
		t.Fatal("no local image model ⇒ no suggested model")
	}

	sig := chatSig(t, api.ChatRequest{Messages: []api.Message{{Role: "user", Content: "Can you make me a picture of a fox?"}}})
	dec, _, err := s.resolveRouter(t.Context(), "auto", sig)
	if err != nil || dec.Suggestion == nil {
		t.Fatalf("dry run: %+v %v", dec, err)
	}
	sug := dec.Suggestion
	if sug.Source != "heuristic" || sug.Unavailable == "" || sug.Body["prompt"] != "Can you make me a picture of a fox?" {
		t.Fatalf("suggestion = %+v", sug)
	}

	// Ordinary turns carry no suggestion.
	_, hdr, _ = routeChat(t, s, api.ChatRequest{Model: "auto", Messages: []api.Message{{Role: "user", Content: "what's 2+2?"}}})
	if hdr.Get("X-Zerollama-Route-Suggest") != "" {
		t.Fatalf("unexpected suggestion: %v", hdr)
	}
}

func TestSuggestionSpeechReadsBackLastAnswer(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	sig := chatSig(t, api.ChatRequest{Messages: []api.Message{
		{Role: "user", Content: "what is a haiku?"},
		{Role: "assistant", Content: "A haiku is a three-line poem."},
		{Role: "user", Content: "read that out loud"},
	}})
	dec, _, err := s.resolveRouter(t.Context(), "auto", sig)
	if err != nil || dec.Suggestion == nil || dec.Suggestion.Modality != "speech" || dec.Suggestion.Endpoint != "/v1/audio/speech" {
		t.Fatalf("dec = %+v err %v", dec, err)
	}
	if dec.Suggestion.Body["input"] != "A haiku is a three-line poem." {
		t.Fatalf("speech input = %v", dec.Suggestion.Body["input"])
	}
}

func TestSuggestionModes(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	sig := chatSig(t, api.ChatRequest{Messages: []api.Message{{Role: "user", Content: "Animate this logo spinning"}}})

	for mode, want := range map[string]bool{"": true, "auto": true, "jev": false, "off": false} {
		t.Setenv("ZEROLLAMA_ROUTER_SUGGEST", mode)
		dec, _, err := s.resolveRouter(t.Context(), "auto", sig)
		if err != nil {
			t.Fatal(err)
		}
		if got := dec.Suggestion != nil; got != want {
			t.Errorf("mode %q: suggestion=%v want %v", mode, got, want)
		}
	}
}

func TestSuggestionFromJevQuestion(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)

	var asked atomic.Int32
	srv, _ := fakeCLM(t, func(req map[string]any) map[string]any {
		qs, _ := req["questions"].(map[string]any)
		out := map[string]any{}
		if _, ok := qs["task"]; ok {
			out["task"] = map[string]any{"type": "choice", "choice": "chat", "probabilities": map[string]float64{"chat": 0.9}}
		}
		if _, ok := qs["output_modality"]; ok {
			asked.Add(1)
			out["output_modality"] = map[string]any{"type": "choice", "choice": "music", "probabilities": map[string]float64{"music": 0.8, "text": 0.2}}
		}
		return out
	})
	t.Setenv("ZEROLLAMA_CLM_URL", srv.URL)
	resetRouterFile(t, `
routers:
  smart:
    classifier: jev
    decider: clm
    decider_timeout: 5s
    questions:
      - {id: task, type: choice, instructions: "kind?", criteria: {chat: talk}}
      - {id: output_modality, type: choice, instructions: "reply as?", criteria: {text: text, image: picture, speech: voice, video: clip, music: song}}
    candidates:
      - {model: plain, tier: low, strengths: {task: [chat]}}
`)
	c, _ := gin.CreateTestContext(NewRecorder())
	req := api.ChatRequest{Messages: []api.Message{{Role: "user", Content: "something to study to"}}}
	sig := chatRouteSignals(c, &req)

	_, dec, err := s.applyRouterRewrite(t.Context(), "smart", sig)
	if err != nil || dec.Suggestion == nil {
		t.Fatalf("turn 1: %+v %v", dec, err)
	}
	if sug := dec.Suggestion; sug.Modality != "music" || sug.Source != "jev" || sug.Probability != 0.8 || sug.Endpoint != "/v1/audio/generations" {
		t.Fatalf("suggestion = %+v", sug)
	}
	if !strings.Contains(musicCaption(dec), "something to study to") {
		t.Fatalf("music body = %v", dec.Suggestion.Body)
	}
	if input, _ := dec.Suggestion.Body["input"].(string); !strings.Contains(input, "something to study to") {
		t.Fatalf("music lyrics (input) missing: %v", dec.Suggestion.Body)
	}

	// Turn 2 is sticky (decider batch skipped) — the modality question alone is still asked.
	before := asked.Load()
	_, dec, err = s.applyRouterRewrite(t.Context(), "smart", sig)
	if err != nil || !dec.Sticky || dec.Suggestion == nil || asked.Load() != before+1 {
		t.Fatalf("sticky turn: dec=%+v err=%v asked %d→%d", dec, err, before, asked.Load())
	}
}

func musicCaption(dec *RouterDecision) string {
	v, _ := dec.Suggestion.Body["instructions"].(string)
	return v
}

func TestSuggestMetrics(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	routeChat(t, s, api.ChatRequest{Model: "auto", Messages: []api.Message{{Role: "user", Content: "please sketch a cat"}}})
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodGet, "/api/metrics", nil)
	s.MetricsHandler(c)
	if !strings.Contains(w.Body.String(), `zerollama_router_suggest_total{modality="image",source="heuristic"}`) {
		t.Fatalf("metrics:\n%s", w.Body.String())
	}
}
