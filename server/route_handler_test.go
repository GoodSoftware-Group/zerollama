package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/middleware"
)

func postRoute(t *testing.T, s *Server, body string) (int, map[string]any) {
	t.Helper()
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodPost, "/v1/route", strings.NewReader(body))
	s.RouteHandler(c)
	var out map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &out)
	return w.Code, out
}

func TestRouteHandlerDryRun(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	t.Run("native chat with tools", func(t *testing.T) {
		code, out := postRoute(t, s, `{"model":"auto","messages":[{"role":"user","content":"weather?"}],
			"tools":[{"type":"function","function":{"name":"get_weather"}}]}`)
		if code != http.StatusOK {
			t.Fatalf("status %d: %v", code, out)
		}
		if !strings.HasPrefix(out["model"].(string), "tooly") || out["surface"] != "chat" {
			t.Fatalf("out = %v", out)
		}
		auto := out["auto"].(map[string]any)
		filtered := auto["filtered"].([]any)
		if len(filtered) != 1 || !strings.Contains(filtered[0].(map[string]any)["reason"].(string), "tools") {
			t.Fatalf("filtered = %v", filtered)
		}
	})

	t.Run("openai content parts", func(t *testing.T) {
		code, out := postRoute(t, s, `{"router":"auto","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
		if code != http.StatusOK || out["surface"] != "chat" {
			t.Fatalf("status %d: %v", code, out)
		}
	})

	t.Run("image surface without image models explains", func(t *testing.T) {
		code, out := postRoute(t, s, `{"model":"auto","prompt":"a red fox","width":512,"height":512}`)
		if code != http.StatusServiceUnavailable {
			t.Fatalf("status %d: %v", code, out)
		}
		if len(out["filtered"].([]any)) != 2 {
			t.Fatalf("filtered = %v", out["filtered"])
		}
	})

	t.Run("generate", func(t *testing.T) {
		code, out := postRoute(t, s, `{"model":"auto","prompt":"once upon a time"}`)
		if code != http.StatusOK || out["surface"] != "generate" {
			t.Fatalf("status %d: %v", code, out)
		}
	})

	t.Run("not a router", func(t *testing.T) {
		code, _ := postRoute(t, s, `{"model":"plain","prompt":"x"}`)
		if code != http.StatusNotFound {
			t.Fatalf("status %d", code)
		}
	})

	t.Run("bad surface", func(t *testing.T) {
		code, _ := postRoute(t, s, `{"model":"auto","surface":"speech","prompt":"x"}`)
		if code != http.StatusBadRequest {
			t.Fatalf("status %d", code)
		}
	})
}

func TestRouterDecideHandlerAuto(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)

	w := createRequest(t, s.RouterDecideHandler, map[string]string{"router": "auto", "prompt": "hello"})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	var dec RouterDecision
	if err := json.Unmarshal(w.Body.Bytes(), &dec); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(dec.Candidate, "plain") || dec.Classifier != "deterministic" {
		t.Fatalf("dec = %+v", dec)
	}

	w = createRequest(t, s.RouterDecideHandler, map[string]string{"router": "nope", "prompt": "hello"})
	if w.Code != http.StatusNotFound {
		t.Fatalf("unknown router status %d", w.Code)
	}
}

func TestRouterInventoryHandler(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodGet, "/api/router/inventory", nil)
	s.RouterInventoryHandler(c)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var out struct {
		Candidates []map[string]any `json:"candidates"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.Candidates) != 1 {
		t.Fatalf("candidates = %v", out.Candidates)
	}
	row := out.Candidates[0]
	if !strings.HasPrefix(row["name"].(string), "tooly") || row["can_load"] != true || row["context_length"].(float64) != 8192 {
		t.Fatalf("row = %v", row)
	}
	if _, leaked := row["SchedKey"]; leaked {
		t.Fatal("scheduler key must not be serialised")
	}
	caps := row["capabilities"].([]any)
	if !strings.Contains(strings.Trim(strings.Join(anyStrings(caps), ","), ","), "tools") {
		t.Fatalf("capabilities = %v", caps)
	}
}

func anyStrings(xs []any) []string {
	out := make([]string, len(xs))
	for i, x := range xs {
		out[i], _ = x.(string)
	}
	return out
}

func listModels(t *testing.T, s *Server, viaV1 bool) []api.ListModelResponse {
	t.Helper()
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodGet, "/api/tags", nil)
	if viaV1 {
		c.Set(middleware.CtxKeyListVirtualRouters, true)
	}
	s.ListHandler(c)
	var resp api.ListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	return resp.Models
}

func TestListVirtualAutoOnlyOnV1(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	hasAuto := func(ms []api.ListModelResponse) *api.ListModelResponse {
		for i := range ms {
			if ms[i].Name == "auto" {
				return &ms[i]
			}
		}
		return nil
	}
	if hasAuto(listModels(t, s, false)) != nil {
		t.Fatal("/api/tags must not list virtual auto (bench/launch enumerate it)")
	}
	auto := hasAuto(listModels(t, s, true))
	if auto == nil {
		t.Fatal("/v1/models should list auto")
	}
	if auto.Details.Family != "router" || !strings.Contains(strings.Join(capStrings(auto), ","), "tools") {
		t.Fatalf("auto entry = %+v", auto)
	}

	t.Setenv("ZEROLLAMA_AUTO_ROUTER", "0")
	if hasAuto(listModels(t, s, true)) != nil {
		t.Fatal("disabled auto router must not be listed")
	}
}

func capStrings(m *api.ListModelResponse) []string {
	out := make([]string, len(m.Capabilities))
	for i, c := range m.Capabilities {
		out[i] = string(c)
	}
	return out
}

func TestRouterMetricsAfterInBandRewrite(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)

	if code, _, body := routeChat(t, s, api.ChatRequest{
		Model: "auto", Messages: []api.Message{{Role: "user", Content: "hi"}}, Tools: []api.Tool{weatherTool},
	}); code != http.StatusOK {
		t.Fatalf("chat status %d: %s", code, body)
	}
	_, _, _ = routeChat(t, s, api.ChatRequest{
		Model: "auto", Messages: []api.Message{{Role: "user", Content: "look", Images: []api.ImageData{[]byte("x")}}},
	})

	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodGet, "/api/metrics", nil)
	s.MetricsHandler(c)
	text := w.Body.String()
	for _, want := range []string{
		`zerollama_router_decisions_total{router="auto",surface="chat",reason="only-candidate"}`,
		`zerollama_router_errors_total{router="auto",surface="chat"}`,
		"zerollama_router_decision_latency_ms_count",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %s:\n%s", want, text)
		}
	}
}

func TestVersionAdvertisesAutoRouter(t *testing.T) {
	caps := zerollamaVersionCapabilities()
	ar, ok := caps["auto_router"].(map[string]any)
	if !ok || ar["dry_run"] != "/v1/route" || ar["name"] != "auto" {
		t.Fatalf("auto_router = %v", caps["auto_router"])
	}
	b, _ := json.Marshal(ar)
	if !bytes.Contains(b, []byte(`"enabled":true`)) {
		t.Fatalf("expected enabled by default: %s", b)
	}
}
