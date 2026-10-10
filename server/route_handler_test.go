package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func TestRouterInventoryIncludesRouterPolicies(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "plain", routePlainTemplate)
	resetRouterFile(t, `
routers:
  smart:
    classifier: jev
    decider: laya
    fallback: plain
    questions:
      - {id: task, type: choice, instructions: "what kind?", criteria: {code: code}}
    candidates:
      - {model: plain, tier: low, strengths: {task: [chat]}, max_difficulty: 2}
`)
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodGet, "/api/router/inventory", nil)
	s.RouterInventoryHandler(c)
	var out struct {
		Routers map[string]struct {
			Classifier string            `json:"classifier"`
			Decider    string            `json:"decider"`
			Questions  int               `json:"questions"`
			Candidates []RouterCandidate `json:"candidates"`
		} `json:"routers"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	r, ok := out.Routers["smart"]
	if !ok || r.Classifier != "jev" || r.Decider != "laya" || r.Questions != 1 || len(r.Candidates) != 1 {
		t.Fatalf("routers = %+v", out.Routers)
	}
	if c := r.Candidates[0]; c.Tier != "low" || c.MaxDifficulty != 2 || c.Strengths["task"][0] != "chat" {
		t.Fatalf("candidate policy = %+v", c)
	}
}

func TestRouterInventoryReportsConfigError(t *testing.T) {
	s := newRouteTestServer(t)
	resetRouterFile(t, "routers:\n  bad: [unclosed\n")
	w := NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request, _ = http.NewRequest(http.MethodGet, "/api/router/inventory", nil)
	s.RouterInventoryHandler(c)
	if !strings.Contains(w.Body.String(), `"router_config_error"`) {
		t.Fatalf("invalid router.yaml must be visible in the inventory: %s", w.Body.String())
	}
}

func TestV1ModelRetrieveVirtualRouter(t *testing.T) {
	s := newRouteTestServer(t)
	createRouteTestModel(t, s, "tooly", routeToolsTemplate)
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/v1/models/:model", s.virtualRouterModelGet(), func(c *gin.Context) {
		c.JSON(http.StatusTeapot, gin.H{"fell_through": c.Param("model")})
	})
	get := func(name string) (int, map[string]any) {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/v1/models/"+name, nil))
		var out map[string]any
		_ = json.Unmarshal(w.Body.Bytes(), &out)
		return w.Code, out
	}

	code, out := get("auto")
	if code != http.StatusOK || out["id"] != "auto" || out["object"] != "model" || out["owned_by"] != "library" {
		t.Fatalf("auto: %d %v", code, out)
	}
	if caps, _ := out["capabilities"].([]any); len(caps) == 0 {
		t.Fatalf("auto should advertise candidate capabilities: %v", out)
	}
	if code, _ := get("tooly"); code != http.StatusTeapot {
		t.Fatalf("real models must fall through to the normal retrieve, got %d", code)
	}
	t.Setenv("ZEROLLAMA_AUTO_ROUTER", "0")
	if code, _ := get("auto"); code != http.StatusTeapot {
		t.Fatalf("disabled router must fall through, got %d", code)
	}
}
