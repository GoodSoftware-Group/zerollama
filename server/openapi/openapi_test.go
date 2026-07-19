package openapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestDocumentInjectsVersionAndServer(t *testing.T) {
	doc, err := Document("http://127.0.0.1:2083")
	if err != nil {
		t.Fatal(err)
	}
	info := doc["info"].(map[string]any)
	if info["title"] != "zerollama API" {
		t.Fatalf("title=%v", info["title"])
	}
	if info["version"] == "" || info["version"] == nil {
		t.Fatal("missing version")
	}
	servers := doc["servers"].([]any)
	srv := servers[0].(map[string]any)
	if srv["url"] != "http://127.0.0.1:2083" {
		t.Fatalf("server url=%v", srv["url"])
	}
	paths := doc["paths"].(map[string]any)
	for _, p := range []string{
		"/v1/audio/speech", "/v1/audio/voices", "/openapi.json", "/docs",
		"/api/status", "/api/can-load", "/api/propose-load", "/api/pin", "/api/metrics", "/api/version",
	} {
		if _, ok := paths[p]; !ok {
			t.Fatalf("missing path %s", p)
		}
	}
	schemas := doc["components"].(map[string]any)["schemas"].(map[string]any)
	for _, s := range []string{
		"FleetStatusResponse", "CanLoadRequest", "CanLoadResponse", "InferenceError",
		"InferenceConfigStatus", "PinRequest", "PinResponse", "PinStatus",
		"ProposeLoadRequest", "ProposeLoadResponse",
	} {
		if _, ok := schemas[s]; !ok {
			t.Fatalf("missing schema %s", s)
		}
	}
	inference := schemas["FleetStatusResponse"].(map[string]any)["properties"].(map[string]any)["inference"].(map[string]any)
	infProps := inference["properties"].(map[string]any)
	if _, ok := infProps["pins"]; !ok {
		t.Fatal("FleetStatusResponse.inference missing pins")
	}
	cause := schemas["InferenceError"].(map[string]any)["properties"].(map[string]any)["cause"].(map[string]any)
	enums, _ := cause["enum"].([]any)
	wantCauses := map[string]bool{
		"host_unstable": true, "runtime_pin_gguf": true,
		"runtime_pin_ggml": true, "runtime_vram": true,
	}
	for _, e := range enums {
		delete(wantCauses, e.(string))
	}
	if len(wantCauses) > 0 {
		t.Fatalf("InferenceError.cause missing enums %v", wantCauses)
	}
}

func TestHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	Register(r)

	req := httptest.NewRequest(http.MethodGet, "/openapi.json", nil)
	req.Host = "example.test:2083"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("json status=%d body=%s", w.Code, w.Body.String())
	}
	var doc map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	servers := doc["servers"].([]any)
	if servers[0].(map[string]any)["url"] != "http://example.test:2083" {
		t.Fatalf("injected server=%v", servers[0])
	}

	req = httptest.NewRequest(http.MethodGet, "/openapi.yaml", nil)
	req.Host = "example.test:2083"
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "zerollama API") {
		t.Fatalf("yaml status=%d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "yaml") {
		t.Fatalf("content-type=%q", ct)
	}

	req = httptest.NewRequest(http.MethodGet, "/docs", nil)
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "swagger-ui") {
		t.Fatalf("docs status=%d", w.Code)
	}
}
