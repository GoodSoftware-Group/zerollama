package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestRuntimeKVSnapshotHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/kv-snapshot" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kv_forward_plans": []any{},
			"kv_page_bind":     map[string]any{"status": "not_implemented"},
			"kv_page_migration": map[string]any{
				"migration_summary": map[string]any{
					"pages_live":          1,
					"n_layers":            2,
					"full_plan_endpoint":  "/internal/kv-snapshot",
				},
			},
		})
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)

	s := &Server{}
	r := gin.New()
	r.GET("/internal/kv-snapshot", internalLoopbackOnly(), s.RuntimeKVSnapshotHandler)

	req := httptest.NewRequest(http.MethodGet, "/internal/kv-snapshot", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["kv_page_bind"]; !ok {
		t.Fatalf("missing kv_page_bind: %v", body)
	}
	if mig, ok := body["kv_page_migration"].(map[string]any); !ok {
		t.Fatalf("missing kv_page_migration: %v", body)
	} else if _, ok := mig["migration_summary"]; !ok {
		t.Fatalf("missing migration_summary: %v", mig)
	}
}

func TestRuntimeKVSnapshotHandler_noRuntime(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")

	s := &Server{}
	r := gin.New()
	r.GET("/internal/kv-snapshot", internalLoopbackOnly(), s.RuntimeKVSnapshotHandler)

	req := httptest.NewRequest(http.MethodGet, "/internal/kv-snapshot", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d body %s", w.Code, w.Body.String())
	}
}
