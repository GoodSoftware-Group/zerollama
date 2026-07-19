package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/version"
)

func TestVersionHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/version", nil)
	VersionHandler(c)
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["version"] == nil || body["version"] == "" {
		t.Fatalf("version=%v", body["version"])
	}
	if edge, ok := body["edge_build"].(bool); !ok || edge {
		t.Fatalf("edge_build=%v want false", body["edge_build"])
	}
	if body["distribution"] != "zerollama" {
		t.Fatalf("distribution=%v want zerollama", body["distribution"])
	}
	z, ok := body["zerollama"].(map[string]any)
	if !ok {
		t.Fatal("missing zerollama block")
	}
	if caps, ok := z["capabilities"].(map[string]any); !ok || caps["mlx_qos"] != true {
		t.Fatalf("capabilities=%v", z["capabilities"])
	}
	caps := z["capabilities"].(map[string]any)
	for _, key := range []string{"runtime_config", "can_load", "metrics", "admission_retry_after", "error_timings", "empty_gen_classify", "pin_reserve", "propose_sidecar"} {
		if caps[key] != true {
			t.Fatalf("expected %s true, caps=%v", key, caps)
		}
	}
	if caps["stable_multi_model_swap"] != false {
		t.Fatalf("expected stable_multi_model_swap false, caps=%v", caps)
	}
}

func TestVersionHandlerEdgeBuild(t *testing.T) {
	version.EdgeBuild = "true"
	t.Cleanup(func() { version.EdgeBuild = "false" })

	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodGet, "/api/version", nil)
	VersionHandler(c)

	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if edge, ok := body["edge_build"].(bool); !ok || !edge {
		t.Fatalf("edge_build=%v want true", body["edge_build"])
	}
}
