package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestRuntimeInferenceHealthCache(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"waiting":         1,
			"running":         0,
			"inference_state": "idle",
			"llama_server":    true,
		})
	}))
	defer srv.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)

	runtimeHealthCacheMu.Lock()
	runtimeHealthCached = runtimeHealthSnapshot{}
	runtimeHealthCacheURL = ""
	runtimeHealthCachedAt = time.Time{}
	runtimeHealthCacheMu.Unlock()

	ctx := context.Background()
	h1 := runtimeInferenceHealth(ctx)
	h2 := runtimeInferenceHealth(ctx)
	if !h1.ok || !h2.ok {
		t.Fatal("expected ok health snapshots")
	}
	if !h1.llamaLoaded || !h2.llamaLoaded {
		t.Fatal("expected llama_server=true")
	}
	if hits != 1 {
		t.Fatalf("expected 1 /health request, got %d", hits)
	}
}
