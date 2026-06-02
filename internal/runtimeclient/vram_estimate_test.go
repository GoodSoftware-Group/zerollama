package runtimeclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProbeVramEstimate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/vram-estimate" {
			t.Fatalf("path %q", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"vram_budget": map[string]any{"fits_with_margin": true},
		})
	}))
	defer srv.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", srv.URL)
	out := ProbeVramEstimate(context.Background(), "/tmp/m.gguf", map[string]any{"num_ctx": 4096})
	if out == nil {
		t.Fatal("expected response")
	}
}
