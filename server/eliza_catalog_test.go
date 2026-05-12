package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func resetElizaCatalogAfterTest(t *testing.T) {
	t.Helper()
	t.Cleanup(func() {
		elizaCatalogMu.Lock()
		elizaCatalogCached = nil
		elizaCatalogFetched = time.Time{}
		elizaCatalogTTL = time.Hour
		elizaCatalogMu.Unlock()
	})
}

func TestMergeElizaCloudModels_NoFetchWhenNoCloud(t *testing.T) {
	t.Setenv("OLLAMA_NO_CLOUD", "1")
	local := []api.ListModelResponse{{Model: "local:latest", Name: "local:latest"}}
	out := mergeElizaCloudModels(context.Background(), local)
	if len(out) != 1 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestMergeElizaCloudModels_AppendsFromUpstream(t *testing.T) {
	resetElizaCatalogAfterTest(t)
	t.Setenv("OLLAMA_NO_CLOUD", "0")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/models" {
			t.Fatalf("path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"acme/foo"}]}`))
	}))
	defer srv.Close()

	orig := cloudProxyBaseURL
	cloudProxyBaseURL = srv.URL
	t.Cleanup(func() { cloudProxyBaseURL = orig })

	elizaCatalogMu.Lock()
	elizaCatalogCached = nil
	elizaCatalogFetched = time.Time{}
	elizaCatalogTTL = time.Hour
	elizaCatalogMu.Unlock()

	local := []api.ListModelResponse{{Model: "local:latest", Name: "local:latest"}}
	out := mergeElizaCloudModels(context.Background(), local)
	if len(out) != 2 {
		t.Fatalf("len=%d want 2", len(out))
	}
	if out[1].Model != "acme/foo:cloud" {
		t.Fatalf("got %q", out[1].Model)
	}
}

func TestMergeElizaCloudModels_DedupesByModelName(t *testing.T) {
	resetElizaCatalogAfterTest(t)
	t.Setenv("OLLAMA_NO_CLOUD", "0")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"dup"}]}`))
	}))
	defer srv.Close()

	orig := cloudProxyBaseURL
	cloudProxyBaseURL = srv.URL
	t.Cleanup(func() { cloudProxyBaseURL = orig })

	elizaCatalogMu.Lock()
	elizaCatalogCached = nil
	elizaCatalogFetched = time.Time{}
	elizaCatalogTTL = time.Hour
	elizaCatalogMu.Unlock()

	local := []api.ListModelResponse{{Model: "dup:cloud", Name: "dup:cloud"}}
	out := mergeElizaCloudModels(context.Background(), local)
	if len(out) != 1 {
		t.Fatalf("len=%d want 1 (dedupe)", len(out))
	}
}

func TestCacheTTLFromCacheControl(t *testing.T) {
	h := make(http.Header)
	h.Set("Cache-Control", "public, s-maxage=120, stale-while-revalidate=7200")
	if got := cacheTTLFromCacheControl(h); got != 2*time.Minute {
		t.Fatalf("got %v", got)
	}
	if got := cacheTTLFromCacheControl(make(http.Header)); got != time.Hour {
		t.Fatalf("default got %v", got)
	}
}
