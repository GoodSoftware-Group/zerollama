package runtimeclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchKVSnapshot(t *testing.T) {
	rt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/kv-snapshot" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"kv_page_bind": map[string]any{"status": "not_implemented"},
		})
	}))
	defer rt.Close()

	t.Setenv("ZEROLLAMA_RUNTIME_URL", rt.URL)
	body, status, err := FetchKVSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusOK {
		t.Fatalf("status %d body %s", status, string(body))
	}
	var snap map[string]any
	if err := json.Unmarshal(body, &snap); err != nil {
		t.Fatal(err)
	}
	pb, ok := snap["kv_page_bind"].(map[string]any)
	if !ok || pb["status"] != "not_implemented" {
		t.Fatalf("kv_page_bind %v", snap["kv_page_bind"])
	}
}

func TestFetchKVSnapshot_noRuntime(t *testing.T) {
	t.Setenv("ZEROLLAMA_RUNTIME_URL", "")
	_, status, err := FetchKVSnapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if status != http.StatusServiceUnavailable {
		t.Fatalf("status %d", status)
	}
}
