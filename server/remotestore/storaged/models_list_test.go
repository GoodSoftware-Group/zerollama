package storaged

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/server/remotestore"
)

func TestHandleModelsListTableShape(t *testing.T) {
	dir := t.TempDir()
	rel := filepath.Join("registry.ollama.ai", "library", "tiny", "latest")
	mp := filepath.Join(dir, "manifests", rel)
	if err := os.MkdirAll(filepath.Dir(mp), 0o755); err != nil {
		t.Fatal(err)
	}
	mf := map[string]any{
		"schemaVersion": 2,
		"config":        map[string]any{"digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "size": 100},
		"layers": []map[string]any{
			{"digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "size": 1_000_000},
		},
	}
	b, _ := json.Marshal(mf)
	if err := os.WriteFile(mp, b, 0o644); err != nil {
		t.Fatal(err)
	}
	auth, err := remotestore.NewAuth("test-secret-for-storage-ls")
	if err != nil {
		t.Fatal(err)
	}
	s := New(dir, auth)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	if err := auth.SignRequest(req, nil); err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	s.ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ModelsListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Models) != 1 {
		t.Fatalf("models=%d", len(resp.Models))
	}
	m := resp.Models[0]
	if m.Name != "tiny:latest" {
		t.Fatalf("name=%q", m.Name)
	}
	if m.Size != 1_000_100 {
		t.Fatalf("size=%d", m.Size)
	}
	if m.Layers != 1 {
		t.Fatalf("layers=%d", m.Layers)
	}
}

func TestStorageDisplayName(t *testing.T) {
	if got := storageDisplayName("registry.ollama.ai/library/llama3.2/3b"); got != "llama3.2:3b" {
		t.Fatalf("got %q", got)
	}
	if got := storageDisplayName("registry.ollama.ai/comfy/flux1-dev/latest"); got != "comfy/flux1-dev:latest" {
		t.Fatalf("got %q", got)
	}
}
