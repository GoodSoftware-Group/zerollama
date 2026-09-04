package server

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
)

func TestLoadHandlerMissingModel(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	for _, path := range []string{"/api/load", "/backend/load"} {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+path, bytes.NewBufferString(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		b := new(bytes.Buffer)
		_, _ = b.ReadFrom(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s: expected 400, got %d (%s)", path, resp.StatusCode, b.String())
		}
	}
}

func TestLoadHandlerNotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/load", bytes.NewBufferString(`{"model":"definitely-missing-la21:latest"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound && resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 404/400, got %d", resp.StatusCode)
	}
}

func TestLoadHandlerCloudRejected(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+"/api/load", bytes.NewBufferString(`{"model":"gpt-4:cloud"}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := local.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400 for cloud, got %d", resp.StatusCode)
	}
}

func TestUnloadHandlerWrongKey(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	s := &Server{}
	router, err := s.GenerateRoutes(nil)
	if err != nil {
		t.Fatal(err)
	}
	local := httptest.NewServer(router)
	defer local.Close()

	for _, path := range []string{"/api/unload"} {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, local.URL+path, bytes.NewBufferString(`{"model_id":"llama3.2:latest"}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := local.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		b := new(bytes.Buffer)
		_, _ = b.ReadFrom(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s: expected 400, got %d (%s)", path, resp.StatusCode, b.String())
		}
		if !bytes.Contains(b.Bytes(), []byte("model")) {
			t.Fatalf("%s: error should name model key, got %s", path, b.String())
		}
	}
}

func TestSnapshotHasLoadedModel(t *testing.T) {
	ps := api.ProcessResponse{Models: []api.ProcessModelResponse{{Name: "llama3.2:3b", Model: "llama3.2:3b"}}}
	if !snapshotHasLoadedModel(ps, "llama3.2:3b") {
		t.Fatal("expected hit")
	}
	if snapshotHasLoadedModel(ps, "other:latest") {
		t.Fatal("expected miss")
	}
}
