package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorGETStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	st, err := doctorGETStatus(srv.Client(), srv.URL)
	if err != nil || st != http.StatusOK {
		t.Fatalf("status=%d err=%v", st, err)
	}
}

func TestDoctorCheckModelReadiness_NoListener(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "127.0.0.1:1")
	c := doctorCheckModelReadiness()
	if c.Status != "ok" {
		t.Fatalf("unreachable host should skip ok: %+v", c)
	}
	if !strings.Contains(c.Name, "112") {
		t.Fatalf("name=%s", c.Name)
	}
}

func TestDoctorFetchLoadedModelsEmptyPS(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
	}))
	t.Cleanup(srv.Close)
	loaded, err := doctorFetchLoadedModels(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != 0 {
		t.Fatalf("%+v", loaded)
	}
}
