package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckOrphanedThinkClose(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "OK"},
			"done":    true,
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckOrphanedThinkClose(srv.URL, doctorLoadedModel{Name: "qwen3:0.6b", SupportsThinking: true})
	if c.Status != "ok" || !strings.Contains(c.Detail, "no leading") {
		t.Fatalf("%+v", c)
	}

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "</think>\nOK"},
			"done":    true,
		})
	}))
	t.Cleanup(bad.Close)
	c = doctorCheckOrphanedThinkClose(bad.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "02") {
		t.Fatalf("want orphan warn: %+v", c)
	}

	skip := doctorCheckOrphanedThinkClose("http://127.0.0.1:1", doctorLoadedModel{Name: "m"})
	if skip.Status != "ok" || !strings.Contains(skip.Detail, "skipped") {
		t.Fatalf("%+v", skip)
	}
}
