package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckTokenCeiling_SkippedByDefault(t *testing.T) {
	t.Setenv("ZEROLLAMA_DOCTOR_DEEP", "")
	c := doctorCheckTokenCeiling("http://127.0.0.1:1", doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "ok" || !strings.Contains(c.Detail, "skipped") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckTokenCeiling_EmptyContent(t *testing.T) {
	t.Setenv("ZEROLLAMA_DOCTOR_DEEP", "1")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":     map[string]any{"content": "", "thinking": strings.Repeat("reason ", 40)},
			"done":        true,
			"done_reason": "length",
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckTokenCeiling(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "12") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckTokenCeiling_OK(t *testing.T) {
	t.Setenv("ZEROLLAMA_DOCTOR_DEEP", "1")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":     map[string]any{"content": "def validate():\n  pass", "thinking": "plan"},
			"done":        true,
			"done_reason": "stop",
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckTokenCeiling(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}
}
