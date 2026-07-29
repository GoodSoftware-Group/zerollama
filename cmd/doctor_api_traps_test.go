package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckThinkingGate(t *testing.T) {
	t.Setenv("ZEROLLAMA_THINKING_GATE", "")
	c := doctorCheckThinkingGate(false)
	if c.Status != "ok" || !strings.Contains(c.Detail, "default allow") {
		t.Fatalf("cold default: %+v", c)
	}
	c = doctorCheckThinkingGate(true)
	if c.Status != "warn" || !strings.Contains(c.Detail, "29") {
		t.Fatalf("warm thinking: %+v", c)
	}
	t.Setenv("ZEROLLAMA_THINKING_GATE", "deny")
	c = doctorCheckThinkingGate(true)
	if c.Status != "ok" || !strings.Contains(c.Detail, "deny") {
		t.Fatalf("deny: %+v", c)
	}
}

func TestDoctorCheckUnknownFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if _, ok := body["__minefield_unvalidated_field_probe__"]; ok {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"unknown field: __minefield_unvalidated_field_probe__"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"message":{"content":"hi"}}`))
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckUnknownFields(srv.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckUnknownFields_Accepts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"message":{"content":"hi"}}`))
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckUnknownFields(srv.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "warn" {
		t.Fatalf("want warn when 200: %+v", c)
	}
}

func TestDoctorCheckToolChoiceNone(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{"content": "no tools"},
			}},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckToolChoiceNone(srv.URL, doctorLoadedModel{Name: "m", SupportsTools: true})
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{
					"tool_calls": []map[string]any{{"id": "1"}},
				},
			}},
		})
	}))
	t.Cleanup(bad.Close)
	c = doctorCheckToolChoiceNone(bad.URL, doctorLoadedModel{Name: "m", SupportsTools: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "78") {
		t.Fatalf("want trap 78 warn: %+v", c)
	}
}
