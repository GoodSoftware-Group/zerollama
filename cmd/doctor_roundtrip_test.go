package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckReasoningRoundTrip(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if kwargs, _ := req["chat_template_kwargs"].(map[string]any); kwargs != nil {
			if _, ok := kwargs["truncate_history_thinking"]; ok {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"unknown field: chat_template_kwargs.truncate_history_thinking"}`))
				return
			}
		}
		msgs, _ := req["messages"].([]any)
		render := "no-marker"
		if len(msgs) >= 2 {
			asst, _ := msgs[1].(map[string]any)
			if th, _ := asst["thinking"].(string); strings.Contains(th, doctorHistoryMarker) {
				render = "<think>" + doctorHistoryMarker + "</think>\nA1"
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"_debug_info": map[string]any{"rendered_template": render},
		})
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckReasoningRoundTrip(srv.URL, doctorLoadedModel{Name: "qwen3:0.6b", SupportsThinking: true})
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}
	if !strings.Contains(c.Detail, "thinking") || !strings.Contains(c.Detail, "63") {
		t.Fatalf("detail=%s", c.Detail)
	}
	if !strings.Contains(c.Detail, "truncate_history_thinking") {
		t.Fatalf("expected gate note: %s", c.Detail)
	}
}

func TestDoctorCheckReasoningRoundTrip_WrongField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if kwargs, _ := req["chat_template_kwargs"].(map[string]any); kwargs != nil {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"unknown field: chat_template_kwargs.truncate_history_thinking"}`))
			return
		}
		// Broken stack: only reasoning_content reaches the template.
		msgs, _ := req["messages"].([]any)
		render := "empty"
		if len(msgs) >= 2 {
			asst, _ := msgs[1].(map[string]any)
			if rc, _ := asst["reasoning_content"].(string); strings.Contains(rc, doctorHistoryMarker) {
				render = doctorHistoryMarker
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"_debug_info": map[string]any{"rendered_template": render},
		})
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckReasoningRoundTrip(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" {
		t.Fatalf("want warn: %+v", c)
	}
}

func TestDoctorCheckReasoningRoundTrip_Skip(t *testing.T) {
	c := doctorCheckReasoningRoundTrip("http://127.0.0.1:1", doctorLoadedModel{Name: "m"})
	if c.Status != "ok" || !strings.Contains(c.Detail, "skipped") {
		t.Fatalf("%+v", c)
	}
}
