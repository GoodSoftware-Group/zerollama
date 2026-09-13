package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckThinkOffSpill_Skip(t *testing.T) {
	c := doctorCheckThinkOffSpill("http://127.0.0.1:1", doctorLoadedModel{Name: "m"})
	if c.Status != "ok" || !strings.Contains(c.Detail, "skipped") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckThinkOffSpill_TagsInContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if _, ok := req["chat_template_kwargs"]; ok {
			_ = json.NewEncoder(w).Encode(map[string]any{"debug_info": map[string]any{"rendered_template": "x"}})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "reason</think>\nping"},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckThinkOffSpill(srv.URL, doctorLoadedModel{Name: "ling", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "126") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckThinkOffSpill_Clean(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "ping"},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckThinkOffSpill(srv.URL, doctorLoadedModel{Name: "qwen3:0.6b", SupportsThinking: true})
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}
}
