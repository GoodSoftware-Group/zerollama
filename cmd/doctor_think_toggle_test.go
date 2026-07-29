package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckThinkToggleInjection(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if dbg, _ := req["_debug_render_only"].(bool); dbg {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"_debug_info": map[string]any{
					"rendered_template": "<|im_start|>user\nReply with exactly: OK /think\n",
				},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "OK"},
			"done":    true,
		})
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckThinkToggleInjection(srv.URL, doctorLoadedModel{Name: "qwen3:0.6b", SupportsThinking: true})
	if c.Status != "ok" || !strings.Contains(c.Detail, "injects") {
		t.Fatalf("%+v", c)
	}

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if dbg, _ := req["_debug_render_only"].(bool); dbg {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"_debug_info": map[string]any{"rendered_template": "user OK /think"},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "OK /think"},
			"done":    true,
		})
	}))
	t.Cleanup(bad.Close)
	c = doctorCheckThinkToggleInjection(bad.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "66") {
		t.Fatalf("want leak warn: %+v", c)
	}
}
