package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckHistoryAssembly(t *testing.T) {
	const marker = doctorHistoryMarker
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		msgs, _ := req["messages"].([]any)
		render := "user/assistant scaffold"
		// Detect last-assistant write-field arms vs multi-turn strip arm.
		if len(msgs) == 2 {
			asst, _ := msgs[1].(map[string]any)
			if th, _ := asst["thinking"].(string); strings.Contains(th, marker) {
				render = "<think>" + marker + "</think>ok"
			}
			// reasoning key ignored → no marker
		} else {
			// multi-turn: strip prior thinking (trap 04 signature)
			render = "<|im_start|>user\nStep 3"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"debug_info": map[string]any{"rendered_template": render},
		})
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckHistoryAssembly(srv.URL, doctorLoadedModel{Name: "qwen3:0.6b", SupportsThinking: true})
	if c.Status != "warn" {
		t.Fatalf("want warn for stripped history: %+v", c)
	}
	if !strings.Contains(c.Detail, "04") || !strings.Contains(c.Detail, "20") {
		t.Fatalf("detail=%s", c.Detail)
	}
	if !strings.Contains(c.Detail, "25") {
		t.Fatalf("expected trap 25 mention: %s", c.Detail)
	}
}

func TestDoctorCheckHistoryAssembly_Preserved(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"debug_info": map[string]any{
				"rendered_template": "prior " + doctorHistoryMarker + " and again " + doctorHistoryMarker,
			},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckHistoryAssembly(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	// thinking+reasoning both contain marker in this stub → unusual but ok/warn
	if !strings.Contains(c.Detail, "preserved") && c.Status == "warn" && strings.Contains(c.Detail, "04") {
		t.Fatalf("unexpected strip: %+v", c)
	}
	if !strings.Contains(c.Detail, "trap 04") && !strings.Contains(c.Detail, "preserved") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckHistoryAssembly_SkipNonThinking(t *testing.T) {
	c := doctorCheckHistoryAssembly("http://127.0.0.1:1", doctorLoadedModel{Name: "m"})
	if c.Status != "ok" || !strings.Contains(c.Detail, "skipped") {
		t.Fatalf("%+v", c)
	}
}
