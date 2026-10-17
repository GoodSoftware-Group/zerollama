package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckThinkOnContent_Spill(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":     map[string]any{"content": "", "thinking": "B"},
			"done_reason": "stop",
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckThinkOnContent(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "162") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckThinkBudgetCut(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":     map[string]any{"content": "", "thinking": "17 times"},
			"done_reason": "length",
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckThinkBudgetCut(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "164") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckJSONThinkOn(t *testing.T) {
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"message": map[string]any{"content": `{"ok":true}`},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message":     map[string]any{"content": "{"},
			"done_reason": "length",
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckJSONThinkOn(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "149") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckEmptyThoughtMarkers(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "ping<think></think>", "thinking": ""},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckEmptyThoughtMarkers(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "158") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorThinkLanes_Skip(t *testing.T) {
	c := doctorCheckThinkOnContent("http://127.0.0.1:1", doctorLoadedModel{Name: "m"})
	if c.Status != "ok" || !strings.Contains(c.Detail, "skipped") {
		t.Fatalf("%+v", c)
	}
}
