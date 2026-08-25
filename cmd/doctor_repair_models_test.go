package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckThinkRoundtripGenerateEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/chat":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"message": map[string]any{"content": "ping"},
				"done":    true,
			})
		case "/api/generate":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"response": "",
				"thinking": "<answer>42</answer>",
				"done":     true,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	c := doctorCheckThinkRoundtrip(srv.URL, doctorLoadedModel{
		Name:             "milkey:latest",
		SupportsThinking: true,
	})
	if c.Status != "warn" {
		t.Fatalf("status=%s detail=%s", c.Status, c.Detail)
	}
	if !strings.Contains(c.Detail, "generate") {
		t.Fatalf("detail=%s", c.Detail)
	}
	if !strings.Contains(c.FixHint, "--repair-models") {
		t.Fatalf("fix_hint=%s", c.FixHint)
	}
}

func TestRunDoctorRepairModelsRequiresFlag(t *testing.T) {
	cmd := NewDoctorCommand()
	cmd.SetArgs([]string{"--apply"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--repair-models") {
		t.Fatalf("expected --apply requires --repair-models, got %v", err)
	}
}
