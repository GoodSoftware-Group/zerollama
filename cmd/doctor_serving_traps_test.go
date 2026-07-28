package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDoctorMessageHelpers(t *testing.T) {
	resp := map[string]any{
		"message": map[string]any{
			"content":  "hi",
			"thinking": "reason",
			"tool_calls": []any{
				map[string]any{"function": map[string]any{"name": "get_time"}},
			},
		},
	}
	if got := doctorMessageString(resp, "content"); got != "hi" {
		t.Fatalf("content=%q", got)
	}
	if got := doctorMessageString(resp, "thinking"); got != "reason" {
		t.Fatalf("thinking=%q", got)
	}
	if calls := doctorMessageToolCalls(resp); len(calls) != 1 {
		t.Fatalf("tool_calls len=%d", len(calls))
	}
	if doctorMessageString(map[string]any{}, "content") != "" {
		t.Fatal("empty resp should yield empty content")
	}
}

func TestDoctorCheckServingTrapsNoAPI(t *testing.T) {
	// With no listener on the usual ports and OLLAMA_HOST pointing nowhere,
	// serving traps should warn rather than fail.
	t.Setenv("OLLAMA_HOST", "127.0.0.1:1")
	checks := doctorCheckServingTraps()
	if len(checks) == 0 {
		t.Fatal("expected at least one check")
	}
	if checks[0].Status != "warn" {
		t.Fatalf("status=%s want warn (%s)", checks[0].Status, checks[0].Detail)
	}
}

func TestDoctorFetchLoadedModelsAndProbes(t *testing.T) {
	var chatCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"models": []map[string]any{{
					"name": "thinker:latest",
					"loaded_metadata": map[string]any{
						"supports_thinking": true,
						"supports_tools":    true,
						"has_chat_template": true,
						"num_ctx":           4096,
					},
				}},
			})
		case "/api/chat":
			chatCalls++
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			think, _ := req["think"].(bool)
			msg := map[string]any{"content": "ping"}
			if think {
				msg["thinking"] = "brief"
			}
			if tools, ok := req["tools"]; ok && tools != nil {
				msg = map[string]any{
					"content": "",
					"tool_calls": []map[string]any{{
						"function": map[string]any{"name": "get_time"},
					}},
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model":   "thinker:latest",
				"message": msg,
				"done":    true,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	loaded, err := doctorFetchLoadedModels(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != 1 || !loaded[0].SupportsThinking {
		t.Fatalf("loaded=%+v", loaded)
	}

	c01 := doctorCheckReasoningField(srv.URL, loaded[0])
	if c01.Status != "ok" {
		t.Fatalf("reasoning field: %s %s", c01.Status, c01.Detail)
	}
	c12 := doctorCheckThinkRoundtrip(srv.URL, loaded[0])
	if c12.Status != "ok" {
		t.Fatalf("think roundtrip: %s %s", c12.Status, c12.Detail)
	}
	c19 := doctorCheckToolCallShape(srv.URL, loaded[0])
	if c19.Status != "ok" {
		t.Fatalf("tool shape: %s %s", c19.Status, c19.Detail)
	}
	if chatCalls < 3 {
		t.Fatalf("expected >=3 chat calls, got %d", chatCalls)
	}
}

func TestDoctorCheckReasoningFieldMissing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "OK"},
			"done":    true,
		})
	}))
	defer srv.Close()

	c := doctorCheckReasoningField(srv.URL, doctorLoadedModel{
		Name:             "thinker:latest",
		SupportsThinking: true,
	})
	if c.Status != "warn" {
		t.Fatalf("status=%s want warn (%s)", c.Status, c.Detail)
	}
}

func TestDoctorCheckToolCallProse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{
				"content": "I will call get_time with timezone UTC",
			},
			"done": true,
		})
	}))
	defer srv.Close()

	c := doctorCheckToolCallShape(srv.URL, doctorLoadedModel{
		Name:          "tools:latest",
		SupportsTools: true,
	})
	if c.Status != "warn" {
		t.Fatalf("status=%s want warn (%s)", c.Status, c.Detail)
	}
}
