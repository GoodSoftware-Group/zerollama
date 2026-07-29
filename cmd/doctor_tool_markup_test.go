package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorJudgeToolMarkupFields(t *testing.T) {
	ok := doctorJudgeToolMarkupFields("n", "m", 1, "", "planning")
	if ok.Status != "ok" {
		t.Fatalf("%+v", ok)
	}
	trapped := doctorJudgeToolMarkupFields("n", "m", 0, "", "<think>x<tool_call>{}</tool_call>")
	if trapped.Status != "warn" || !strings.Contains(trapped.Detail, "26") {
		t.Fatalf("%+v", trapped)
	}
	partial := doctorJudgeToolMarkupFields("n", "m", 1, "", "hi <tool_call>x")
	if partial.Status != "warn" || !strings.Contains(partial.Detail, "partial") {
		t.Fatalf("%+v", partial)
	}
}

func TestDoctorCheckToolMarkup(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{
					"content": "",
					"tool_calls": []map[string]any{{
						"id":   "1",
						"type": "function",
						"function": map[string]any{
							"name":      "get_time",
							"arguments": `{"timezone":"Asia/Tokyo"}`,
						},
					}},
				},
			}},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckToolMarkup(srv.URL, doctorLoadedModel{Name: "m", SupportsTools: true, SupportsThinking: true})
	if c.Status != "ok" {
		t.Fatalf("%+v", c)
	}

	skip := doctorCheckToolMarkup("http://127.0.0.1:1", doctorLoadedModel{Name: "m"})
	if skip.Status != "ok" || !strings.Contains(skip.Detail, "skipped") {
		t.Fatalf("%+v", skip)
	}
}
