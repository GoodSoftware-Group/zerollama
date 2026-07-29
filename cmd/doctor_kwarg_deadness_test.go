package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckKwargDeadness(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		if kwargs, _ := req["chat_template_kwargs"].(map[string]any); kwargs != nil {
			if _, ok := kwargs["bogus_kwarg_zzq"]; ok {
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]any{
					"error": "unknown field: chat_template_kwargs.bogus_kwarg_zzq",
				})
				return
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "OK"},
			"done":    true,
		})
	}))
	t.Cleanup(srv.Close)

	c := doctorCheckKwargDeadness(srv.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "ok" || !strings.Contains(c.Detail, "rejected") {
		t.Fatalf("%+v", c)
	}

	open := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"message": map[string]any{"content": "OK"},
			"done":    true,
		})
	}))
	t.Cleanup(open.Close)
	c = doctorCheckKwargDeadness(open.URL, doctorLoadedModel{Name: "m"})
	if c.Status != "warn" || !strings.Contains(c.Detail, "07") {
		t.Fatalf("want accept warn: %+v", c)
	}
}
