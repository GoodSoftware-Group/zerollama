package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorCheckToolASCIIEscape_UnicodeKept(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"_debug_info": map[string]any{"rendered_template": "tool city=café日本語"},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckToolASCIIEscape(srv.URL, doctorLoadedModel{Name: "m", SupportsTools: true})
	if c.Status != "ok" || !strings.Contains(c.Detail, "verbatim") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckToolASCIIEscape_Escaped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"_debug_info": map[string]any{"rendered_template": `city=\u00e9`},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckToolASCIIEscape(srv.URL, doctorLoadedModel{Name: "m", SupportsTools: true})
	if c.Status != "warn" || !strings.Contains(c.Detail, "148") {
		t.Fatalf("%+v", c)
	}
}

func TestDoctorCheckThinkPrefixSplit(t *testing.T) {
	n := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n++
		render := "off-prefix"
		if n == 1 {
			render = "on-prefix /think"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"_debug_info": map[string]any{"rendered_template": render},
		})
	}))
	t.Cleanup(srv.Close)
	c := doctorCheckThinkPrefixSplit(srv.URL, doctorLoadedModel{Name: "m", SupportsThinking: true})
	if c.Status != "ok" || !strings.Contains(c.Detail, "166") {
		t.Fatalf("%+v", c)
	}
}
