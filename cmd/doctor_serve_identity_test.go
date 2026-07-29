package cmd

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDoctorHostPortFromBase(t *testing.T) {
	got := doctorHostPortFromBase("http://127.0.0.1:11435")
	if got != "127.0.0.1:11435" {
		t.Fatalf("got %q", got)
	}
	got = doctorHostPortFromBase("http://localhost:11434")
	if got != "127.0.0.1:11434" {
		t.Fatalf("localhost normalize: %q", got)
	}
}

func TestDoctorCheckContextCeilings(t *testing.T) {
	ok := doctorCheckContextCeilings(doctorLoadedModel{Name: "m", NumCtx: 4096, TrainCtx: 4096})
	if ok.Status != "ok" {
		t.Fatalf("aligned: %+v", ok)
	}
	warn := doctorCheckContextCeilings(doctorLoadedModel{Name: "m", NumCtx: 131072, TrainCtx: 32768})
	if warn.Status != "warn" || !strings.Contains(warn.Detail, "61") {
		t.Fatalf("diverge: %+v", warn)
	}
	over := doctorCheckContextCeilings(doctorLoadedModel{Name: "m", NumCtx: 65536, TrainCtx: 32768})
	if over.Status != "warn" {
		t.Fatalf("served>trained: %+v", over)
	}
}

func TestDoctorFetchVersion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/version" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"version":"0.30.11"}`))
	}))
	t.Cleanup(srv.Close)
	if got := doctorFetchVersion(srv.URL); got != "0.30.11" {
		t.Fatalf("version=%q", got)
	}
}

func TestDoctorCheckServeIdentity_Unreachable(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "127.0.0.1:1")
	c := doctorCheckServeIdentity()
	if c.Status != "warn" {
		t.Fatalf("want warn when nothing listens: %+v", c)
	}
	if !strings.Contains(c.Name, "53") {
		t.Fatalf("name=%q", c.Name)
	}
}
