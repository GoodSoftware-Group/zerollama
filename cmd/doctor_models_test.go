package cmd

import (
	"testing"
)

func TestBuildDoctorModelsReportEmpty(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	report := buildDoctorModelsReport()
	if !report.OK {
		t.Fatalf("expected ok for empty models dir, got failures=%d", report.Failures)
	}
}

func TestBenchPreflightSkip(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	detail, skip := benchPreflightSkip("missing:latest", false)
	if !skip {
		t.Fatal("expected skip for missing manifest")
	}
	if detail == "" {
		t.Fatal("expected detail")
	}
}
