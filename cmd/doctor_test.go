package cmd

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDoctorCheckGo(t *testing.T) {
	c := doctorCheckGo()
	if c.Status != "ok" {
		t.Fatalf("status=%q", c.Status)
	}
	if c.Detail == "" {
		t.Fatal("expected platform detail")
	}
}

func TestDoctorPickTextGGUFSnippetParses(t *testing.T) {
	s := doctorPickTextGGUFSnippet()
	if !strings.Contains(s, "projector") {
		t.Fatal("expected projector skip in picker snippet")
	}
	if !strings.Contains(s, "gemma") {
		t.Fatal("expected gemma skip in picker snippet")
	}
}

func TestRunDoctorChecksNonDarwin(t *testing.T) {
	checks := runDoctorChecks(".")
	if len(checks) < 4 {
		t.Fatalf("expected multiple checks, got %d", len(checks))
	}
}

func TestDoctorEvaluateSidecarHealthAppleSiliconOk(t *testing.T) {
	c := doctorEvaluateSidecarHealth(map[string]any{
		"autoconfig":              map[string]any{"pick": "apple_silicon"},
		"llama_backend":           "inprocess",
		"llama_backend_source":    "config",
		"llama_backend_requested": "inprocess",
		"llama_backend_fallback":  false,
		"vram_probe_effective":    "metal-unified",
	})
	if c.Status != "ok" {
		t.Fatalf("status=%q detail=%q fix=%q", c.Status, c.Detail, c.FixHint)
	}
}

func TestDoctorEvaluateSidecarHealthFallbackWarns(t *testing.T) {
	c := doctorEvaluateSidecarHealth(map[string]any{
		"autoconfig":              map[string]any{"pick": "apple_silicon"},
		"llama_backend":           "subprocess",
		"llama_backend_source":    "config",
		"llama_backend_requested": "inprocess",
		"llama_backend_fallback":  true,
	})
	if c.Status != "warn" {
		t.Fatalf("status=%q", c.Status)
	}
	if !strings.Contains(c.FixHint, "text-only GGUF") {
		t.Fatalf("fix=%q", c.FixHint)
	}
}

func TestDoctorEvaluateSidecarHealthEnvOverrideWarns(t *testing.T) {
	c := doctorEvaluateSidecarHealth(map[string]any{
		"autoconfig":           map[string]any{"pick": "apple_silicon"},
		"llama_backend":        "subprocess",
		"llama_backend_source": "env",
	})
	if c.Status != "warn" {
		t.Fatalf("status=%q", c.Status)
	}
	if !strings.Contains(c.FixHint, "unset ZEROLLAMA_RUNTIME_LLAMA_BACKEND") {
		t.Fatalf("fix=%q", c.FixHint)
	}
}

func TestDoctorOllamaHostDefault(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")
	if got := doctorOllamaHost(); got != "http://127.0.0.1:11434" {
		t.Fatalf("default host=%q", got)
	}
}

func TestDoctorCheckServeModesNoServers(t *testing.T) {
	c := doctorCheckServeModes()
	if c.Name != "serve mode" {
		t.Fatalf("name=%q", c.Name)
	}
	if c.Status != "warn" && c.Status != "ok" {
		t.Fatalf("status=%q", c.Status)
	}
}

func TestBuildDoctorReportJSON(t *testing.T) {
	report := buildDoctorReport(".")
	if len(report.Checks) == 0 {
		t.Fatal("expected checks")
	}
	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"checks"`) {
		t.Fatalf("json=%s", string(b))
	}
}
