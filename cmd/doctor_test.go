package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
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

func TestDoctorCheckMacCGO(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin only")
	}
	repo := doctorRepoRoot()
	c := doctorCheckMacCGO(repo)
	if c.Name != "mac cgo build" {
		t.Fatalf("name=%q", c.Name)
	}
	if c.Status == "fail" && !strings.Contains(c.FixHint, "build_zerollama_mac") {
		t.Fatalf("unexpected fix=%q", c.FixHint)
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

func TestDoctorIsStaleFlatMLXPath(t *testing.T) {
	repo := t.TempDir()
	build := filepath.Join(repo, "build")
	flat := filepath.Join(build, "lib", "ollama", "libmlxc.dylib")
	variant := filepath.Join(build, "metal-v3", "lib", "ollama", "libmlxc.dylib")
	if err := os.MkdirAll(filepath.Dir(flat), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(variant), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(flat, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(variant, []byte("y"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !doctorIsStaleFlatMLXPath(repo, flat) {
		t.Fatal("expected stale flat detection")
	}
	if doctorIsStaleFlatMLXPath(repo, variant) {
		t.Fatal("variant path should not be stale flat")
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

func TestDoctorEnsureLlamaCppSiblingScript(t *testing.T) {
	repo := doctorRepoRoot()
	script := filepath.Join(repo, "scripts", "ensure_llama_cpp_sibling.sh")
	if _, err := os.Stat(script); err != nil {
		t.Fatalf("missing ensure script: %v", err)
	}
}
