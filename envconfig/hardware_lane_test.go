package envconfig

import (
	"os"
	"path/filepath"
	"testing"
)

func TestApplyHardwareLaneDefaultsFromYAML(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "dual_4090.yaml")
	if err := os.WriteFile(cfg, []byte(`device_count: 2
serve:
  max_loaded_models: 1
  keep_alive: 5m
  num_parallel: 1
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_RUNTIME_CONFIG", cfg)
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "")
	t.Setenv("OLLAMA_KEEP_ALIVE", "")
	t.Setenv("OLLAMA_NUM_PARALLEL", "")

	ApplyHardwareLaneDefaults()

	if got := os.Getenv("OLLAMA_MAX_LOADED_MODELS"); got != "1" {
		t.Fatalf("OLLAMA_MAX_LOADED_MODELS=%q want 1", got)
	}
	if got := os.Getenv("OLLAMA_KEEP_ALIVE"); got != "5m" {
		t.Fatalf("OLLAMA_KEEP_ALIVE=%q want 5m", got)
	}
	if got := os.Getenv("OLLAMA_NUM_PARALLEL"); got != "1" {
		t.Fatalf("OLLAMA_NUM_PARALLEL=%q want 1", got)
	}
}

func TestApplyHardwareLaneVRAMDefaultsFromYAML(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "dual_4090.yaml")
	if err := os.WriteFile(cfg, []byte(`device_count: 2
vram:
  clamp_num_ctx: auto
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_RUNTIME_CONFIG", cfg)
	t.Setenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX", "")

	ApplyHardwareLaneDefaults()

	if got := os.Getenv("ZEROLLAMA_GGML_CLAMP_NUM_CTX"); got != "auto" {
		t.Fatalf("ZEROLLAMA_GGML_CLAMP_NUM_CTX=%q want auto", got)
	}
}

func TestApplyHardwareLaneDefaultsRespectsExplicitEnv(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "dual_4090.yaml")
	if err := os.WriteFile(cfg, []byte(`serve:
  max_loaded_models: 1
`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_RUNTIME_CONFIG", cfg)
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "3")

	ApplyHardwareLaneDefaults()

	if got := os.Getenv("OLLAMA_MAX_LOADED_MODELS"); got != "3" {
		t.Fatalf("explicit env overridden: %q", got)
	}
}

func TestResolveAutoRuntimeConfigPathPicksSingleGPU(t *testing.T) {
	repo := t.TempDir()
	configs := filepath.Join(repo, "runtime", "configs")
	if err := os.MkdirAll(configs, 0o755); err != nil {
		t.Fatal(err)
	}
	single := filepath.Join(configs, "single_gpu.yaml")
	dual := filepath.Join(configs, "dual_4090.yaml")
	if err := os.WriteFile(single, []byte("device_count: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dual, []byte("device_count: 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_RUNTIME_CONFIG", "")
	t.Setenv("ZEROLLAMA_AUTO_CONFIG", "1")

	prev := detectVisibleGPUCount
	t.Cleanup(func() { detectVisibleGPUCount = prev })
	detectVisibleGPUCount = func() (int, bool) { return 1, true }

	got := resolveAutoRuntimeConfigPath()
	if got != single {
		t.Fatalf("got %q want %q", got, single)
	}
}

func TestResolveAutoRuntimeConfigPathPicksDual(t *testing.T) {
	repo := t.TempDir()
	configs := filepath.Join(repo, "runtime", "configs")
	if err := os.MkdirAll(configs, 0o755); err != nil {
		t.Fatal(err)
	}
	single := filepath.Join(configs, "single_gpu.yaml")
	dual := filepath.Join(configs, "dual_4090.yaml")
	if err := os.WriteFile(single, []byte("device_count: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dual, []byte("device_count: 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_RUNTIME_CONFIG", "")
	t.Setenv("ZEROLLAMA_AUTO_CONFIG", "1")

	prev := detectVisibleGPUCount
	t.Cleanup(func() { detectVisibleGPUCount = prev })
	detectVisibleGPUCount = func() (int, bool) { return 2, true }

	got := resolveAutoRuntimeConfigPath()
	if got != dual {
		t.Fatalf("got %q want %q", got, dual)
	}
}

func TestResolveAutoRuntimeConfigPathProbeFailPrefersSingle(t *testing.T) {
	repo := t.TempDir()
	configs := filepath.Join(repo, "runtime", "configs")
	if err := os.MkdirAll(configs, 0o755); err != nil {
		t.Fatal(err)
	}
	single := filepath.Join(configs, "single_gpu.yaml")
	dual := filepath.Join(configs, "dual_4090.yaml")
	if err := os.WriteFile(single, []byte("device_count: 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dual, []byte("device_count: 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("ZEROLLAMA_REPO", repo)
	t.Setenv("ZEROLLAMA_RUNTIME_CONFIG", "")
	t.Setenv("ZEROLLAMA_AUTO_CONFIG", "1")

	prev := detectVisibleGPUCount
	t.Cleanup(func() { detectVisibleGPUCount = prev })
	detectVisibleGPUCount = func() (int, bool) { return 0, false }

	got := resolveAutoRuntimeConfigPath()
	if got != single {
		t.Fatalf("got %q want %q", got, single)
	}
}
