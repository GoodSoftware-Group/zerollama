package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestMergeLMStudioModels_Disabled(t *testing.T) {
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "false")
	local := []api.ListModelResponse{{Model: "local:latest", Name: "local:latest"}}
	out := mergeLMStudioModels(local)
	if len(out) != 1 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestMergeLMStudioModels_AppendsFromCache(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")

	local := []api.ListModelResponse{{Model: "local:latest", Name: "local:latest"}}
	out := mergeLMStudioModels(local)
	if len(out) != 2 {
		t.Fatalf("len=%d want 2", len(out))
	}
	if out[1].RemoteHost != "lmstudio" {
		t.Fatalf("remote_host=%q", out[1].RemoteHost)
	}
	if out[1].Model != "lmstudio-community/gemma-4-31b-it:q8_0" {
		t.Fatalf("model=%q", out[1].Model)
	}
}

func TestMergeLMStudioModels_DedupesLocal(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")

	local := []api.ListModelResponse{{Model: "lmstudio-community/gemma-4-31b-it:q8_0", Name: "lmstudio-community/gemma-4-31b-it:q8_0"}}
	out := mergeLMStudioModels(local)
	if len(out) != 1 {
		t.Fatalf("len=%d want 1 (dedupe)", len(out))
	}
}
