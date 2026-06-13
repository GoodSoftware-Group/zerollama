package server

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/lmstudio"
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

func TestMergeLMStudioModels_SkipsSafetensorsWithoutDisk(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "Hermes-4-70B-MLX-8bit")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "model-00001-of-00002.safetensors"} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	restore := lmstudio.SetModelsFreeBytesHook(func(string) (int64, error) { return 0, nil })
	t.Cleanup(restore)

	local := []api.ListModelResponse{{Model: "local:latest", Name: "local:latest"}}
	out := mergeLMStudioModels(local)
	if len(out) != 1 {
		t.Fatalf("len=%d want 1 (safetensors hidden when no disk)", len(out))
	}
}

func TestMergeLMStudioModels_ListAllIgnoresDisk(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "Hermes-4-70B-MLX-8bit")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "model-00001-of-00002.safetensors"} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")
	t.Setenv("OLLAMA_LMSTUDIO_LIST_ALL", "true")
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	restore := lmstudio.SetModelsFreeBytesHook(func(string) (int64, error) { return 0, nil })
	t.Cleanup(restore)

	out := mergeLMStudioModels(nil)
	if len(out) != 1 {
		t.Fatalf("len=%d want 1 (LIST_ALL shows safetensors despite no disk)", len(out))
	}
}

func TestMergeLMStudioModels_ShowsLegacySafetensorsWithoutDisk(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "legacy-safetensors")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	restore := lmstudio.SetModelsFreeBytesHook(func(string) (int64, error) { return 0, nil })
	t.Cleanup(restore)

	out := mergeLMStudioModels(nil)
	if len(out) != 1 {
		t.Fatalf("len=%d want 1 (legacy safetensors without config.json still listed)", len(out))
	}
}

func TestMergeLMStudioModels_SkipsMLXWhenDiskCheckErrors(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "Hermes-4-70B-MLX-8bit")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config.json", "model-00001-of-00002.safetensors"} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)
	t.Setenv("OLLAMA_LMSTUDIO_IMPORT", "true")
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	restore := lmstudio.SetModelsFreeBytesHook(func(string) (int64, error) {
		return 0, fmt.Errorf("statfs failed")
	})
	t.Cleanup(restore)

	out := mergeLMStudioModels(nil)
	if len(out) != 0 {
		t.Fatalf("len=%d want 0 (MLX hidden when disk check errors)", len(out))
	}
}
