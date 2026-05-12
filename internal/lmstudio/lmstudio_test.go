package lmstudio

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/types/model"
)

func TestMatchDir_gemma(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("gemma4:31b")
	got, ok := MatchDir(n)
	if !ok {
		t.Fatal("expected match")
	}
	if got != modelDir {
		t.Fatalf("got %q want %q", got, modelDir)
	}
}

func TestMatchDir_shardedSkipped(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "Qwen", "Qwen2.5-Coder-32B-Instruct-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf",
		"qwen2.5-coder-32b-instruct-fp16-00002-of-00009.gguf",
	} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("qwen2.5-coder:32b")
	_, ok := MatchDir(n)
	if ok {
		t.Fatal("expected sharded layout to be skipped")
	}
}

func TestMatchDir_ambiguous(t *testing.T) {
	root := t.TempDir()
	for _, sub := range []string{"a/gemma-4-31B-it-GGUF", "b/gemma-4-31B-it-GGUF"} {
		modelDir := filepath.Join(root, sub)
		if err := os.MkdirAll(modelDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(modelDir, "x.gguf"), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("gemma4:31b")
	_, ok := MatchDir(n)
	if ok {
		t.Fatal("expected ambiguous match to be rejected")
	}
}

func TestDirLooksLikeLMStudioModel_mmproj(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mmproj-model.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !dirLooksLikeLMStudioModel(dir) {
		t.Fatal("expected model + mmproj to be accepted")
	}
}
