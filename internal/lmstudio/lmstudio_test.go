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

func TestMatchDir_shardedGGUF(t *testing.T) {
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
	got, ok := MatchDir(n)
	if !ok {
		t.Fatal("expected sharded GGUF layout to match")
	}
	if got != modelDir {
		t.Fatalf("got %q want %q", got, modelDir)
	}
}

func TestMatchDir_safetensors(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "Hermes-4-70B-MLX-8bit")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"model-00001-of-00002.safetensors",
		"model-00002-of-00002.safetensors",
		"config.json",
	} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	n := model.ParseName("lmstudio-community/hermes-4-70b:8bit")
	got, ok := MatchDir(n)
	if !ok {
		t.Fatal("expected safetensors layout to match suggested name")
	}
	if got != modelDir {
		t.Fatalf("got %q want %q", got, modelDir)
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

func TestSuggestedName(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "lmstudio-community", "gemma-4-31B-it-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "gemma-4-31B-it-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := SuggestedName(root, modelDir)
	want := "lmstudio-community/gemma-4-31b-it:q8_0"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestList(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "Qwen", "Qwen3.6-27B-GGUF")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "Qwen3.6-27B-Q8_0.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	entries := List()
	if len(entries) != 1 {
		t.Fatalf("len=%d want 1", len(entries))
	}
	if entries[0].Format != "gguf" {
		t.Fatalf("format=%q want gguf", entries[0].Format)
	}
	if entries[0].Name != "qwen/qwen3.6-27b:q8_0" {
		t.Fatalf("name=%q", entries[0].Name)
	}
}

func TestMatchDir_multiQuantGGUF(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "driaforall", "Tiny-Agent-a-0.5B")
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"dria-agent-a-0.5b.Q4_K_M.gguf",
		"dria-agent-a-0.5b.Q8_0.gguf",
	} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("GGUF"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	t.Setenv("OLLAMA_LMSTUDIO_MODELS", root)

	entries := List()
	if len(entries) != 2 {
		t.Fatalf("len=%d want 2 quant variants", len(entries))
	}

	n := model.ParseName("driaforall/tiny-agent-a-0.5b:q8_0")
	dir, weight, ok := MatchSelection(n)
	if !ok || dir != modelDir || weight != "dria-agent-a-0.5b.Q8_0.gguf" {
		t.Fatalf("match=%v dir=%q weight=%q", ok, dir, weight)
	}
}

func TestListRealLMStudioCache(t *testing.T) {
	cache := "/Users/user1/.lmstudio/models"
	if st, err := os.Stat(cache); err != nil || !st.IsDir() {
		t.Skip("real LM Studio cache not present")
	}
	t.Setenv("OLLAMA_LMSTUDIO_MODELS", cache)

	entries := List()
	if len(entries) < 12 {
		t.Fatalf("expected at least 12 models, got %d", len(entries))
	}
	formats := map[string]int{}
	for _, e := range entries {
		formats[e.Format]++
		if e.Name == "" || e.Dir == "" {
			t.Fatalf("incomplete entry: %+v", e)
		}
	}
	if formats["gguf"] == 0 {
		t.Fatal("expected at least one gguf model")
	}
	if formats["safetensors"] == 0 {
		t.Fatal("expected at least one safetensors model")
	}
}
