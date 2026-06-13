package server

import (
	"os"
	"path/filepath"
	"testing"

	xcreate "github.com/ollama/ollama/x/create"
)

func TestRelativePathsInDir(t *testing.T) {
	dir := "/Users/user1/.lmstudio/models/lmstudio-community/GLM-4.7-Flash-MLX-8bit"
	files := map[string]string{
		filepath.Join(dir, "model-00001-of-00007.safetensors"): "sha256:aaa",
		filepath.Join(dir, "config.json"):                      "sha256:bbb",
	}

	got := relativePathsInDir(dir, files)
	if len(got) != 2 {
		t.Fatalf("len=%d", len(got))
	}
	if got["model-00001-of-00007.safetensors"] != "sha256:aaa" {
		t.Fatalf("safetensors key=%q", got["model-00001-of-00007.safetensors"])
	}
	if got["config.json"] != "sha256:bbb" {
		t.Fatalf("config key=%q", got["config.json"])
	}
}

func TestLMStudioUseNativeSafetensorsImport(t *testing.T) {
	dir := t.TempDir()
	if lmStudioUseNativeSafetensorsImport(dir) {
		t.Fatal("empty dir should not use native safetensors import")
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if lmStudioUseNativeSafetensorsImport(dir) {
		t.Fatal("config only should not use native safetensors import")
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !lmStudioUseNativeSafetensorsImport(dir) {
		t.Fatal("config + safetensors should use native import")
	}
	if !xcreate.IsSafetensorsModelDir(dir) {
		t.Fatal("IsSafetensorsModelDir should match helper")
	}
}
