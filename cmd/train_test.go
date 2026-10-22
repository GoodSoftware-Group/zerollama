package cmd

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestResolveTrainRecipePythonPrefersExplicit(t *testing.T) {
	got, err := resolveTrainRecipePython("/nonexistent", "/usr/bin/custom-python")
	if err != nil {
		t.Fatal(err)
	}
	if got != "/usr/bin/custom-python" {
		t.Fatalf("got %q", got)
	}
}

func TestResolveTrainRecipePythonVenv(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix venv layout")
	}
	root := t.TempDir()
	bin := filepath.Join(root, ".venv-training", "bin")
	if err := os.MkdirAll(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	py := filepath.Join(bin, "python")
	if err := os.WriteFile(py, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TRAINING_PYTHON", "")
	got, err := resolveTrainRecipePython(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if got != py {
		t.Fatalf("got %q want %q", got, py)
	}
}

func TestRunTrainRecipeMissingScript(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "training.py"), []byte("#"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ZEROLLAMA_REPO", root)
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", root)
	err := runTrainRecipe(filepath.Join(root, "in.csv"), trainRecipeOpts{
		Out:       filepath.Join(root, "out.jsonl"),
		Format:    "auto",
		PromptCol: "prompt", ResponseCol: "response",
		ChunkChars: 1200,
		PythonBin:  "python3",
	})
	if err == nil {
		t.Fatal("expected error for missing training_recipes.py")
	}
}
