package trainingworker

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveTrainingRepoRoot_env(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "training.py"), []byte("# test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", dir)
	t.Setenv("ZEROLLAMA_REPO", "")
	got, err := resolveTrainingRepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != dir {
		t.Fatalf("got %q want %q", got, dir)
	}
}

func TestResolveTrainingRepoRoot_invalidEnv(t *testing.T) {
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", "/definitely/not/a/zerollama/repo")
	t.Setenv("ZEROLLAMA_REPO", "")
	_, err := resolveTrainingRepoRoot()
	if err == nil {
		t.Fatal("expected error for invalid OLLAMA_TRAINING_PYTHONPATH")
	}
	if !strings.Contains(err.Error(), "training.py") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestResolveTrainingRepoRoot_homeZerollama(t *testing.T) {
	home := t.TempDir()
	repo := filepath.Join(home, "zerollama")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "training.py"), []byte("# test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", "")
	t.Setenv("ZEROLLAMA_REPO", "")
	t.Chdir(t.TempDir())
	got, err := resolveTrainingRepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != repo {
		t.Fatalf("got %q want %q", got, repo)
	}
}

func TestResolveTrainingRepoRoot_cwdBeforeHome(t *testing.T) {
	home := t.TempDir()
	homeRepo := filepath.Join(home, "zerollama")
	if err := os.MkdirAll(homeRepo, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(homeRepo, "training.py"), []byte("# home\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cwdRepo := t.TempDir()
	if err := os.WriteFile(filepath.Join(cwdRepo, "training.py"), []byte("# cwd\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", "")
	t.Setenv("ZEROLLAMA_REPO", "")
	t.Chdir(cwdRepo)
	got, err := resolveTrainingRepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != cwdRepo {
		t.Fatalf("got %q want cwd repo %q (home repo must not win)", got, cwdRepo)
	}
}

func TestResolveTrainingRepoRoot_missing(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("OLLAMA_TRAINING_PYTHONPATH", "")
	t.Setenv("ZEROLLAMA_REPO", "")
	t.Chdir(t.TempDir())
	got, err := resolveTrainingRepoRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Fatalf("expected empty, got %q", got)
	}
}
