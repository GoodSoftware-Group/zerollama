package reporoots

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSearchRootsIncludesCWD(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	roots := SearchRoots()
	found := false
	for _, root := range roots {
		if root == wd {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("SearchRoots() = %v, want cwd %q", roots, wd)
	}
}

func TestSearchRootsWithEnv(t *testing.T) {
	t.Setenv("ZEROLLAMA_REPO", "")
	dir := t.TempDir()
	t.Setenv("ZEROLLAMA_REPO", dir)
	roots := SearchRootsWithEnv("ZEROLLAMA_REPO")
	found := false
	for _, root := range roots {
		if root == filepath.Clean(dir) {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("SearchRootsWithEnv() = %v, want %q", roots, dir)
	}
}
