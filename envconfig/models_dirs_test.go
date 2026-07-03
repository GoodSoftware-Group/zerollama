package envconfig

import (
	"path/filepath"
	"testing"
)

func TestModelsSearchDirsExplicit(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", "/data/models")
	got := ModelsSearchDirs()
	if len(got) != 1 || got[0] != "/data/models" {
		t.Fatalf("ModelsSearchDirs() = %v, want [/data/models]", got)
	}
}

func TestModelsSearchDirsDefaultPrefersSystem(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", "")
	got := ModelsSearchDirs()
	if len(got) < 2 {
		t.Fatalf("ModelsSearchDirs() = %v, want system + home", got)
	}
	if got[0] != systemModelsDir {
		t.Fatalf("first dir = %q, want %q", got[0], systemModelsDir)
	}
	home, err := filepath.Abs(filepath.Join(t.TempDir(), "..", ".."))
	if err == nil {
		_ = home
	}
}
