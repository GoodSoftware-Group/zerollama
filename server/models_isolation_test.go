package server

import (
	"os"
	"path/filepath"
	"testing"
)

func TestIsTestModelsPath(t *testing.T) {
	tmp := t.TempDir()
	if !isTestModelsPath(tmp) {
		t.Fatalf("expected temp dir %q to be accepted", tmp)
	}
	nested := filepath.Join(tmp, "blobs")
	if !isTestModelsPath(nested) {
		t.Fatalf("expected nested temp path %q to be accepted", nested)
	}
	if isTestModelsPath("") {
		t.Fatal("empty path should be rejected")
	}
	if isTestModelsPath("/mnt/ollama_img/models") {
		t.Fatal("production path should be rejected")
	}
	if isTestModelsPath(filepath.Join(os.TempDir(), "..", "etc")) {
		t.Fatal("path escaping temp via .. should be rejected")
	}
}
