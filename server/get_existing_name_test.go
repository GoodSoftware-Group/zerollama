package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// getExistingName must not borrow a tag from an unrelated manifest.
func TestGetExistingNameNoCrossModelTagBorrow(t *testing.T) {
	gin.SetMode(gin.TestMode)
	modelsRoot := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsRoot)

	var s Server
	_, digest := createBinFile(t, nil, nil)
	createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "driaforall/tiny-agent-a:latest",
		Files: map[string]string{"test.gguf": digest},
	})
	createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "library/other:3B",
		Files: map[string]string{"test.gguf": digest},
	})

	got, err := getExistingName(model.ParseName("driaforall/tiny-agent-a:3B"))
	if err != nil {
		t.Fatal(err)
	}
	if got.Tag != "3B" {
		t.Fatalf("tag=%q want 3B (unchanged)", got.Tag)
	}

	tagPath := filepath.Join(modelsRoot, "manifests", "registry.ollama.ai", "driaforall", "tiny-agent-a", "3B")
	if _, err := os.Stat(tagPath); err == nil {
		t.Fatalf("unexpected manifest at %s", tagPath)
	}
}
