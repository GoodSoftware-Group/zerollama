package modelhealth

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestCheckNameOKInExplicitModelsDir(t *testing.T) {
	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)

	n := model.ParseName("registry.ollama.ai/library/tiny-agent:q8")
	cfgDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000001"
	layerDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000002"
	for _, digest := range []string{cfgDigest, layerDigest} {
		path, err := manifest.BlobsPathIn(modelsDir, digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	p := filepath.Join(modelsDir, "manifests", n.Filepath())
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(n, manifest.Layer{
		MediaType: "application/vnd.ollama.image.model",
		Digest:    cfgDigest,
		Size:      1,
	}, []manifest.Layer{{
		MediaType: "application/vnd.ollama.image.model",
		Digest:    layerDigest,
		Size:      1,
	}}); err != nil {
		t.Fatal(err)
	}

	report, err := CheckName("tiny-agent:q8")
	if err != nil {
		t.Fatal(err)
	}
	if report.Status != StatusOK {
		t.Fatalf("status = %s, want ok (%s)", report.Status, report.Detail)
	}
}

func TestCheckNameSearchesServiceStoreWhenUnset(t *testing.T) {
	if _, err := os.Stat(systemModelsDir); err != nil {
		t.Skip("service models dir not present")
	}
	t.Setenv("OLLAMA_MODELS", "")

	report, err := CheckName("gemma4:e2b-it-qat")
	if err != nil {
		t.Fatal(err)
	}
	if report.Status != StatusOK {
		t.Skipf("gemma4:e2b-it-qat not in service store: %s %s", report.Status, report.Detail)
	}
}

const systemModelsDir = "/usr/share/ollama/.ollama/models"
