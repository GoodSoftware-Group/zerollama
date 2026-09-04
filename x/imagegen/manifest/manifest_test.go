package manifest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTotalTensorSize(t *testing.T) {
	m := &ModelManifest{
		Manifest: &Manifest{
			Layers: []ManifestLayer{
				{MediaType: "application/vnd.ollama.image.tensor", Size: 1000},
				{MediaType: "application/vnd.ollama.image.tensor", Size: 2000},
				{MediaType: "application/vnd.ollama.image.json", Size: 500}, // not a tensor
				{MediaType: "application/vnd.ollama.image.tensor", Size: 3000},
			},
		},
	}

	got := m.TotalTensorSize()
	want := int64(6000)
	if got != want {
		t.Errorf("TotalTensorSize() = %d, want %d", got, want)
	}
}

func TestBlobPathFileDigest(t *testing.T) {
	m := &ModelManifest{BlobDir: "/unused"}
	got := m.BlobPath("file:/tmp/model.safetensors")
	if got != "/tmp/model.safetensors" {
		t.Fatalf("BlobPath file digest = %q", got)
	}
	got = m.BlobPath("sha256:abc")
	want := filepath.Join("/unused", "sha256-abc")
	if got != want {
		t.Fatalf("BlobPath sha = %q, want %q", got, want)
	}
}

func TestSourceDirTensorLayersAndReadConfig(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"architectures":["Test"]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	shard := filepath.Join(dir, "model-00001-of-00002.safetensors")
	if err := os.WriteFile(shard, []byte("dummy-weights"), 0o644); err != nil {
		t.Fatal(err)
	}

	m := &ModelManifest{
		Manifest:  &Manifest{},
		SourceDir: dir,
	}
	layers := m.GetTensorLayers("")
	if len(layers) != 1 {
		t.Fatalf("GetTensorLayers = %d, want 1 (index.json skipped)", len(layers))
	}
	if layers[0].Digest != "file:"+shard {
		t.Fatalf("digest = %q", layers[0].Digest)
	}
	if m.BlobPath(layers[0].Digest) != shard {
		t.Fatalf("BlobPath = %q, want %q", m.BlobPath(layers[0].Digest), shard)
	}
	if !m.HasTensorLayers() {
		t.Fatal("HasTensorLayers = false")
	}
	if m.TotalTensorSize() != int64(len("dummy-weights")) {
		t.Fatalf("TotalTensorSize = %d", m.TotalTensorSize())
	}
	cfg, err := m.ReadConfig("config.json")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(cfg), "Test") {
		t.Fatalf("config = %s", cfg)
	}
	if _, err := m.ReadConfig("../escape.json"); err == nil {
		t.Fatal("expected path escape to fail")
	}
}

func TestTotalTensorSizeEmpty(t *testing.T) {
	m := &ModelManifest{
		Manifest: &Manifest{
			Layers: []ManifestLayer{},
		},
	}

	if got := m.TotalTensorSize(); got != 0 {
		t.Errorf("TotalTensorSize() = %d, want 0", got)
	}
}

func TestManifestAndBlobDirsRespectOLLAMAModels(t *testing.T) {
	modelsDir := filepath.Join(t.TempDir(), "models")

	// Simulate packaged/systemd environment
	t.Setenv("OLLAMA_MODELS", modelsDir)
	t.Setenv("HOME", "/usr/share/ollama")

	// Manifest dir must respect OLLAMA_MODELS
	wantManifest := filepath.Join(modelsDir, "manifests")
	if got := DefaultManifestDir(); got != wantManifest {
		t.Fatalf("DefaultManifestDir() = %q, want %q", got, wantManifest)
	}

	// Blob dir must respect OLLAMA_MODELS
	wantBlobs := filepath.Join(modelsDir, "blobs")
	if got := DefaultBlobDir(); got != wantBlobs {
		t.Fatalf("DefaultBlobDir() = %q, want %q", got, wantBlobs)
	}
}
