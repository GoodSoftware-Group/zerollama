package server

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestGGUFMetadataPersistRoundTrip(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	src := filepath.Join("..", "fs", "ggml", "testdata", "llama2-7b.Q4_0.gguf")
	if _, err := os.Stat(src); err != nil {
		t.Skip("testdata gguf missing")
	}
	digest := "sha256:0000000000000000000000000000000000000000000000000000000000000001"
	blob, err := manifestTestBlobPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(blob), 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(blob, data, 0o644); err != nil {
		t.Fatal(err)
	}

	md1, err := readGGUFMetadata(digest)
	if err != nil {
		t.Fatal(err)
	}
	if !md1.Valid("general.architecture") {
		t.Fatal("expected architecture in extracted metadata")
	}
	path, err := ggufMetadataPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal("metadata file not written", err)
	}
	md2, err := readGGUFMetadata(digest)
	if err != nil {
		t.Fatal(err)
	}
	if md1.String("general.architecture") != md2.String("general.architecture") {
		t.Fatalf("cache miss: %q vs %q", md1.String("general.architecture"), md2.String("general.architecture"))
	}
	_ = envconfig.Models()
}

func manifestTestBlobPath(digest string) (string, error) {
	return filepath.Join(envconfig.Models(), "blobs", digest), nil
}
