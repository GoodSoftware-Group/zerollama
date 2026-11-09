package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
)

func TestResolveGGUFSplitUniqueNames(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	blobs := filepath.Join(models, "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	a := filepath.Join(blobs, "sha256-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	b := filepath.Join(blobs, "sha256-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	if err := os.WriteFile(a, []byte("shard-a"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(b, []byte("shard-b"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Same From on every layer is what create-from writes.
	path, _, _, err := resolveGGUFModelPath([]ggufModelLayer{
		{digest: "sha256:aa", from: "library/model:latest", blobPath: a},
		{digest: "sha256:bb", from: "library/model:latest", blobPath: b},
	})
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(path) != "shard-00001-of-00002.gguf" {
		t.Fatalf("first shard %s", path)
	}
	dir := filepath.Dir(path)
	second := filepath.Join(dir, "shard-00002-of-00002.gguf")
	gotA, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	gotB, err := os.ReadFile(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotA) == string(gotB) {
		t.Fatalf("shards collapsed onto one blob: %q", gotA)
	}
}

func TestEnsureBlobKeepsEmptyContent(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	const digest = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
	path, err := ensureBlob(digest, 0)
	if err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(path)
	if err != nil || fi.Size() != 0 {
		t.Fatalf("empty blob stat=%v err=%v", fi, err)
	}
	again, err := ensureBlob(digest, 0)
	if err != nil || again != path {
		t.Fatalf("second load %s %v", again, err)
	}
}

func TestEnsureBlobDropsTruncatedZero(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	digest := "sha256:" + strings.Repeat("ab", 32)
	path, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := ensureBlob(digest, 0); err == nil {
		t.Fatal("truncated blob accepted")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("truncated file kept: %v", err)
	}
}
