package blobaudit

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
)

func TestAudit_dedupeAndOrphans(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	root := envconfig.Models()
	blobs := filepath.Join(root, "blobs")
	manifestFileA := filepath.Join(root, "manifests", "registry.ollama.ai", "library", "a", "latest")
	manifestFileB := filepath.Join(root, "manifests", "registry.ollama.ai", "library", "b", "latest")
	for _, dir := range []string{blobs, filepath.Dir(manifestFileA), filepath.Dir(manifestFileB)} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	digestA := repeatByte('a', 64)
	digestB := repeatByte('b', 64)
	digestC := repeatByte('c', 64)

	writeBlob := func(hex string, data []byte) {
		if err := os.WriteFile(filepath.Join(blobs, "sha256-"+hex), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeBlob(digestA, bytesRepeat('a', 100))
	writeBlob(digestB, bytesRepeat('b', 200))
	writeBlob(digestC, bytesRepeat('c', 50)) // orphan

	mf := `{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:` + digestA + `","size":100},
  "layers": [
    {"mediaType":"application/vnd.ollama.image.tensor","digest":"sha256:` + digestB + `","size":200,"name":"w"}
  ]
}`
	if err := os.WriteFile(manifestFileA, []byte(mf), 0o644); err != nil {
		t.Fatal(err)
	}

	mf2 := `{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:` + digestA + `","size":100},
  "layers": []
}`
	if err := os.WriteFile(manifestFileB, []byte(mf2), 0o644); err != nil {
		t.Fatal(err)
	}

	r, err := Audit()
	if err != nil {
		t.Fatal(err)
	}
	if r.TagCount != 2 {
		t.Fatalf("tags=%d want 2", r.TagCount)
	}
	if r.OrphanFileCount != 1 || r.OrphanBytes != 50 {
		t.Fatalf("orphans=%d bytes=%d", r.OrphanFileCount, r.OrphanBytes)
	}
	if r.SharedDigests != 1 {
		t.Fatalf("shared=%d want 1", r.SharedDigests)
	}
	if r.TotalBytes != 350 {
		t.Fatalf("total=%d want 350", r.TotalBytes)
	}
	if r.ReferencedBytes != 300 {
		t.Fatalf("referenced=%d want 300", r.ReferencedBytes)
	}
	if len(r.Tags) != 2 {
		t.Fatalf("rollups=%d", len(r.Tags))
	}

	out := FormatHuman(r)
	if out == "" {
		t.Fatal("empty human output")
	}

	enc, err := json.Marshal(r)
	if err != nil || len(enc) == 0 {
		t.Fatalf("json: %v len=%d", err, len(enc))
	}
}

func TestCanonicalDigest(t *testing.T) {
	got := canonicalDigest("sha256-abc")
	if got != "sha256:abc" {
		t.Fatalf("got %q", got)
	}
}

func repeatByte(b byte, n int) string {
	out := make([]byte, n)
	for i := range out {
		out[i] = b
	}
	return string(out)
}

func bytesRepeat(b byte, n int) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = b
	}
	return out
}

func TestManifestLayerConstants(t *testing.T) {
	_ = manifest.MediaTypeImageTensor
}
