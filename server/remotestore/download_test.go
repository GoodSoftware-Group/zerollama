package remotestore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAtomicKeepPartialOnMismatch(t *testing.T) {
	dir := t.TempDir()
	partial := filepath.Join(dir, "blob.partial")
	final := filepath.Join(dir, "blob")
	payload := []byte("not-the-right-bytes")
	wrong := "sha256-" + hex.EncodeToString(bytes.Repeat([]byte{0}, 32))
	err := writeAtomicKeepPartial(partial, final, bytes.NewReader(payload), int64(len(payload)), wrong)
	if err == nil {
		t.Fatal("expected digest mismatch")
	}
	if _, err := os.Stat(partial); err != nil {
		t.Fatalf("partial should remain: %v", err)
	}
	if _, err := os.Stat(final); !os.IsNotExist(err) {
		t.Fatal("final should not exist")
	}
}

func TestParallelAssembleSmallWindows(t *testing.T) {
	oldMin, oldChunk := parallelMinSize, parallelChunkSize
	parallelMinSize = 1024
	parallelChunkSize = 512
	defer func() {
		parallelMinSize, parallelChunkSize = oldMin, oldChunk
	}()

	// Unit-level: finalizePartialFile hashes correctly.
	dir := t.TempDir()
	payload := bytes.Repeat([]byte("xy"), 2048)
	sum := sha256.Sum256(payload)
	digest := "sha256-" + hex.EncodeToString(sum[:])
	partial := filepath.Join(dir, "p.partial")
	final := filepath.Join(dir, "p")
	if err := os.WriteFile(partial, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := finalizePartialFile(partial, final, digest); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(final)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("mismatch")
	}
}
