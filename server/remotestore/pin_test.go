package remotestore

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPinRefcountBlocksEviction(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	blobs := filepath.Join(dir, "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	d1 := "sha256-" + "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	d2 := "sha256-" + "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	p1 := filepath.Join(blobs, d1)
	p2 := filepath.Join(blobs, d2)
	if err := os.WriteFile(p1, make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p2, make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}

	r := &Resolver{
		CacheDir: dir,
		MaxBytes: 150, // forces eviction of one 100-byte blob
		Mode:     CachePersist,
		pinned:   make(map[string]int),
	}
	r.Pin(d1)
	r.Pin(d1) // second holder (shared layer)
	if err := r.evictIfNeeded(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p1); err != nil {
		t.Fatalf("pinned blob evicted: %v", err)
	}
	if _, err := os.Stat(p2); err == nil {
		t.Fatal("unpinned blob should have been evicted")
	}

	r.Unpin(d1)
	// still pinned once
	_ = os.WriteFile(p2, make([]byte, 100), 0o644)
	if err := r.evictIfNeeded(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p1); err != nil {
		t.Fatalf("still-pinned blob evicted after one Unpin: %v", err)
	}

	r.Unpin(d1)
	// Only p1 remains (~100 bytes). Cap below that so eviction must remove it.
	r.MaxBytes = 50
	if err := r.evictIfNeeded(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p1); err == nil {
		t.Fatal("blob should be evictable after final Unpin")
	}
}

func TestReleaseModelBlobsEphemeral(t *testing.T) {
	t.Parallel()
	scratch := t.TempDir()
	d := "sha256-" + "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	path := filepath.Join(scratch, d)
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	r := &Resolver{
		Mode:       CacheEphemeral,
		ScratchDir: scratch,
		ephemeral:  map[string]string{d: path},
		pinned:     map[string]int{d: 1},
	}
	r.ReleaseModelBlobs(d)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("scratch file should be gone, err=%v", err)
	}
	if r.pinned[d] != 0 {
		t.Fatalf("pin should be cleared, got %d", r.pinned[d])
	}
}
