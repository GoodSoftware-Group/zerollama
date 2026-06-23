package benchcache

import (
	"path/filepath"
	"testing"
	"time"
)

func TestLoadSaveRoundTrip(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	cache := Cache{
		"sha256-abc": {
			Model:     "llama3.2:latest",
			TokPerSec: 42.3,
			BenchedAt: time.Date(2026, 6, 20, 12, 0, 0, 0, time.UTC),
		},
	}
	if err := cache.Save(); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	loaded, err := Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	entry, ok := loaded["sha256-abc"]
	if !ok {
		t.Fatal("expected entry for sha256-abc")
	}
	if entry.Model != "llama3.2:latest" || entry.TokPerSec != 42.3 {
		t.Fatalf("unexpected entry: %+v", entry)
	}
}

func TestLoadMissingFile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	cache, err := Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(cache) != 0 {
		t.Fatalf("expected empty cache, got %d entries", len(cache))
	}
}

func TestCachePath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir)

	path, err := CachePath()
	if err != nil {
		t.Fatalf("CachePath() error = %v", err)
	}
	want := filepath.Join(dir, ".ollama", "bench.json")
	if path != want {
		t.Fatalf("CachePath() = %q, want %q", path, want)
	}
}
