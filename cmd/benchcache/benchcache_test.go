package benchcache

import (
	"testing"
	"time"
)

func TestEntryPerfString(t *testing.T) {
	if got := (Entry{Kind: "completion", TokPerSec: 42.3}).PerfString(); got != "42.3" {
		t.Fatalf("completion = %q", got)
	}
	if got := (Entry{Kind: "image", GenSec: 33.2}).PerfString(); got != "33s" {
		t.Fatalf("image = %q", got)
	}
	if got := (Entry{Kind: "video_gen", GenSec: 120.7}).PerfString(); got != "121s" {
		t.Fatalf("video = %q", got)
	}
	if got := (Entry{}).PerfString(); got != "--" {
		t.Fatalf("empty = %q", got)
	}
}

func TestEntryCached(t *testing.T) {
	if !(Entry{Kind: "image", GenSec: 10}).Cached() {
		t.Fatal("expected image cached")
	}
	if (Entry{Kind: "completion"}).Cached() {
		t.Fatal("expected completion not cached")
	}
}

func TestLoadSaveRoundTrip(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	cache := Cache{
		"sha256-abc": {
			Model:     "llama3.2:latest",
			Kind:      "completion",
			TokPerSec: 42.3,
			BenchedAt: time.Date(2026, 6, 20, 12, 0, 0, 0, time.UTC),
		},
		"sha256-img": {
			Model:     "sd15-vulkan:latest",
			Kind:      "image",
			GenSec:    33,
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
	if loaded["sha256-img"].GenSec != 33 || loaded["sha256-img"].PerfString() != "33s" {
		t.Fatalf("image entry: %+v", loaded["sha256-img"])
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
