package benchcache

import (
	"os"
	"path/filepath"
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

func writeBench(t *testing.T, home, body string, mod time.Time) {
	t.Helper()
	path := filepath.Join(home, ".ollama", "bench.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
}

func TestSnapshotReloadsOnChange(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	if got := Snapshot(); len(got) != 0 {
		t.Fatalf("missing file: want empty, got %v", got)
	}

	t0 := time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)
	writeBench(t, home, `{"sha256-a":{"model":"a","kind":"completion","tok_per_sec":10}}`, t0)
	if got := Snapshot()["sha256-a"].TokPerSec; got != 10 {
		t.Fatalf("first read: tok/s = %v", got)
	}

	writeBench(t, home, `{"sha256-a":{"model":"a","kind":"completion","tok_per_sec":25}}`, t0.Add(time.Minute))
	if got := Snapshot()["sha256-a"].TokPerSec; got != 25 {
		t.Fatalf("after rewrite: tok/s = %v", got)
	}
}

func TestSnapshotCorruptIsEmpty(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	writeBench(t, home, `{not json`, time.Date(2026, 10, 2, 0, 0, 0, 0, time.UTC))
	if got := Snapshot(); len(got) != 0 {
		t.Fatalf("corrupt file: want empty, got %v", got)
	}
}
