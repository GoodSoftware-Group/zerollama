// Package benchcache persists local model benchmark results for zerollama ls.
//
// WHY a separate package: bench and list both read/write ~/.ollama/bench.json without
// importing the full cmd tree (Cobra, TUI, server hooks). Digest keys keep cache valid
// across tag renames and invalidate automatically when weights change.
//
// WHY aliases: types and Load live in internal/benchcache so the server (auto-router)
// can read perf without importing cmd; this package keeps the writer because it needs
// cmd/internal/fileutil.
package benchcache

import (
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	shared "github.com/ollama/ollama/internal/benchcache"
)

// Entry is one benchmark result keyed by model digest.
type Entry = shared.Entry

// Cache maps model digest to benchmark entry.
type Cache = shared.Cache

// CachePath returns ~/.ollama/bench.json.
func CachePath() (string, error) { return shared.CachePath() }

// Load reads the bench cache. Missing file returns an empty cache.
func Load() (Cache, error) { return shared.Load() }

// Save writes the cache atomically via temp file + rename.
// WHY WriteWithBackup: same durability pattern as ~/.ollama/config.json; partial bench runs must not corrupt JSON.
func Save(c Cache) error {
	if c == nil {
		c = make(Cache) // WHY: nil map marshals to JSON null; ls expects object {}
	}
	path, err := CachePath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}

	return fileutil.WriteWithBackup(path, data)
}
