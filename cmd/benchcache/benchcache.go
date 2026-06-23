// Package benchcache persists local model benchmark results for zerollama ls.
//
// WHY a separate package: bench and list both read/write ~/.ollama/bench.json without
// importing the full cmd tree (Cobra, TUI, server hooks). Digest keys keep cache valid
// across tag renames and invalidate automatically when weights change.
package benchcache

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/cmd/internal/fileutil"
)

// Entry is one benchmark result keyed by model digest.
type Entry struct {
	Model     string    `json:"model"`      // display name at bench time; not used for lookup
	TokPerSec float64   `json:"tok_per_sec"` // generation decode only (EvalCount/EvalDuration)
	BenchedAt time.Time `json:"benched_at"`  // audit; future TTL could use this
}

// Cache maps model digest to benchmark entry.
// WHY digest not name: cp/rename keeps one weights blob; re-pull changes digest → stale entry ignored in ls.
type Cache map[string]Entry

// CachePath returns ~/.ollama/bench.json.
func CachePath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "bench.json"), nil
}

// Load reads the bench cache. Missing file returns an empty cache.
// WHY empty not error on ENOENT: ls must work before the operator ever runs bench.
func Load() (Cache, error) {
	path, err := CachePath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return make(Cache), nil
		}
		return nil, err
	}

	var cache Cache
	if err := json.Unmarshal(data, &cache); err != nil {
		return nil, fmt.Errorf("parse bench cache: %w", err)
	}
	if cache == nil {
		cache = make(Cache)
	}
	return cache, nil
}

// Save writes the cache atomically via temp file + rename.
// WHY WriteWithBackup: same durability pattern as ~/.ollama/config.json; partial bench runs must not corrupt JSON.
func (c Cache) Save() error {
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
