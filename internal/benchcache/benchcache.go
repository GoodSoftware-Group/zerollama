// Package benchcache reads local model benchmark results (~/.ollama/bench.json).
//
// WHY a shared internal package: `zerollama bench` (cmd) writes the cache and `zerollama ls`
// reads it, but the auto-router (server) also needs read-only perf to rank candidates
// (docs/auto-router.md §5). server must not import cmd, and the writer depends on
// cmd/internal/fileutil — so types + Load live here and cmd/benchcache keeps Save.
// Digest keys keep the cache valid across tag renames and invalidate when weights change.
package benchcache

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Entry is one benchmark result keyed by model digest.
type Entry struct {
	Model     string    `json:"model"`                 // display name at bench time; not used for lookup
	Kind      string    `json:"kind,omitempty"`        // completion | image | video_gen
	TokPerSec float64   `json:"tok_per_sec,omitempty"` // completion decode only (EvalCount/EvalDuration)
	GenSec    float64   `json:"gen_sec,omitempty"`     // image/video wall seconds per generation
	BenchedAt time.Time `json:"benched_at"`            // audit; future TTL could use this
}

// PerfString formats the ls/bench summary metric for this entry.
// WHY one formatter: ListHandler uses a single PERF column; kind selects tok/s vs seconds.
func (e Entry) PerfString() string {
	switch e.Kind {
	case "image", "video_gen":
		if e.GenSec > 0 {
			return fmt.Sprintf("%.0fs", e.GenSec)
		}
	default:
		if e.TokPerSec > 0 {
			return fmt.Sprintf("%.1f", e.TokPerSec)
		}
	}
	// Legacy entries without kind
	if e.GenSec > 0 {
		return fmt.Sprintf("%.0fs", e.GenSec)
	}
	if e.TokPerSec > 0 {
		return fmt.Sprintf("%.1f", e.TokPerSec)
	}
	return "--"
}

// Cached returns true when this entry has a usable benchmark for its kind.
func (e Entry) Cached() bool {
	switch e.Kind {
	case "image", "video_gen":
		return e.GenSec > 0
	default:
		return e.TokPerSec > 0 || e.GenSec > 0
	}
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
	return loadPath(path)
}

func loadPath(path string) (Cache, error) {
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

var (
	snapMu    sync.Mutex
	snapPath  string
	snapMod   time.Time
	snapSize  int64
	snapCache Cache
)

// Snapshot returns the bench cache, re-reading the file only when its mtime or size changes.
// WHY: the server consults perf on routed requests; re-parsing JSON per request is waste,
// but `zerollama bench` may rewrite the file while serve runs. Missing or corrupt files
// yield an empty cache — perf is a soft score input, never a reason to fail a request.
// The returned map is shared; callers must not mutate it.
func Snapshot() Cache {
	path, err := CachePath()
	if err != nil {
		return Cache{}
	}
	st, err := os.Stat(path)
	if err != nil {
		return Cache{}
	}
	snapMu.Lock()
	defer snapMu.Unlock()
	if snapCache != nil && snapPath == path && st.ModTime().Equal(snapMod) && st.Size() == snapSize {
		return snapCache
	}
	cache, err := loadPath(path)
	if err != nil {
		cache = Cache{}
	}
	snapCache, snapPath, snapMod, snapSize = cache, path, st.ModTime(), st.Size()
	return snapCache
}
